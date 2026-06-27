import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.tracking
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.embeddings import clean_text, compute_embedding  # noqa: E402
from common.models import Base, PredictionLog  # noqa: E402
from common.schema import CATEGORICAL_FEATURES, EMBEDDING_FEATURES, NUMERIC_FEATURES  # noqa: E402
from database import SessionLocal, engine  # noqa: E402
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

app = FastAPI()

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + EMBEDDING_FEATURES

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "triage_urgence_model")

MODEL = None
MODEL_VERSION = None


# ==========================
# API KEY SECURITY
# ==========================
API_KEY = os.getenv("API_KEY")


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")


# ==========================
# DB SESSION
# ==========================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==========================
# SCHEMAS
# ==========================
class PredictRequest(BaseModel):
    sexe: str
    tranche_age: str
    source: str
    antecedents: bool
    duree_symptomes: float
    freq_cardiaque: int
    tension_sys: int
    temp: float
    sat_oxygene: int
    description_texte: str


class FeedbackRequest(BaseModel):
    actual_niveau_urgence: int


# ==========================
# ROUTES
# ==========================


@app.on_event("startup")
def create_tables():
    print("Creating tables...")
    Base.metadata.create_all(bind=engine)


@app.on_event("startup")
def load_model():
    global MODEL, MODEL_VERSION
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError(f"Aucune version 'Production' trouvée pour le modèle '{MLFLOW_MODEL_NAME}'")
    version = versions[0]

    # mlflow.sklearn.load_model("models:/name/Production") ne résout pas correctement
    # le proxy d'artifacts HTTP avec les "Logged Models" (mlflow >= 3) ; on charge donc
    # directement via le run d'origine, qui fonctionne de façon fiable à travers le réseau docker.
    model_uri = f"runs:/{version.run_id}/model"
    logger.info(f"Chargement du modèle '{MLFLOW_MODEL_NAME}' v{version.version} depuis {model_uri}...")
    MODEL = mlflow.sklearn.load_model(model_uri)
    MODEL_VERSION = f"{MLFLOW_MODEL_NAME}/v{version.version}"
    logger.success(f"Modèle chargé : {MODEL_VERSION}")


@app.get("/")
def root():
    return {"status": "API running (SQLite)"}


@app.get("/patients", dependencies=[Depends(verify_api_key)])
def get_patients(db: Session = Depends(get_db)):
    result = db.execute(text("SELECT * FROM patient"))
    return [dict(row._mapping) for row in result]


@app.get("/observations", dependencies=[Depends(verify_api_key)])
def get_observations(limit: int = 10, db: Session = Depends(get_db)):
    result = db.execute(
        text("SELECT * FROM observation LIMIT :limit"), {"limit": limit}
    )
    return [dict(row._mapping) for row in result]


@app.get("/urgence_stats", dependencies=[Depends(verify_api_key)])
def get_stats(db: Session = Depends(get_db)):
    result = db.execute(
        text("""
        SELECT niveau_urgence, COUNT(*)
        FROM decision
        GROUP BY niveau_urgence
    """)
    )
    return [dict(row._mapping) for row in result]


@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(payload: PredictRequest, db: Session = Depends(get_db)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    embedding = compute_embedding(clean_text(payload.description_texte))

    row = {
        "sexe": payload.sexe,
        "tranche_age": payload.tranche_age,
        "source": payload.source,
        "antecedents": payload.antecedents,
        "duree_symptomes": payload.duree_symptomes,
        "freq_cardiaque": payload.freq_cardiaque,
        "tension_sys": payload.tension_sys,
        "temp": payload.temp,
        "sat_oxygene": payload.sat_oxygene,
    }
    for i, value in enumerate(embedding):
        row[f"emb_{i}"] = value

    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    proba = MODEL.predict_proba(X)[0]
    predicted = int(np.argmax(proba))

    log = PredictionLog(
        sexe=payload.sexe,
        tranche_age=payload.tranche_age,
        source=payload.source,
        duree_symptomes=payload.duree_symptomes,
        freq_cardiaque=payload.freq_cardiaque,
        tension_sys=payload.tension_sys,
        temp=payload.temp,
        sat_oxygene=payload.sat_oxygene,
        antecedents=payload.antecedents,
        description_texte=payload.description_texte,
        predicted_niveau_urgence=predicted,
        proba_0=float(proba[0]),
        proba_1=float(proba[1]),
        proba_2=float(proba[2]),
        model_version=MODEL_VERSION,
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    return {
        "id_prediction": log.id_prediction,
        "niveau_urgence": predicted,
        "probabilities": {"0": float(proba[0]), "1": float(proba[1]), "2": float(proba[2])},
        "model_version": MODEL_VERSION,
    }


@app.patch("/predict/{id_prediction}/feedback", dependencies=[Depends(verify_api_key)])
def feedback(id_prediction: int, payload: FeedbackRequest, db: Session = Depends(get_db)):
    log = db.query(PredictionLog).filter(PredictionLog.id_prediction == id_prediction).first()
    if log is None:
        raise HTTPException(status_code=404, detail="Prédiction introuvable")

    log.actual_niveau_urgence = payload.actual_niveau_urgence
    db.commit()

    return {"status": "ok", "id_prediction": id_prediction}
