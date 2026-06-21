import hashlib
import hmac
import math
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.models import (  # noqa: E402
    Antecedent,
    Base,
    Constantes,
    Decision,
    Observation,
    Patient,
    Symptome,
)

load_dotenv()


# ==========================
# CONFIG
# ==========================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / "dataset_telemed.csv"
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL is None:
    raise ValueError("❌ DATABASE_URL manquant dans le fichier .env")

BATCH_SIZE = 200

MODEL_NAME = "almanach/camembert-bio-base"  # modèle FR médical


# ==========================
# LOAD MODEL
# ==========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()


# ==========================
# UTILS RGPD
# ==========================
PATIENT_ID_SALT = os.getenv("PATIENT_ID_SALT")

if PATIENT_ID_SALT is None:
    raise ValueError("❌ PATIENT_ID_SALT manquant dans le fichier .env")


def hash_patient_id(raw_id):
    return hmac.new(
        PATIENT_ID_SALT.encode(), str(raw_id).encode(), hashlib.sha256
    ).hexdigest()


def age_to_tranche(age):
    if pd.isna(age) or age is None:
        return "unknown"  # ou "NA"

    age = int(age)

    if age < 18:
        return "0-18"
    elif age < 40:
        return "18-40"
    elif age < 65:
        return "40-65"
    else:
        return "65+"


def clean_text(text):
    if pd.isna(text):
        return "aucun symptome"

    text = str(text)
    text = re.sub(r"\d+", "", text)

    return text.strip()


# ==========================
# CAMEMBERT EMBEDDING
# ==========================
def get_embedding(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # mean pooling
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

    return embedding.tolist()


def safe_int(value):
    try:
        val = float(value)
        if math.isnan(val):
            return None
        return int(val)
    except (TypeError, ValueError):
        return None


def safe_float(value):
    try:
        val = float(value)
        if math.isnan(val):
            return None
        return val
    except (TypeError, ValueError):
        return None


def safe_str(value):
    if pd.isna(value):
        return None
    return str(value)


# ==========================
# IMPORT
# ==========================
def import_csv():

    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    if session.query(Patient).first() is not None:
        print("ℹ️ Données déjà importées, import ignoré.")
        session.close()
        return

    df = pd.read_csv(csv_path)

    print(f"📊 {len(df)} lignes à traiter")

    patients_cache = {}

    for i, (_, row) in enumerate(df.iterrows(), 1):
        # ===== PATIENT =====
        patient_hash = hash_patient_id(row["patient_id"])

        if patient_hash not in patients_cache:
            patient = Patient(
                id_patient=patient_hash,
                sexe=safe_str(row["sexe"]),
                tranche_age=age_to_tranche(row["age"]),
            )
            session.add(patient)
            patients_cache[patient_hash] = patient

        # ===== OBSERVATION =====
        observation = Observation(
            id_patient=patient_hash,
            source=safe_str(row["source"]),
            duree_symptomes=safe_float(row["duree_symptomes"]),
        )

        session.add(observation)
        session.flush()

        obs_id = observation.id_observation

        # ===== CONSTANCES =====
        session.add(
            Constantes(
                id_observation=obs_id,
                freq_cardiaque=safe_int((row["freq_cardiaque"])),
                tension_sys=safe_int(row["tension_sys"]),
                temp=safe_float(row["temp"]),
                sat_oxygene=safe_int((row["sat_oxygene"])),
            )
        )

        # ===== EMBEDDING CAMEMBERT =====
        cleaned_text = clean_text(row["description_symptomes"])
        embedding = get_embedding(cleaned_text)

        session.add(Symptome(id_observation=obs_id, description_embedding=embedding))

        # ===== ANTECEDENT =====
        session.add(
            Antecedent(
                id_observation=obs_id, antecedents=safe_int((float(row["antecedents"])))
            )
        )

        # ===== DECISION =====
        session.add(
            Decision(id_observation=obs_id, niveau_urgence=int(row["niveau_urgence"]))
        )

        # batch commit
        if i % BATCH_SIZE == 0:
            session.commit()
            print(f"✅ {i} lignes")

    session.commit()
    session.close()

    print("🎉 Import terminé avec CamemBERT-bio !")


if __name__ == "__main__":
    import_csv()
