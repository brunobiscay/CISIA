import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.models import Base  # noqa: E402
from database import SessionLocal, engine
from fastapi import Depends, FastAPI, Header, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

app = FastAPI()


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
# ROUTES
# ==========================


@app.on_event("startup")
def create_tables():
    print("Creating tables...")
    Base.metadata.create_all(bind=engine)


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
