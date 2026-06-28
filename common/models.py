from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# ==========================
# PATIENT
# ==========================
class Patient(Base):
    __tablename__ = "patient"

    id_patient = Column(String, primary_key=True)
    sexe = Column(String(1))
    tranche_age = Column(String(10))

    observations = relationship("Observation", back_populates="patient")


# ==========================
# OBSERVATION
# ==========================
class Observation(Base):
    __tablename__ = "observation"

    id_observation = Column(Integer, primary_key=True, autoincrement=True)

    id_patient = Column(String, ForeignKey("patient.id_patient"))
    source = Column(String(10))
    duree_symptomes = Column(Float)

    patient = relationship("Patient", back_populates="observations")

    constantes = relationship("Constantes", uselist=False, back_populates="observation")
    symptome = relationship("Symptome", uselist=False, back_populates="observation")
    antecedent = relationship("Antecedent", uselist=False, back_populates="observation")
    decision = relationship("Decision", uselist=False, back_populates="observation")


# ==========================
# CONSTANCES
# ==========================
class Constantes(Base):
    __tablename__ = "constantes"

    id_constantes = Column(Integer, primary_key=True, autoincrement=True)

    id_observation = Column(Integer, ForeignKey("observation.id_observation"))

    freq_cardiaque = Column(Integer)
    tension_sys = Column(Integer)
    temp = Column(Float)
    sat_oxygene = Column(Integer)

    observation = relationship("Observation", back_populates="constantes")


# ==========================
# SYMPTOME (embedding)
# ==========================
class Symptome(Base):
    __tablename__ = "symptome"

    id_symptome = Column(Integer, primary_key=True, autoincrement=True)

    id_observation = Column(Integer, ForeignKey("observation.id_observation"))

    description_embedding = Column(JSON)  # vecteur CamemBERT

    observation = relationship("Observation", back_populates="symptome")


# ==========================
# ANTECEDENT
# ==========================
class Antecedent(Base):
    __tablename__ = "antecedent"

    id_antecedent = Column(Integer, primary_key=True, autoincrement=True)

    id_observation = Column(Integer, ForeignKey("observation.id_observation"))

    antecedents = Column(Boolean)

    observation = relationship("Observation", back_populates="antecedent")


# ==========================
# DECISION
# ==========================
class Decision(Base):
    __tablename__ = "decision"

    id_decision = Column(Integer, primary_key=True, autoincrement=True)

    id_observation = Column(Integer, ForeignKey("observation.id_observation"))

    niveau_urgence = Column(Integer)

    observation = relationship("Observation", back_populates="decision")


# ==========================
# PREDICTION_LOG (inférences API + feedback différé)
# ==========================
class PredictionLog(Base):
    __tablename__ = "prediction_log"

    id_prediction = Column(Integer, primary_key=True, autoincrement=True)

    # Features brutes envoyées à /predict (mêmes colonnes que ml/config.py)
    sexe = Column(String(1))
    tranche_age = Column(String(10))
    source = Column(String(10))
    duree_symptomes = Column(Float)
    freq_cardiaque = Column(Integer)
    tension_sys = Column(Integer)
    temp = Column(Float)
    sat_oxygene = Column(Integer)
    antecedents = Column(Boolean)
    description_texte = Column(String)

    # Résultat de la prédiction
    predicted_niveau_urgence = Column(Integer)
    proba_0 = Column(Float)
    proba_1 = Column(Float)
    proba_2 = Column(Float)
    model_version = Column(String(50))

    # Feedback différé (vraie décision médicale, renseignée plus tard via PATCH /predict/{id}/feedback)
    actual_niveau_urgence = Column(Integer, nullable=True)

    # Passe à True une fois la ligne intégrée dans un dataset de réentraînement (évite les doublons)
    incorporated_in_training = Column(Boolean, default=False, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)


# ==========================
# DRIFT_METRICS (monitoring data/concept drift)
# ==========================
class DriftMetric(Base):
    __tablename__ = "drift_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    computed_at = Column(DateTime, default=datetime.utcnow)

    n_samples = Column(Integer)
    drift_share = Column(Float)  # fraction de features avec drift détecté
    per_feature_drift = Column(JSON)  # {feature: p_value}

    performance_metrics = Column(JSON, nullable=True)  # f1_macro/critical_undertriage_rate si labels dispo
    n_labeled_samples = Column(Integer, nullable=True)

    alert_triggered = Column(Boolean, default=False)
