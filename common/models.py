from sqlalchemy import JSON, Boolean, Column, Float, ForeignKey, Integer, String
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
