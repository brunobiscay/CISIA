import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ==========================
# PATHS
# ==========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "dataset.parquet"

ML_DATABASE_URL = os.getenv("ML_DATABASE_URL")

# ==========================
# COLONNES
# ==========================
TARGET = "niveau_urgence"
ID_COLUMNS = ["id_observation"]

NUMERIC_FEATURES = [
    "freq_cardiaque",
    "tension_sys",
    "temp",
    "sat_oxygene",
    "duree_symptomes",
]

CATEGORICAL_FEATURES = [
    "sexe",
    "tranche_age",
    "source",
    "antecedents",
]

EMBEDDING_DIM = 768
EMBEDDING_PREFIX = "emb_"
EMBEDDING_FEATURES = [f"{EMBEDDING_PREFIX}{i}" for i in range(EMBEDDING_DIM)]

# ==========================
# FEATURE SETS
# ==========================
SELECT_KBEST_K = 20
RF_IMPORTANCE_K = 20

FEATURE_SETS = [
    "ALL_FEATURES",
    f"SELECT_KBEST_{SELECT_KBEST_K}",
    f"RF_IMPORTANCE_{RF_IMPORTANCE_K}",
]

# ==========================
# MODELES
# ==========================
SKLEARN_MODELS = ["LogisticRegression", "RandomForestClassifier", "LGBMClassifier", "OrdinalLogistic"]
SAMPLE_WEIGHT_MODELS = {"OrdinalLogistic"}  # pas de class_weight natif → sample_weight passé au fit
NEURAL_NET_MODEL = "NeuralNet"
MODELS = SKLEARN_MODELS + [NEURAL_NET_MODEL]

# ==========================
# VALIDATION
# ==========================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
NEURAL_NET_SEEDS = [42, 43, 44]

# ==========================
# MLFLOW
# ==========================
MLFLOW_EXPERIMENT_NAME = "triage_urgence_classification"
CLASS_LABELS = [0, 1, 2]

# ==========================
# OPTUNA (tuning des meilleurs candidats)
# ==========================
TUNE_FEATURE_SET = f"RF_IMPORTANCE_{RF_IMPORTANCE_K}"
TUNE_MODELS = ["RandomForestClassifier", "LGBMClassifier", NEURAL_NET_MODEL]
OPTUNA_N_TRIALS = {
    "RandomForestClassifier": 30,
    "LGBMClassifier": 30,
    NEURAL_NET_MODEL: 15,
}
