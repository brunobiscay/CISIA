import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.schema import (  # noqa: E402, F401 (réexportés pour ml/preprocessing.py, ml/extract_dataset.py, ...)
    CATEGORICAL_FEATURES,
    CLASS_LABELS,
    EMBEDDING_DIM,
    EMBEDDING_FEATURES,
    EMBEDDING_PREFIX,
    NUMERIC_FEATURES,
    TARGET,
)

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
ID_COLUMNS = ["id_observation"]

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
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = "triage_urgence_classification"
MLFLOW_MODEL_NAME = "triage_urgence_model"

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

# ==========================
# MODELE DE PRODUCTION (combo retenu pour le serving + réentraînement)
# ==========================
PRODUCTION_FEATURE_SET = f"RF_IMPORTANCE_{RF_IMPORTANCE_K}"
PRODUCTION_MODEL_NAME = "LGBMClassifier"
# Meilleurs hyperparams trouvés par Optuna pour RF_IMPORTANCE_20__LGBMClassifier__tuned
PRODUCTION_MODEL_PARAMS = {
    "n_estimators": 450,
    "learning_rate": 0.0056828375585122656,
    "num_leaves": 19,
    "max_depth": 10,
    "min_child_samples": 47,
    "reg_alpha": 0.003077180271250686,
    "reg_lambda": 0.09565499215943825,
}
