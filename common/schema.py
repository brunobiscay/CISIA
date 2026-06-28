TARGET = "niveau_urgence"

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

CLASS_LABELS = [0, 1, 2]
