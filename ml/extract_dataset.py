import sys
from pathlib import Path

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import DATASET_PATH, EMBEDDING_DIM, EMBEDDING_FEATURES, ML_DATABASE_URL  # noqa: E402

QUERY = """
SELECT
    o.id_observation,
    p.sexe,
    p.tranche_age,
    o.source,
    o.duree_symptomes,
    c.freq_cardiaque,
    c.tension_sys,
    c.temp,
    c.sat_oxygene,
    a.antecedents,
    s.description_embedding,
    d.niveau_urgence
FROM observation o
JOIN patient p ON p.id_patient = o.id_patient
LEFT JOIN constantes c ON c.id_observation = o.id_observation
LEFT JOIN symptome s ON s.id_observation = o.id_observation
LEFT JOIN antecedent a ON a.id_observation = o.id_observation
JOIN decision d ON d.id_observation = o.id_observation
"""


def extract_dataset() -> pd.DataFrame:
    if ML_DATABASE_URL is None:
        raise ValueError("❌ ML_DATABASE_URL manquant dans le fichier .env")

    engine = create_engine(ML_DATABASE_URL)

    logger.info("Extraction des observations depuis la base...")
    df = pd.read_sql(QUERY, engine)
    logger.info(f"{len(df)} lignes extraites")

    logger.info(f"Expansion des embeddings CamemBERT ({EMBEDDING_DIM} dimensions)...")
    zero_vector = [0.0] * EMBEDDING_DIM
    embeddings = df["description_embedding"].apply(
        lambda v: v if isinstance(v, list) else zero_vector
    )
    embeddings_df = pd.DataFrame(embeddings.tolist(), columns=EMBEDDING_FEATURES, index=df.index)
    df = pd.concat([df.drop(columns=["description_embedding"]), embeddings_df], axis=1)

    logger.info(f"Répartition des classes : {df['niveau_urgence'].value_counts().sort_index().to_dict()}")

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_PATH, index=False)
    logger.success(f"Dataset sauvegardé dans {DATASET_PATH} ({df.shape[0]} lignes, {df.shape[1]} colonnes)")

    return df


if __name__ == "__main__":
    extract_dataset()
