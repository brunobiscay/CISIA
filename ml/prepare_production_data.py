import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.embeddings import compute_embedding  # noqa: E402
from common.schema import CATEGORICAL_FEATURES, EMBEDDING_FEATURES, NUMERIC_FEATURES, TARGET  # noqa: E402
from ml.config import ML_DATABASE_URL  # noqa: E402

# Plages plausibles pour filtrer les valeurs aberrantes/erreurs de saisie
NUMERIC_RANGES = {
    "freq_cardiaque": (0, 300),
    "tension_sys": (0, 300),
    "temp": (25.0, 45.0),
    "sat_oxygene": (0, 100),
    "duree_symptomes": (0, None),
}


def load_labeled_predictions(since: datetime) -> pd.DataFrame:
    engine = create_engine(ML_DATABASE_URL)
    query = """
        SELECT id_prediction, sexe, tranche_age, source, antecedents, duree_symptomes,
               freq_cardiaque, tension_sys, temp, sat_oxygene, description_texte,
               actual_niveau_urgence, created_at
        FROM prediction_log
        WHERE actual_niveau_urgence IS NOT NULL
          AND incorporated_in_training = FALSE
          AND created_at >= %(since)s
    """
    return pd.read_sql(query, engine, params={"since": since})


def mark_incorporated(id_predictions: list[int]) -> None:
    if not id_predictions:
        return
    engine = create_engine(ML_DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE prediction_log SET incorporated_in_training = TRUE WHERE id_prediction = ANY(:ids)"),
            {"ids": list(id_predictions)},
        )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = NUMERIC_FEATURES + ["description_texte", "sexe", "tranche_age", "source"]
    df = df.dropna(subset=required_cols)

    for col, (low, high) in NUMERIC_RANGES.items():
        if low is not None:
            df = df[df[col] >= low]
        if high is not None:
            df = df[df[col] <= high]

    return df


def prepare_batch(since: datetime) -> pd.DataFrame:
    raw = load_labeled_predictions(since)
    logger.info(f"{len(raw)} prédictions labellisées trouvées depuis {since.date()}")
    if raw.empty:
        return raw

    cleaned = clean(raw)
    logger.info(f"{len(cleaned)} lignes valides après nettoyage/validation")
    if cleaned.empty:
        return cleaned

    embeddings = cleaned["description_texte"].apply(compute_embedding)
    embeddings_df = pd.DataFrame(embeddings.tolist(), columns=EMBEDDING_FEATURES, index=cleaned.index)

    result = pd.concat([cleaned, embeddings_df], axis=1)
    # IDs négatifs : distinguables des id_observation réels (toujours positifs, autoincrement)
    result["id_observation"] = -result["id_prediction"]
    result[TARGET] = result["actual_niveau_urgence"].astype(int)

    mark_incorporated(cleaned["id_prediction"].tolist())

    columns = ["id_observation"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES + EMBEDDING_FEATURES + [TARGET]
    return result[columns].reset_index(drop=True)


if __name__ == "__main__":
    df = prepare_batch(datetime.now(timezone.utc) - timedelta(days=90))
    logger.success(f"Batch préparé : {df.shape}")
