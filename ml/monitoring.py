import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.models import DriftMetric  # noqa: E402
from common.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES  # noqa: E402
from ml.config import DATASET_PATH, ML_DATABASE_URL  # noqa: E402
from ml.metrics import classification_metrics  # noqa: E402

MONITORED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
REFERENCE_SAMPLE_SIZE = 2000


def load_reference_data() -> pd.DataFrame:
    df = pd.read_parquet(DATASET_PATH, columns=MONITORED_FEATURES)
    if len(df) > REFERENCE_SAMPLE_SIZE:
        df = df.sample(REFERENCE_SAMPLE_SIZE, random_state=42)
    return df.reset_index(drop=True)


def load_production_data(since: datetime) -> pd.DataFrame:
    engine = create_engine(ML_DATABASE_URL)
    query = """
        SELECT sexe, tranche_age, source, antecedents, duree_symptomes,
               freq_cardiaque, tension_sys, temp, sat_oxygene,
               predicted_niveau_urgence, actual_niveau_urgence
        FROM prediction_log
        WHERE created_at >= %(since)s
    """
    return pd.read_sql(query, engine, params={"since": since})


def compute_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    data_definition = DataDefinition(
        numerical_columns=NUMERIC_FEATURES,
        categorical_columns=CATEGORICAL_FEATURES,
    )
    reference_ds = Dataset.from_pandas(reference_df[MONITORED_FEATURES], data_definition=data_definition)
    current_ds = Dataset.from_pandas(current_df[MONITORED_FEATURES], data_definition=data_definition)

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=reference_ds, current_data=current_ds).dict()

    drift_share = None
    per_feature_drift = {}
    for metric in result["metrics"]:
        if metric["config"]["type"] == "evidently:metric_v2:DriftedColumnsCount":
            drift_share = metric["value"]["share"]
        elif metric["config"]["type"] == "evidently:metric_v2:ValueDrift":
            per_feature_drift[metric["config"]["column"]] = metric["value"]

    return {"drift_share": drift_share, "per_feature_drift": per_feature_drift}


def compute_performance_drift(current_df: pd.DataFrame) -> dict | None:
    labeled = current_df.dropna(subset=["actual_niveau_urgence"])
    if labeled.empty:
        return None

    metrics = classification_metrics(
        labeled["actual_niveau_urgence"].astype(int),
        labeled["predicted_niveau_urgence"].astype(int),
    )
    return {"n_labeled_samples": len(labeled), "metrics": metrics}


def persist(drift_result: dict, performance_result: dict | None, n_samples: int) -> DriftMetric:
    engine = create_engine(ML_DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    alert_triggered = drift_result["drift_share"] is not None and drift_result["drift_share"] > DRIFT_THRESHOLD

    record = DriftMetric(
        n_samples=n_samples,
        drift_share=drift_result["drift_share"],
        per_feature_drift=drift_result["per_feature_drift"],
        performance_metrics=performance_result["metrics"] if performance_result else None,
        n_labeled_samples=performance_result["n_labeled_samples"] if performance_result else None,
        alert_triggered=alert_triggered,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    session.close()
    return record


def main(lookback_days: int = 7) -> DriftMetric:
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    logger.info("Chargement des données de référence (train)...")
    reference_df = load_reference_data()

    logger.info(f"Chargement des prédictions de production depuis {since.date()}...")
    current_df = load_production_data(since)

    if current_df.empty:
        logger.warning("Aucune prédiction de production sur la fenêtre — drift non calculable.")
        drift_result = {"drift_share": None, "per_feature_drift": {}}
        performance_result = None
    else:
        logger.info(f"{len(current_df)} prédictions de production chargées.")
        drift_result = compute_data_drift(reference_df, current_df)
        performance_result = compute_performance_drift(current_df)

    record = persist(drift_result, performance_result, n_samples=len(current_df))

    logger.info(f"drift_share={record.drift_share}")
    if performance_result:
        logger.info(
            f"Performance sur {performance_result['n_labeled_samples']} échantillons labellisés : "
            f"f1_macro={performance_result['metrics']['f1_macro']:.4f} - "
            f"critical_undertriage_rate={performance_result['metrics']['critical_undertriage_rate']:.4f}"
        )
    else:
        logger.info("Pas de feedback labellisé disponible — performance drift non calculée.")

    if record.alert_triggered:
        logger.warning(f"⚠️ ALERTE DRIFT : drift_share={record.drift_share:.3f} > seuil {DRIFT_THRESHOLD}")
    else:
        logger.success("Pas de dérive significative détectée.")

    return record


if __name__ == "__main__":
    main()
