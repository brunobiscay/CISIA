import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.tracking
import pandas as pd
from loguru import logger
from prefect import flow, task
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (  # noqa: E402
    BASE_DIR,
    DATASET_PATH,
    ID_COLUMNS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    PRODUCTION_FEATURE_SET,
    PRODUCTION_MODEL_NAME,
    PRODUCTION_MODEL_PARAMS,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from ml.monitoring import main as run_drift_monitoring  # noqa: E402
from ml.prepare_production_data import prepare_batch  # noqa: E402
from ml.promote_model import promote  # noqa: E402
from ml.train import build_pipeline, evaluate_pipeline  # noqa: E402

RETRAIN_LOOKBACK_DAYS = 90


@task(name="check_drift")
def check_drift_task(lookback_days: int = 7):
    return run_drift_monitoring(lookback_days=lookback_days)


@flow(name="drift_check_flow")
def drift_check_flow():
    record = check_drift_task()

    if record.drift_share is None:
        logger.info("Pas assez de données de production pour calculer le drift.")
        return

    if record.alert_triggered:
        logger.warning(
            f"Drift détecté (drift_share={record.drift_share:.3f}) -> déclenchement de retrain_flow"
        )
        retrain_flow()
    else:
        logger.success(f"Pas de dérive significative (drift_share={record.drift_share:.3f}).")


@task(name="build_new_dataset")
def build_new_dataset_task() -> pd.DataFrame | None:
    since = datetime.now(timezone.utc) - timedelta(days=RETRAIN_LOOKBACK_DAYS)
    new_df = prepare_batch(since)

    if new_df.empty:
        logger.warning("Aucune donnée de production labellisée disponible : réentraînement annulé.")
        return None

    base_df = pd.read_parquet(DATASET_PATH)
    combined = pd.concat([base_df, new_df], ignore_index=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot_path = DATASET_PATH.parent / f"dataset_v{timestamp}.parquet"
    combined.to_parquet(snapshot_path, index=False)
    combined.to_parquet(DATASET_PATH, index=False)

    logger.success(
        f"Nouveau dataset : {len(combined)} lignes ({len(new_df)} nouvelles) -> {snapshot_path}"
    )

    subprocess.run(["dvc", "add", str(DATASET_PATH)], check=True, cwd=BASE_DIR)
    subprocess.run(["dvc", "push"], check=True, cwd=BASE_DIR)
    logger.success(f"{DATASET_PATH.name} versionné avec DVC et poussé vers le remote.")

    return combined


@task(name="retrain_and_log")
def retrain_and_log_task(df: pd.DataFrame) -> tuple[str, dict]:
    feature_columns = [c for c in df.columns if c not in ID_COLUMNS + [TARGET]]
    X, y = df[feature_columns], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline(PRODUCTION_FEATURE_SET, PRODUCTION_MODEL_NAME, model_params=PRODUCTION_MODEL_PARAMS)
    pipeline.fit(X_train, y_train)
    _, _, test_metrics = evaluate_pipeline(pipeline, X_test, y_test)

    run_name = (
        f"{PRODUCTION_FEATURE_SET}__{PRODUCTION_MODEL_NAME}__retrained_"
        f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("feature_set", PRODUCTION_FEATURE_SET)
        mlflow.log_param("model", PRODUCTION_MODEL_NAME)
        mlflow.log_param("retrained", True)
        mlflow.log_params({f"model__{k}": v for k, v in PRODUCTION_MODEL_PARAMS.items()})
        mlflow.log_metrics(test_metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        run_id = run.info.run_id

    logger.success(
        f"[{run_name}] f1_macro={test_metrics['f1_macro']:.4f} "
        f"critical_undertriage_rate={test_metrics['critical_undertriage_rate']:.4f}"
    )
    return run_id, test_metrics


@task(name="get_current_production_metrics")
def get_current_production_metrics_task() -> dict | None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
    if not versions:
        return None
    run = client.get_run(versions[0].run_id)
    return dict(run.data.metrics)


@flow(name="retrain_flow")
def retrain_flow():
    combined_df = build_new_dataset_task()
    if combined_df is None:
        return

    new_run_id, new_metrics = retrain_and_log_task(combined_df)
    current_metrics = get_current_production_metrics_task()

    new_critical_rate = new_metrics["critical_undertriage_rate"]
    current_critical_rate = current_metrics.get("critical_undertriage_rate") if current_metrics else None

    if current_critical_rate is None or new_critical_rate <= current_critical_rate:
        logger.success(
            f"Nouveau modèle au moins aussi bon (critical_undertriage_rate "
            f"{new_critical_rate:.4f} <= {current_critical_rate}) -> promotion en Production"
        )
        promote(new_run_id)
    else:
        logger.warning(
            f"Nouveau modèle moins bon (critical_undertriage_rate "
            f"{new_critical_rate:.4f} > {current_critical_rate:.4f}) -> pas de promotion"
        )


if __name__ == "__main__":
    drift_check_flow()
