import argparse
import sys
from pathlib import Path

import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_MODEL_NAME, MLFLOW_TRACKING_URI  # noqa: E402


def find_run_id(run_name: str) -> str:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Expérience MLflow introuvable : {MLFLOW_EXPERIMENT_NAME}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"Aucun run trouvé avec le nom : {run_name}")
    return runs[0].info.run_id


def promote(run_id: str, artifact_path: str = "model") -> None:
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"

    logger.info(f"Enregistrement du modèle depuis {model_uri} sous le nom '{MLFLOW_MODEL_NAME}'...")
    model_version = mlflow.register_model(model_uri, MLFLOW_MODEL_NAME)

    logger.info(f"Version {model_version.version} enregistrée. Transition vers le stage 'Production'...")
    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True,
    )
    logger.success(
        f"Modèle '{MLFLOW_MODEL_NAME}' version {model_version.version} (run {run_id}) promu en Production."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=None, help="Run ID MLflow à promouvoir")
    parser.add_argument(
        "--run_name",
        default="RF_IMPORTANCE_20__LGBMClassifier__tuned",
        help="Nom du run à rechercher si --run_id n'est pas fourni",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    run_id = args.run_id or find_run_id(args.run_name)
    promote(run_id)


if __name__ == "__main__":
    main()
