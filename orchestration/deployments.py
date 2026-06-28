import sys
from pathlib import Path

from prefect import serve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestration.flows import drift_check_flow, retrain_flow  # noqa: E402

if __name__ == "__main__":
    drift_deployment = drift_check_flow.to_deployment(
        name="drift-check-weekly",
        cron="0 6 * * 1",  # tous les lundis 06h00
    )
    retrain_deployment = retrain_flow.to_deployment(
        name="retrain-monthly-safety-net",
        cron="0 6 1 * *",  # le 1er de chaque mois 06h00
    )
    serve(drift_deployment, retrain_deployment)
