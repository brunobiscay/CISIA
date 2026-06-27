FROM python:3.11

WORKDIR /app

COPY mlflow/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD mlflow server --host 0.0.0.0 --port 5000 \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root "mlflow-artifacts:/" \
    --artifacts-destination "$MLFLOW_ARTIFACT_ROOT" \
    --allowed-hosts "mlflow,mlflow:5000,localhost,localhost:5000"
