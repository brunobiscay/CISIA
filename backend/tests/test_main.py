from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from common.schema import EMBEDDING_DIM

VALID_PAYLOAD = {
    "sexe": "H",
    "tranche_age": "adulte",
    "source": "ambulance",
    "antecedents": False,
    "duree_symptomes": 1.0,
    "freq_cardiaque": 80,
    "tension_sys": 120,
    "temp": 37.0,
    "sat_oxygene": 98,
    "description_texte": "douleur legere",
}


@pytest.fixture
def client():
    fake_model = MagicMock()
    fake_model.predict_proba.return_value = np.array([[0.7, 0.2, 0.1]])
    fake_version = MagicMock(version="1", run_id="fake_run_id")

    with (
        patch("mlflow.tracking.MlflowClient") as mock_client_cls,
        patch("mlflow.sklearn.load_model", return_value=fake_model),
    ):
        mock_client_cls.return_value.get_latest_versions.return_value = [fake_version]

        import main

        with patch.object(main, "compute_embedding", return_value=[0.0] * EMBEDDING_DIM):
            with TestClient(main.app) as test_client:
                yield test_client


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "API running (SQLite)"}


def test_predict_missing_api_key(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 422


def test_predict_wrong_api_key(client):
    response = client.post(
        "/predict", json=VALID_PAYLOAD, headers={"X-API-Key": "wrong-key"}
    )
    assert response.status_code == 401


def test_predict_success(client):
    response = client.post(
        "/predict", json=VALID_PAYLOAD, headers={"X-API-Key": "test-key"}
    )
    assert response.status_code == 200

    body = response.json()
    assert body["niveau_urgence"] == 0
    assert set(body["probabilities"].keys()) == {"0", "1", "2"}
    assert "id_prediction" in body


def test_feedback_updates_prediction(client):
    predict_response = client.post(
        "/predict", json=VALID_PAYLOAD, headers={"X-API-Key": "test-key"}
    )
    id_prediction = predict_response.json()["id_prediction"]

    feedback_response = client.patch(
        f"/predict/{id_prediction}/feedback",
        json={"actual_niveau_urgence": 2},
        headers={"X-API-Key": "test-key"},
    )
    assert feedback_response.status_code == 200
    assert feedback_response.json() == {"status": "ok", "id_prediction": id_prediction}


def test_feedback_not_found(client):
    response = client.patch(
        "/predict/999999/feedback",
        json={"actual_niveau_urgence": 1},
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 404
