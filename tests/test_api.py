import numpy as np
from fastapi.testclient import TestClient

from heart_mlop.api import app


class StubModel:
    def predict_proba(self, frame):
        return np.array([[0.2, 0.8]])


def test_prediction_and_metrics():
    patient = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 2,
        "ca": 0,
        "thal": 3,
    }
    with TestClient(app) as client:
        app.state.model = StubModel()
        response = client.post("/predict", json=patient)
        assert response.status_code == 200
        assert response.json()["prediction"] == 1
        assert response.json()["confidence"] == 0.8
        metrics_response = client.get("/metrics")
        assert "heart_api_predictions_total 1" in metrics_response.text


def test_prediction_rejects_extra_fields():
    with TestClient(app) as client:
        app.state.model = StubModel()
        response = client.post("/predict", json={"unknown": 1})
        assert response.status_code == 422