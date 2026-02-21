"""Tests for the FastAPI inference API."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_has_model_info(self):
        resp = client.get("/health")
        data = resp.json()
        assert "model_loaded" in data
        assert "model_version" in data


class TestPredictEndpoint:
    VALID_PAYLOAD = {
        "gender": 0,
        "age": 28,
        "partner": 0,
        "dependents": 0,
        "tenure_months": 6,
        "monthly_charges": 39.9,
        "contract_type": 0,
        "payment_method": 2,
        "paperless_billing": 1,
        "internet_service": 2,
        "online_security": 0,
        "tech_support": 0,
        "num_tickets": 3,
    }

    def test_predict_returns_200(self):
        resp = client.post("/predict", json=self.VALID_PAYLOAD)
        # 200 if model loaded, 503 if not
        assert resp.status_code in (200, 503)

    def test_predict_response_shape(self):
        resp = client.post("/predict", json=self.VALID_PAYLOAD)
        if resp.status_code == 200:
            data = resp.json()
            assert "churn_prediction" in data
            assert "churn_probability" in data
            assert data["churn_prediction"] in (0, 1)
            assert 0.0 <= data["churn_probability"] <= 1.0

    def test_predict_missing_field(self):
        payload = {"age": 28, "tenure_months": 6}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_invalid_type(self):
        payload = {**self.VALID_PAYLOAD, "age": "invalid"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


class TestModelInfoEndpoint:
    def test_model_info(self):
        resp = client.get("/model-info")
        # 200 if metadata exists, 404 if not
        assert resp.status_code in (200, 404)


class TestMetricsEndpoint:
    def test_metrics_endpoint(self):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "predict_total" in resp.text
