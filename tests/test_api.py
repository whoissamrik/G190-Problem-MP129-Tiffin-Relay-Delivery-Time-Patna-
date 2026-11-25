"""API tests using FastAPI TestClient (no external CSV dependency)."""

import numpy as np
from fastapi.testclient import TestClient

from app import main as api_main


class _StubModel:
    """Lightweight deterministic regressor stub for isolated API unit tests."""

    def predict(self, df):
        return np.full(len(df), 42.5, dtype=float)


def _valid_payload() -> dict:
    return {
        "distance_km": 10.5,
        "tiffins": 25.0,
        "handovers": 3,
        "wait_at_station_min": 12.0,
        "rider_age": 34.0,
        "temp_c": 32.5,
        "transport": "scooter",
        "route_zone": "Central",
    }


def test_health_endpoint():
    client = TestClient(api_main.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_valid_request(monkeypatch):
    monkeypatch.setattr(api_main, "get_model", lambda: _StubModel())
    client = TestClient(api_main.app)
    resp = client.post("/predict", json=_valid_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], float)


def test_predict_rejects_temp_above_75(monkeypatch):
    monkeypatch.setattr(api_main, "get_model", lambda: _StubModel())
    client = TestClient(api_main.app)
    payload = _valid_payload()
    payload["temp_c"] = 76.0
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_tiffins_below_2_5(monkeypatch):
    monkeypatch.setattr(api_main, "get_model", lambda: _StubModel())
    client = TestClient(api_main.app)
    payload = _valid_payload()
    payload["tiffins"] = 2.0
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
