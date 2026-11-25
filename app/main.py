from __future__ import annotations
"""FastAPI prediction service for MP129 — Tiffin Relay Delivery Time (Patna)."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, Field

from tiffin.config import load_params


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON strings on stdout."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, dict):
            payload.update(extra_fields)
        return json.dumps(payload)


def _setup_logger() -> logging.Logger:
    log = logging.getLogger("mp129.api")
    log.setLevel(logging.INFO)
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        log.addHandler(handler)
    log.propagate = False
    return log


logger = _setup_logger()
app = FastAPI(title="MP129 Tiffin Relay Delivery Time API", version="1.0.0")

_MODEL_CACHE: Any = None


class PredictionRequest(BaseModel):
    """Exact input schema for POST /predict with Patna Dabba Relay Union rules."""

    distance_km: float = Field(..., description="Route length in km")
    tiffins: float = Field(
        ...,
        ge=2.5,
        description="Number of tiffins in the batch (values below 2.5 are rejected)",
    )
    handovers: int = Field(..., description="Number of relay handovers")
    wait_at_station_min: float = Field(..., description="Waiting time at station in minutes")
    rider_age: float = Field(..., description="Age of main runner in years")
    temp_c: float = Field(
        ...,
        le=75.0,
        description="Noon temperature in deg C (values above 75 are rejected)",
    )
    transport: str = Field(..., description="Main transport (bicycle, local_train, scooter)")
    route_zone: str = Field(..., description="Route zone (Central, Suburb, Old City)")


class PredictionResponse(BaseModel):
    prediction: float


def get_model() -> Any:
    """Load and cache the champion model from params.yaml model_path."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    import joblib  # local import so app module imports cleanly even without joblib in lightweight checks

    params = load_params()
    model_path = Path(params["paths"]["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Champion model not found at {model_path}. Run the training pipeline first."
        )
    _MODEL_CACHE = joblib.load(model_path)
    return _MODEL_CACHE


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    row = pd.DataFrame([req.model_dump()])
    pred_value = round(float(model.predict(row)[0]), 4)

    logger.info(
        "prediction_served",
        extra={
            "extra_fields": {
                "event": "predict",
                "route_zone": req.route_zone,
                "transport": req.transport,
                "prediction": pred_value,
            }
        },
    )
    return PredictionResponse(prediction=pred_value)
