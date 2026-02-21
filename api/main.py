import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

from src.features import FEATURE_COLS

logger = logging.getLogger("inference-api")

APP_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = APP_ROOT / "models" / "latest" / "model.joblib"
META_PATH = APP_ROOT / "models" / "latest" / "metadata.json"

# ---------------------------------------------------------------------------
# Global state — loaded once at startup
# ---------------------------------------------------------------------------
_model = None
_metadata: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
PREDICT_COUNT = Counter("predict_total", "Total prediction requests")
PREDICT_CHURN = Counter("predict_churn_total", "Predictions where churn=1")
PREDICT_LATENCY = Histogram("predict_latency_seconds", "Prediction latency")


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load model once at application startup."""
    global _model, _metadata
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    if META_PATH.exists():
        _metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
        logger.info("Metadata loaded: version=%s", _metadata.get("model_version"))
    yield


app = FastAPI(
    title="Local MLOps Inference API",
    version="2.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    age: float
    tenure_months: float
    monthly_charges: float
    contract_type: int
    num_tickets: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_version": _metadata.get("model_version"),
    }


@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    if not _metadata:
        raise HTTPException(status_code=404, detail="No model metadata available")
    return _metadata


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type="text/plain")


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = pd.DataFrame(
        [
            {
                "age": req.age,
                "tenure_months": req.tenure_months,
                "monthly_charges": req.monthly_charges,
                "contract_type": req.contract_type,
                "num_tickets": req.num_tickets,
            }
        ],
        columns=FEATURE_COLS,
    )

    try:
        start = time.perf_counter()
        pred = int(_model.predict(X)[0])
        proba = None
        if hasattr(_model, "predict_proba"):
            proba = float(_model.predict_proba(X)[0][1])
        latency = time.perf_counter() - start

        PREDICT_COUNT.inc()
        PREDICT_LATENCY.observe(latency)
        if pred == 1:
            PREDICT_CHURN.inc()

        return {"churn_prediction": pred, "churn_probability": proba}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
