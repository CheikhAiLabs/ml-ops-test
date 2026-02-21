import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    Summary,
    generate_latest,
)
from pydantic import BaseModel
from starlette.responses import Response

from src.features import FEATURE_COLS, RAW_FEATURE_COLS, engineer_features

# MLflow is optional — won't crash the API if unavailable
try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

logger = logging.getLogger("inference-api")

APP_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = APP_ROOT / "models" / "latest" / "model.joblib"
META_PATH = APP_ROOT / "models" / "latest" / "metadata.json"

# ---------------------------------------------------------------------------
# Global state — loaded once at startup
# ---------------------------------------------------------------------------
_model = None
_metadata: Dict[str, Any] = {}
_feature_means: Dict[str, float] = {}

# ---------------------------------------------------------------------------
# Prometheus metrics — rich instrumentation
# ---------------------------------------------------------------------------
# Request-level
HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

# Prediction-level
PREDICT_COUNT = Counter("predict_total", "Total prediction requests")
PREDICT_CHURN = Counter("predict_churn_total", "Predictions where churn=1")
PREDICT_NO_CHURN = Counter("predict_no_churn_total", "Predictions where churn=0")
PREDICT_LATENCY = Histogram(
    "predict_latency_seconds",
    "Model inference latency (model only)",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)
PREDICT_ERRORS = Counter("predict_errors_total", "Failed prediction requests")
PREDICT_PROBABILITY = Summary(
    "predict_probability",
    "Distribution of churn probabilities",
)

# Feature-level (input monitoring)
FEATURE_AGE = Histogram(
    "feature_age",
    "Distribution of age feature",
    buckets=[20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
)
FEATURE_TENURE = Histogram(
    "feature_tenure_months",
    "Distribution of tenure_months feature",
    buckets=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60],
)
FEATURE_CHARGES = Histogram(
    "feature_monthly_charges",
    "Distribution of monthly_charges feature",
    buckets=[20, 30, 40, 50, 60, 70, 80, 90, 100],
)
FEATURE_TICKETS = Histogram(
    "feature_num_tickets",
    "Distribution of num_tickets feature",
    buckets=[0, 1, 2, 3, 4, 5, 6, 7, 8],
)
FEATURE_CONTRACT = Counter(
    "feature_contract_type_total",
    "Count of contract types seen",
    ["contract_type"],
)
FEATURE_INTERNET = Counter(
    "feature_internet_service_total",
    "Count of internet service types seen",
    ["internet_service"],
)

# System-level
MODEL_INFO = Info("model", "Currently loaded model metadata")
MODEL_LOADED = Gauge("model_loaded", "Whether a model is currently loaded")
APP_UPTIME = Gauge("app_start_time_seconds", "Timestamp when app started")

# MLflow tracing config
MLFLOW_TRACING_ENABLED = False


def _mlflow_server_reachable(uri: str, timeout: float = 2.0) -> bool:
    """Quick check if MLflow server is reachable (non-blocking)."""
    if not uri.startswith("http"):
        return False
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=timeout)
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load model once at application startup."""
    global _model, _metadata, _feature_means, MLFLOW_TRACING_ENABLED
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
        MODEL_LOADED.set(1)
    else:
        MODEL_LOADED.set(0)

    if META_PATH.exists():
        _metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
        logger.info("Metadata loaded: version=%s", _metadata.get("model_version"))
        MODEL_INFO.info(
            {
                "version": str(_metadata.get("model_version", "unknown")),
                "type": str(_metadata.get("model_type", "unknown")),
                "f1": str(_metadata.get("val_f1", "unknown")),
            }
        )
        # Load feature means for contribution computation
        _feature_means = _metadata.get("feature_means", {})

    APP_UPTIME.set_to_current_time()

    # Configure MLflow tracing — controlled by ENABLE_MLFLOW_TRACING env var
    tracing_requested = os.getenv("ENABLE_MLFLOW_TRACING", "true").lower() == "true"
    if tracing_requested and _MLFLOW_AVAILABLE:
        uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            _metadata.get("mlflow_uri", "http://localhost:5000"),
        )
        if _mlflow_server_reachable(uri):
            try:
                mlflow.set_tracking_uri(uri)
                mlflow.set_experiment("churn-classifier")
                MLFLOW_TRACING_ENABLED = True
                logger.info("MLflow tracing enabled at %s", uri)
            except Exception as exc:
                logger.warning("MLflow tracing setup failed: %s", exc)
        else:
            logger.info("MLflow server not reachable at %s — tracing disabled", uri)
    elif not tracing_requested:
        logger.info("MLflow tracing disabled via ENABLE_MLFLOW_TRACING=false")
    else:
        logger.info("MLflow not installed — tracing disabled")

    yield


app = FastAPI(
    title="ChurnGuard Inference API",
    version="3.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Static files & UI
# ---------------------------------------------------------------------------
STATIC_DIR = APP_ROOT / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_ui():
    """Serve the ChurnGuard frontend."""
    return FileResponse(str(STATIC_DIR / "index.html"))


class PredictRequest(BaseModel):
    gender: int
    age: float
    partner: int
    dependents: int
    tenure_months: float
    monthly_charges: float
    contract_type: int
    payment_method: int
    paperless_billing: int
    internet_service: int
    online_security: int
    tech_support: int
    num_tickets: float


# ---------------------------------------------------------------------------
# Middleware — instrument ALL HTTP requests
# ---------------------------------------------------------------------------
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    endpoint = request.url.path
    method = request.method
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = str(response.status_code)
    except Exception:
        status = "500"
        raise
    finally:
        duration = time.perf_counter() - start
        HTTP_REQUESTS.labels(method=method, endpoint=endpoint, status=status).inc()
        HTTP_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
    return response


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

    # Build raw features dict (what the user controls)
    raw_features: Dict[str, Any] = {
        "gender": req.gender,
        "age": req.age,
        "partner": req.partner,
        "dependents": req.dependents,
        "tenure_months": req.tenure_months,
        "monthly_charges": req.monthly_charges,
        "contract_type": req.contract_type,
        "payment_method": req.payment_method,
        "paperless_billing": req.paperless_billing,
        "internet_service": req.internet_service,
        "online_security": req.online_security,
        "tech_support": req.tech_support,
        "num_tickets": req.num_tickets,
    }

    # Engineer features & build model input
    X = engineer_features(pd.DataFrame([raw_features]))
    X = X[FEATURE_COLS]

    # Observe input feature distributions
    FEATURE_AGE.observe(req.age)
    FEATURE_TENURE.observe(req.tenure_months)
    FEATURE_CHARGES.observe(req.monthly_charges)
    FEATURE_TICKETS.observe(req.num_tickets)
    FEATURE_CONTRACT.labels(contract_type=str(req.contract_type)).inc()
    FEATURE_INTERNET.labels(internet_service=str(req.internet_service)).inc()

    try:
        # Run inference inside MLflow trace if enabled, otherwise directly
        if MLFLOW_TRACING_ENABLED:
            result, latency = _predict_with_trace(X, raw_features)
        else:
            result, latency = _run_inference(X)

        # Compute per-feature contributions (explainability)
        contributions = _compute_contributions(raw_features)
        result["feature_contributions"] = contributions
        result["churn_rate_baseline"] = _metadata.get("churn_rate", 0.33)

        # Prometheus counters
        PREDICT_COUNT.inc()
        PREDICT_LATENCY.observe(latency)
        if result["churn_prediction"] == 1:
            PREDICT_CHURN.inc()
        else:
            PREDICT_NO_CHURN.inc()
        if result["churn_probability"] is not None:
            PREDICT_PROBABILITY.observe(result["churn_probability"])

        return result
    except Exception as e:
        PREDICT_ERRORS.inc()
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


def _run_inference(X: pd.DataFrame) -> tuple:
    """Run model prediction and return (result_dict, latency_seconds)."""
    assert _model is not None  # noqa: S101
    start = time.perf_counter()
    pred = int(_model.predict(X)[0])
    proba = None
    if hasattr(_model, "predict_proba"):
        proba = float(_model.predict_proba(X)[0][1])
    latency = time.perf_counter() - start
    return {"churn_prediction": pred, "churn_probability": proba}, latency


def _predict_with_trace(X: pd.DataFrame, features: Dict[str, Any]) -> tuple:
    """Run inference wrapped in an MLflow trace — captures real latency."""
    try:
        with mlflow.start_span(name="churn-prediction", span_type="CHAIN") as root:
            root.set_inputs(features)
            root.set_attributes(
                {
                    "model_version": _metadata.get("model_version", ""),
                    "model_type": _metadata.get("model_type", ""),
                }
            )

            with mlflow.start_span(name="model-inference", span_type="LLM") as span:
                span.set_inputs({"n_features": X.shape[1], "n_samples": X.shape[0]})
                result, latency = _run_inference(X)
                span.set_outputs(result)
                span.set_attributes({"latency_ms": round(latency * 1000, 2)})

            root.set_outputs(result)
        return result, latency
    except Exception as exc:
        logger.debug("MLflow trace failed, running without trace: %s", exc)
        return _run_inference(X)


def _compute_contributions(
    raw_features: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Leave-one-out feature attribution.

    For each raw feature, swap it to its training-set mean, re-engineer
    derived features, and measure the probability change.  This tells the
    user *which features drive this specific prediction*.
    """
    assert _model is not None  # noqa: S101
    if not _feature_means:
        return []

    # Current probability
    X_current = engineer_features(pd.DataFrame([raw_features]))[FEATURE_COLS]
    current_proba = float(_model.predict_proba(X_current)[0][1])

    contributions: List[Dict[str, Any]] = []
    for feature in RAW_FEATURE_COLS:
        if feature not in _feature_means:
            continue
        modified = raw_features.copy()
        modified[feature] = _feature_means[feature]
        X_mod = engineer_features(pd.DataFrame([modified]))[FEATURE_COLS]
        mod_proba = float(_model.predict_proba(X_mod)[0][1])
        contrib = current_proba - mod_proba
        if abs(contrib) > 0.003:  # Only include meaningful contributions
            contributions.append(
                {
                    "feature": feature,
                    "value": raw_features[feature],
                    "contribution": round(contrib, 4),
                }
            )

    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions[:8]
