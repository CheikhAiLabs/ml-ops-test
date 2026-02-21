from pathlib import Path
from typing import Any, Dict

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

APP_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = APP_ROOT / "models" / "latest" / "model.joblib"
META_PATH = APP_ROOT / "models" / "latest" / "metadata.json"

app = FastAPI(title="Local MLOps Inference API", version="1.0.0")

class PredictRequest(BaseModel):
    age: float
    tenure_months: float
    monthly_charges: float
    contract_type: int
    num_tickets: float

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_present": MODEL_PATH.exists(), "meta_present": META_PATH.exists()}

@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    try:
        model = load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    X = [[
        req.age,
        req.tenure_months,
        req.monthly_charges,
        req.contract_type,
        req.num_tickets,
    ]]

    try:
        pred = int(model.predict(X)[0])
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0][1])
        return {"churn_prediction": pred, "churn_probability": proba}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
