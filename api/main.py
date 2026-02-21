from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.features import FEATURE_COLS

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

    X = pd.DataFrame([{
        "age": req.age,
        "tenure_months": req.tenure_months,
        "monthly_charges": req.monthly_charges,
        "contract_type": req.contract_type,
        "num_tickets": req.num_tickets,
    }], columns=FEATURE_COLS)

    try:
        pred = int(model.predict(X)[0])
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0][1])
        return {"churn_prediction": pred, "churn_probability": proba}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
