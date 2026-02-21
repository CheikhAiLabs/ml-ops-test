"""Model behavior tests — ensure the trained model makes sensible predictions."""

import pandas as pd
import pytest

from src.features import FEATURE_COLS, engineer_features, load_dataset, split_xy
from src.train import build_model


def _make_raw(**overrides):
    """Return engineered feature DataFrame from raw feature overrides."""
    base = {
        "gender": 0,
        "age": 35,
        "partner": 0,
        "dependents": 0,
        "tenure_months": 12,
        "monthly_charges": 55.0,
        "contract_type": 1,
        "payment_method": 1,
        "paperless_billing": 0,
        "internet_service": 1,
        "online_security": 0,
        "tech_support": 0,
        "num_tickets": 2,
    }
    base.update(overrides)
    df = engineer_features(pd.DataFrame([base]))
    return df[FEATURE_COLS]


@pytest.fixture(scope="module")
def model():
    """Train a fresh model for behavior tests (independent of disk state)."""
    from src.config import DATA_RAW

    df = load_dataset(str(DATA_RAW))
    X, y = split_xy(df)
    pipe = build_model()
    pipe.fit(X, y)
    return pipe


class TestModelBehavior:
    """Sanity-check predictions on archetypal customers."""

    def test_high_risk_customer(self, model):
        """Month-to-month, short tenure, many tickets → high churn risk."""
        X = _make_raw(
            age=22,
            tenure_months=1,
            monthly_charges=95.0,
            contract_type=0,
            internet_service=2,
            online_security=0,
            tech_support=0,
            payment_method=2,
            paperless_billing=1,
            num_tickets=5,
        )
        proba = float(model.predict_proba(X)[0][1])
        assert proba > 0.4, f"Expected high risk, got {proba:.2f}"

    def test_low_risk_customer(self, model):
        """Long contract, long tenure, no tickets → low churn risk."""
        X = _make_raw(
            age=50,
            tenure_months=48,
            monthly_charges=45.0,
            contract_type=2,
            internet_service=1,
            online_security=1,
            tech_support=1,
            payment_method=0,
            paperless_billing=0,
            partner=1,
            dependents=1,
            num_tickets=0,
        )
        proba = float(model.predict_proba(X)[0][1])
        assert proba < 0.6, f"Expected low risk, got {proba:.2f}"

    def test_prediction_is_binary(self, model):
        X = _make_raw()
        pred = int(model.predict(X)[0])
        assert pred in (0, 1)

    def test_proba_in_range(self, model):
        X = _make_raw()
        proba = float(model.predict_proba(X)[0][1])
        assert 0.0 <= proba <= 1.0
