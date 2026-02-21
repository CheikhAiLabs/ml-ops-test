"""Model behavior tests — ensure the trained model makes sensible predictions."""

import pandas as pd
import pytest

from src.features import FEATURE_COLS, load_dataset, split_xy
from src.train import build_model


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

    def _predict(self, model, row: dict) -> int:
        X = pd.DataFrame([row], columns=FEATURE_COLS)
        return int(model.predict(X)[0])

    def _proba(self, model, row: dict) -> float:
        X = pd.DataFrame([row], columns=FEATURE_COLS)
        return float(model.predict_proba(X)[0][1])

    def test_high_risk_customer(self, model):
        """Month-to-month, short tenure, many tickets → high churn risk."""
        proba = self._proba(
            model,
            {
                "age": 22,
                "tenure_months": 1,
                "monthly_charges": 89.9,
                "contract_type": 0,
                "num_tickets": 5,
            },
        )
        assert proba > 0.4, f"Expected high risk, got {proba:.2f}"

    def test_low_risk_customer(self, model):
        """Long contract, long tenure, no tickets → low churn risk."""
        proba = self._proba(
            model,
            {
                "age": 50,
                "tenure_months": 48,
                "monthly_charges": 45.0,
                "contract_type": 2,
                "num_tickets": 0,
            },
        )
        assert proba < 0.6, f"Expected low risk, got {proba:.2f}"

    def test_prediction_is_binary(self, model):
        pred = self._predict(
            model,
            {
                "age": 35,
                "tenure_months": 12,
                "monthly_charges": 55.0,
                "contract_type": 1,
                "num_tickets": 2,
            },
        )
        assert pred in (0, 1)

    def test_proba_in_range(self, model):
        proba = self._proba(
            model,
            {
                "age": 35,
                "tenure_months": 12,
                "monthly_charges": 55.0,
                "contract_type": 1,
                "num_tickets": 2,
            },
        )
        assert 0.0 <= proba <= 1.0
