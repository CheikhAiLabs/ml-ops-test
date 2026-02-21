"""Unit tests for data validation and feature module."""

import pandas as pd
import pandera.errors
import pytest

from src.features import (
    FEATURE_COLS,
    TARGET_COL,
    engineer_features,
    load_dataset,
    split_xy,
)
from src.schemas import churn_schema


def _make_row(**overrides):
    """Return a single-row DataFrame with all required schema columns."""
    base = {
        "gender": 0,
        "age": 30,
        "partner": 0,
        "dependents": 0,
        "tenure_months": 12,
        "monthly_charges": 49.9,
        "contract_type": 1,
        "payment_method": 1,
        "paperless_billing": 0,
        "internet_service": 1,
        "online_security": 0,
        "tech_support": 0,
        "num_tickets": 1,
        "churn": 0,
    }
    base.update(overrides)
    return pd.DataFrame([base])


class TestSchema:
    """Pandera schema validation tests."""

    def test_valid_data(self):
        df = pd.concat(
            [_make_row(age=25, churn=1), _make_row(age=40)], ignore_index=True
        )
        validated = churn_schema.validate(df)
        assert len(validated) == 2

    def test_invalid_age_too_low(self):
        df = _make_row(age=5)
        with pytest.raises(pandera.errors.SchemaError):
            churn_schema.validate(df)

    def test_invalid_contract_type(self):
        df = _make_row(contract_type=5)
        with pytest.raises(pandera.errors.SchemaError):
            churn_schema.validate(df)

    def test_invalid_churn_value(self):
        df = _make_row(churn=2)
        with pytest.raises(pandera.errors.SchemaError):
            churn_schema.validate(df)

    def test_extra_column_rejected(self):
        df = _make_row()
        df["extra"] = 1
        with pytest.raises(pandera.errors.SchemaError):
            churn_schema.validate(df)


class TestFeatures:
    """Feature splitting tests."""

    def test_split_xy_columns(self):
        df = _make_row(churn=1)
        df = engineer_features(df)
        X, y = split_xy(df)
        assert list(X.columns) == FEATURE_COLS
        assert len(y) == 1
        assert y.iloc[0] == 1

    def test_load_real_dataset(self):
        df = load_dataset("data/raw/churn.csv")
        assert len(df) > 100
        assert TARGET_COL in df.columns
        for col in FEATURE_COLS:
            assert col in df.columns
