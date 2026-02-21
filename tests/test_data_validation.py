"""Unit tests for data validation and feature module."""

import pandas as pd
import pandera.errors
import pytest

from src.features import FEATURE_COLS, TARGET_COL, load_dataset, split_xy
from src.schemas import churn_schema


class TestSchema:
    """Pandera schema validation tests."""

    def test_valid_data(self):
        df = pd.DataFrame(
            {
                "age": [25, 40],
                "tenure_months": [6, 24],
                "monthly_charges": [29.9, 79.9],
                "contract_type": [0, 1],
                "num_tickets": [1, 0],
                "churn": [1, 0],
            }
        )
        validated = churn_schema.validate(df)
        assert len(validated) == 2

    def test_invalid_age_too_low(self):
        df = pd.DataFrame(
            {
                "age": [5],
                "tenure_months": [6],
                "monthly_charges": [29.9],
                "contract_type": [0],
                "num_tickets": [1],
                "churn": [1],
            }
        )
        with pytest.raises(pandera.errors.SchemaError):
            churn_schema.validate(df)

    def test_invalid_contract_type(self):
        df = pd.DataFrame(
            {
                "age": [30],
                "tenure_months": [12],
                "monthly_charges": [49.9],
                "contract_type": [5],
                "num_tickets": [0],
                "churn": [0],
            }
        )
        with pytest.raises(pandera.errors.SchemaError):
            churn_schema.validate(df)

    def test_invalid_churn_value(self):
        df = pd.DataFrame(
            {
                "age": [30],
                "tenure_months": [12],
                "monthly_charges": [49.9],
                "contract_type": [1],
                "num_tickets": [0],
                "churn": [2],
            }
        )
        with pytest.raises(pandera.errors.SchemaError):
            churn_schema.validate(df)

    def test_extra_column_rejected(self):
        df = pd.DataFrame(
            {
                "age": [30],
                "tenure_months": [12],
                "monthly_charges": [49.9],
                "contract_type": [1],
                "num_tickets": [0],
                "churn": [0],
                "extra": [1],
            }
        )
        with pytest.raises(pandera.errors.SchemaError):
            churn_schema.validate(df)


class TestFeatures:
    """Feature splitting tests."""

    def test_split_xy_columns(self):
        df = pd.DataFrame(
            {
                "age": [25],
                "tenure_months": [6],
                "monthly_charges": [29.9],
                "contract_type": [0],
                "num_tickets": [1],
                "churn": [1],
            }
        )
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
