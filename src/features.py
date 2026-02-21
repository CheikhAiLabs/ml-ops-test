import pandas as pd

from src.schemas import churn_schema

FEATURE_COLS = [
    "age",
    "tenure_months",
    "monthly_charges",
    "contract_type",
    "num_tickets",
]
TARGET_COL = "churn"


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = set(FEATURE_COLS + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {sorted(missing)}")
    df = churn_schema.validate(df)
    return df


def split_xy(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(int).copy()
    return X, y
