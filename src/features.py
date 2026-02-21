import pandas as pd

from src.schemas import churn_schema

# ── Raw features (user-provided inputs) ──────────────────────
RAW_FEATURE_COLS = [
    "gender",
    "age",
    "partner",
    "dependents",
    "tenure_months",
    "monthly_charges",
    "contract_type",
    "payment_method",
    "paperless_billing",
    "internet_service",
    "online_security",
    "tech_support",
    "num_tickets",
]

# ── Engineered features (computed from raw) ──────────────────
ENGINEERED_COLS = [
    "senior_citizen",
    "total_charges",
    "ticket_rate",
]

# ── All model features ──────────────────────────────────────
FEATURE_COLS = RAW_FEATURE_COLS + ENGINEERED_COLS
TARGET_COL = "churn"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features from raw inputs."""
    df = df.copy()
    df["senior_citizen"] = (df["age"] >= 65).astype(int)
    df["total_charges"] = (df["tenure_months"] * df["monthly_charges"]).round(2)
    df["ticket_rate"] = (
        df["num_tickets"] / df["tenure_months"].clip(lower=1) * 12
    ).round(4)
    return df


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = set(RAW_FEATURE_COLS + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {sorted(missing)}")
    df = churn_schema.validate(df)
    df = engineer_features(df)
    return df


def split_xy(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(int).copy()
    return X, y
