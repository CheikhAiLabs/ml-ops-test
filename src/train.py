import argparse
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ARTIFACTS_DIR, DATA_RAW, MODELS_VERSIONS_DIR, RANDOM_STATE
from src.features import FEATURE_COLS, load_dataset, split_xy
from src.utils import ensure_dir, sha256_file, sha256_str, write_json

def build_model() -> Pipeline:
    numeric_features = ["age", "tenure_months", "monthly_charges", "num_tickets"]
    categorical_features = ["contract_type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", "passthrough", categorical_features),
        ]
    )

    clf = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)
    return Pipeline(steps=[("prep", preprocessor), ("clf", clf)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DATA_RAW))
    parser.add_argument("--out-dir", type=str, default=str(MODELS_VERSIONS_DIR))
    parser.add_argument("--report-dir", type=str, default=str(ARTIFACTS_DIR))
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)

    ensure_dir(out_dir)
    ensure_dir(report_dir)

    df = load_dataset(str(data_path))
    X, y = split_xy(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    f1 = float(f1_score(y_val, val_pred))

    data_hash = sha256_file(data_path)
    model_fingerprint = sha256_str(f"{data_hash}:{RANDOM_STATE}:logreg:v1")
    version_dir = out_dir / model_fingerprint
    ensure_dir(version_dir)

    model_path = version_dir / "model.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "model_version": model_fingerprint,
        "data_hash": data_hash,
        "random_state": RANDOM_STATE,
        "val_f1": f1,
        "feature_cols": FEATURE_COLS,
    }
    write_json(version_dir / "metadata.json", metadata)
    write_json(report_dir / "train_report.json", metadata)

    print(f"Saved model to: {model_path}")
    print(f"Validation F1: {f1:.4f}")
    print(f"Model version: {model_fingerprint}")

if __name__ == "__main__":
    main()
