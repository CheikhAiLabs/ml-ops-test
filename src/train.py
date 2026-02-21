import argparse
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    ARTIFACTS_DIR,
    DATA_RAW,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_VERSIONS_DIR,
    RANDOM_STATE,
)
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

    # MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        model = build_model()
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        f1 = float(f1_score(y_val, val_pred))

        data_hash = sha256_file(data_path)
        model_fingerprint = sha256_str(f"{data_hash}:{RANDOM_STATE}:logreg:v1")

        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 200)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", 0.25)
        mlflow.log_param("data_hash", data_hash)
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("n_features", len(FEATURE_COLS))

        # Log metrics
        mlflow.log_metric("val_f1", f1)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("val_samples", len(X_val))

        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_train.head(1),
        )

        # Save model locally (backward compat)
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
            "mlflow_run_id": mlflow.active_run().info.run_id,
        }
        write_json(version_dir / "metadata.json", metadata)
        write_json(report_dir / "train_report.json", metadata)

        # Log metadata as artifact
        mlflow.log_artifact(str(version_dir / "metadata.json"))

        print(f"Saved model to: {model_path}")
        print(f"Validation F1: {f1:.4f}")
        print(f"Model version: {model_fingerprint}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
