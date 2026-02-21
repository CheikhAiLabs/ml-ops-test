import argparse
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.data.pandas_dataset
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
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
from src.features import FEATURE_COLS, RAW_FEATURE_COLS, load_dataset, split_xy
from src.utils import ensure_dir, sha256_file, sha256_str, write_json


def build_model() -> Pipeline:
    numeric_features = [
        "age",
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "ticket_rate",
    ]
    categorical_features = [
        "gender",
        "partner",
        "dependents",
        "senior_citizen",
        "contract_type",
        "payment_method",
        "paperless_billing",
        "internet_service",
        "online_security",
        "tech_support",
        "num_tickets",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", "passthrough", categorical_features),
        ]
    )

    clf = GradientBoostingClassifier(random_state=RANDOM_STATE)
    return Pipeline(steps=[("prep", preprocessor), ("clf", clf)])


# Hyperparameter grid for tuning
PARAM_GRID = {
    "clf__n_estimators": [200, 300, 500],
    "clf__max_depth": [3, 5, 7],
    "clf__learning_rate": [0.05, 0.1],
    "clf__subsample": [0.8, 1.0],
    "clf__min_samples_leaf": [5, 10],
}


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
        # ── Log dataset to MLflow ──────────────────────
        dataset = mlflow.data.pandas_dataset.from_pandas(
            df, source=str(data_path), name="churn", targets="churn"
        )
        mlflow.log_input(dataset, context="training")

        # ── GitHub CI/CD tags ─────────────────────────
        github_env = {
            "github.repository": os.getenv("GITHUB_REPOSITORY", ""),
            "github.ref": os.getenv("GITHUB_REF", ""),
            "github.sha": os.getenv("GITHUB_SHA", ""),
            "github.run_id": os.getenv("GITHUB_RUN_ID", ""),
            "github.run_url": (
                f"https://github.com/{os.getenv('GITHUB_REPOSITORY', '')}"
                f"/actions/runs/{os.getenv('GITHUB_RUN_ID', '')}"
                if os.getenv("GITHUB_RUN_ID")
                else ""
            ),
            "github.actor": os.getenv("GITHUB_ACTOR", ""),
            "github.event_name": os.getenv("GITHUB_EVENT_NAME", ""),
        }
        for tag_key, tag_val in github_env.items():
            if tag_val:
                mlflow.set_tag(tag_key, tag_val)

        pipe = build_model()

        # Hyperparameter tuning via 5-fold cross-validation
        print("Running hyperparameter search (GridSearchCV 5-fold) ...")
        grid = GridSearchCV(
            pipe,
            PARAM_GRID,
            cv=5,
            scoring="f1",
            n_jobs=1,
            verbose=1,
        )
        grid.fit(X_train, y_train)

        model = grid.best_estimator_
        best_params = grid.best_params_
        cv_f1 = float(grid.best_score_)

        print(f"Best CV F1: {cv_f1:.4f}")
        print(f"Best params: {best_params}")

        # Validation metrics
        val_pred = model.predict(X_val)
        val_proba = model.predict_proba(X_val)[:, 1]

        f1 = float(f1_score(y_val, val_pred))
        accuracy = float(accuracy_score(y_val, val_pred))
        precision = float(precision_score(y_val, val_pred))
        recall = float(recall_score(y_val, val_pred))
        roc_auc = float(roc_auc_score(y_val, val_proba))

        data_hash = sha256_file(data_path)
        model_fingerprint = sha256_str(f"{data_hash}:{RANDOM_STATE}:gbm:{best_params}")

        # Log parameters
        mlflow.log_param("model_type", "GradientBoostingClassifier")
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", 0.25)
        mlflow.log_param("data_hash", data_hash)
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("n_features", len(FEATURE_COLS))
        for k, v in best_params.items():
            mlflow.log_param(k.replace("clf__", ""), v)

        # Log metrics
        mlflow.log_metric("cv_f1", cv_f1)
        mlflow.log_metric("val_f1", f1)
        mlflow.log_metric("val_accuracy", accuracy)
        mlflow.log_metric("val_precision", precision)
        mlflow.log_metric("val_recall", recall)
        mlflow.log_metric("val_roc_auc", roc_auc)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("val_samples", len(X_val))

        # Log model (best-effort — may fail with remote tracking servers)
        try:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                input_example=X_train.head(1),
            )
        except Exception as exc:
            print(f"⚠️  mlflow.sklearn.log_model skipped: {exc}")

        # Save model locally (backward compat)
        version_dir = out_dir / model_fingerprint
        ensure_dir(version_dir)

        model_path = version_dir / "model.joblib"
        joblib.dump(model, model_path)

        # Feature means for explainability (leave-one-out contributions)
        feature_means = {
            col: round(float(X_train[col].mean()), 4) for col in RAW_FEATURE_COLS
        }

        metadata = {
            "model_version": model_fingerprint,
            "model_type": "GradientBoostingClassifier",
            "best_params": best_params,
            "data_hash": data_hash,
            "random_state": RANDOM_STATE,
            "cv_f1": cv_f1,
            "val_f1": f1,
            "val_accuracy": accuracy,
            "val_precision": precision,
            "val_recall": recall,
            "val_roc_auc": roc_auc,
            "feature_cols": FEATURE_COLS,
            "raw_feature_cols": RAW_FEATURE_COLS,
            "feature_means": feature_means,
            "churn_rate": round(float(y.mean()), 4),
            "mlflow_run_id": mlflow.active_run().info.run_id,
        }
        write_json(version_dir / "metadata.json", metadata)
        write_json(report_dir / "train_report.json", metadata)

        # Log metadata as artifact (best-effort)
        try:
            mlflow.log_artifact(str(version_dir / "metadata.json"))
        except Exception as exc:
            print(f"⚠️  mlflow.log_artifact skipped: {exc}")

        print(f"Saved model to: {model_path}")
        print(f"Validation F1: {f1:.4f}")
        print(f"Model version: {model_fingerprint}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
