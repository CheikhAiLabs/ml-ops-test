import argparse
from pathlib import Path

import joblib
import mlflow
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import (
    ARTIFACTS_DIR,
    DATA_RAW,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
)
from src.features import load_dataset, split_xy
from src.utils import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--data", type=str, default=str(DATA_RAW))
    parser.add_argument("--report-dir", type=str, default=str(ARTIFACTS_DIR))
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_path = model_dir / "model.joblib"
    meta_path = model_dir / "metadata.json"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError("model.joblib or metadata.json not found in model-dir")

    model = joblib.load(model_path)
    meta = read_json(meta_path)

    df = load_dataset(args.data)
    X, y = split_xy(df)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    f1 = float(f1_score(y_test, pred))
    accuracy = float(accuracy_score(y_test, pred))
    precision = float(precision_score(y_test, pred))
    recall = float(recall_score(y_test, pred))
    roc_auc = float(roc_auc_score(y_test, proba))
    report = classification_report(y_test, pred, output_dict=True)

    payload = {
        "model_version": meta["model_version"],
        "test_f1": f1,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_roc_auc": roc_auc,
        "classification_report": report,
    }

    out_path = Path(args.report_dir) / "eval_report.json"
    write_json(out_path, payload)

    # Log evaluation metrics to MLflow (reuse training run if available)
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        train_run_id = meta.get("mlflow_run_id")
        if train_run_id:
            # Continue the training run with eval metrics
            with mlflow.start_run(run_id=train_run_id):
                mlflow.log_metric("eval_f1", f1)
                mlflow.log_metric("eval_accuracy", accuracy)
                mlflow.log_metric("eval_precision", precision)
                mlflow.log_metric("eval_recall", recall)
                mlflow.log_metric("eval_roc_auc", roc_auc)
                mlflow.log_artifact(str(out_path))
                mlflow.set_tag("evaluation.status", "completed")
                print(f"✅ Eval metrics logged to MLflow run {train_run_id}")
        else:
            # Create a new evaluation run
            with mlflow.start_run(run_name="churn-evaluation"):
                mlflow.log_metric("eval_f1", f1)
                mlflow.log_metric("eval_accuracy", accuracy)
                mlflow.log_metric("eval_precision", precision)
                mlflow.log_metric("eval_recall", recall)
                mlflow.log_metric("eval_roc_auc", roc_auc)
                mlflow.log_param("model_version", meta["model_version"])
                mlflow.log_artifact(str(out_path))
                mlflow.set_tag("evaluation.status", "completed")
                print("✅ Eval metrics logged to new MLflow run")
    except Exception as exc:
        print(f"⚠️  MLflow eval logging skipped: {exc}")

    print(f"Eval F1: {f1:.4f}")
    print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()
