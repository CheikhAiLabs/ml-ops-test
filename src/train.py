import argparse
import os
import platform
import sys
import tempfile
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.data.pandas_dataset
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
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

matplotlib.use("Agg")  # Non-interactive backend for CI


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
# Full grid: 3 × 3 × 2 × 2 × 2 = 72 candidates (local / production)
_PARAM_GRID_FULL = {
    "clf__n_estimators": [200, 300, 500],
    "clf__max_depth": [3, 5, 7],
    "clf__learning_rate": [0.05, 0.1],
    "clf__subsample": [0.8, 1.0],
    "clf__min_samples_leaf": [5, 10],
}

# Lighter grid: 2 × 2 × 1 × 1 × 1 = 4 candidates (CI / fast mode)
_PARAM_GRID_CI = {
    "clf__n_estimators": [200, 300],
    "clf__max_depth": [3, 5],
    "clf__learning_rate": [0.1],
    "clf__subsample": [0.8],
    "clf__min_samples_leaf": [5],
}

# Auto-select: CI env var is set by GitHub Actions
PARAM_GRID = _PARAM_GRID_CI if os.getenv("CI") else _PARAM_GRID_FULL


def _log_confusion_matrix(y_true, y_pred, artifact_dir: Path):
    """Save and log confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Retain", "Churn"], cmap="Blues", ax=ax
    )
    ax.set_title("Confusion Matrix (Validation Set)")
    path = artifact_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(path), artifact_path="plots")


def _log_roc_curve(model, X_val, y_val, artifact_dir: Path):
    """Save and log ROC curve plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_val, y_val, ax=ax, name="GBM")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random baseline")
    ax.set_title("ROC Curve (Validation Set)")
    ax.legend()
    path = artifact_dir / "roc_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(path), artifact_path="plots")


def _log_precision_recall_curve(model, X_val, y_val, artifact_dir: Path):
    """Save and log precision-recall curve plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_estimator(model, X_val, y_val, ax=ax, name="GBM")
    ax.set_title("Precision-Recall Curve (Validation Set)")
    path = artifact_dir / "precision_recall_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(path), artifact_path="plots")


def _log_feature_importance(model, feature_names, artifact_dir: Path):
    """Save and log feature importances (native GBM + permutation)."""
    # Native GBM feature importance (from the pipeline's classifier step)
    clf = model.named_steps["clf"]
    importances = clf.feature_importances_

    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_idx = np.argsort(importances)
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        importances[sorted_idx],
        color="steelblue",
    )
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Feature Importances — GradientBoosting")
    fig.tight_layout()
    path = artifact_dir / "feature_importances.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(path), artifact_path="plots")

    # Log feature importances as table
    fi_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    fi_path = artifact_dir / "feature_importances.csv"
    fi_df.to_csv(fi_path, index=False)
    mlflow.log_artifact(str(fi_path), artifact_path="tables")

    # Log each importance as individual metric
    for feat, imp in zip(feature_names, importances):
        mlflow.log_metric(f"importance_{feat}", round(float(imp), 6))


def _log_learning_curve(model, X_train, y_train, artifact_dir: Path):
    """Save and log learning curve plot."""
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(
        train_sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.1,
        color="blue",
    )
    ax.fill_between(
        train_sizes,
        val_scores.mean(axis=1) - val_scores.std(axis=1),
        val_scores.mean(axis=1) + val_scores.std(axis=1),
        alpha=0.1,
        color="orange",
    )
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", color="blue", label="Train")
    ax.plot(train_sizes, val_scores.mean(axis=1), "o-", color="orange", label="CV Val")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("F1 Score")
    ax.set_title("Learning Curve")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    path = artifact_dir / "learning_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(path), artifact_path="plots")


def _log_cv_results(grid: GridSearchCV, artifact_dir: Path):
    """Log full GridSearchCV results as CSV artifact + best/worst metrics."""
    cv_df = pd.DataFrame(grid.cv_results_)
    cv_path = artifact_dir / "cv_results.csv"
    cv_df.to_csv(cv_path, index=False)
    mlflow.log_artifact(str(cv_path), artifact_path="tables")

    # Log summary stats
    mlflow.log_metric("cv_f1_std", float(cv_df.loc[grid.best_index_, "std_test_score"]))
    mlflow.log_metric("cv_f1_worst", float(cv_df["mean_test_score"].min()))
    mlflow.log_metric("cv_f1_best", float(cv_df["mean_test_score"].max()))
    mlflow.log_metric("cv_n_candidates", len(cv_df))
    mlflow.log_metric("cv_mean_fit_time", float(cv_df["mean_fit_time"].mean()))


def _log_shap_summary(model, X_val, feature_names, artifact_dir: Path):
    """Log SHAP summary plot (best-effort — requires shap package)."""
    try:
        import shap

        # Get transformed features for the classifier
        preprocessor = model.named_steps["prep"]
        X_transformed = preprocessor.transform(X_val)

        clf = model.named_steps["clf"]
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_transformed)

        # Summary bar plot
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            X_transformed,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
        )
        path = artifact_dir / "shap_summary_bar.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close("all")
        mlflow.log_artifact(str(path), artifact_path="plots")

        # Beeswarm plot
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            X_transformed,
            feature_names=feature_names,
            show=False,
        )
        path = artifact_dir / "shap_summary_beeswarm.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close("all")
        mlflow.log_artifact(str(path), artifact_path="plots")

        # Log mean absolute SHAP values as metrics
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        for feat, val in zip(feature_names, mean_abs_shap):
            mlflow.log_metric(f"shap_mean_abs_{feat}", round(float(val), 6))

        print("✅ SHAP values logged")
    except Exception as exc:
        print(f"⚠️  SHAP logging skipped: {exc}")


def _log_prediction_distribution(y_proba, artifact_dir: Path):
    """Log probability distribution histogram."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_proba, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(x=0.5, color="red", linestyle="--", label="Decision threshold")
    ax.set_xlabel("Predicted Churn Probability")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Predicted Probabilities (Validation Set)")
    ax.legend()
    path = artifact_dir / "probability_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(path), artifact_path="plots")


def _log_data_profile(df, X_train, y_train, artifact_dir: Path):
    """Log dataset statistics and profile."""
    # Dataset overview
    stats = {
        "total_samples": len(df),
        "train_samples": len(X_train),
        "n_features_raw": len(RAW_FEATURE_COLS),
        "n_features_total": len(FEATURE_COLS),
        "churn_rate": round(float(df["churn"].mean()), 4),
        "class_0_count": int((df["churn"] == 0).sum()),
        "class_1_count": int((df["churn"] == 1).sum()),
        "class_balance_ratio": round(
            float((df["churn"] == 1).sum()) / float((df["churn"] == 0).sum()), 4
        ),
    }

    # Feature statistics
    desc = X_train.describe().T
    desc_path = artifact_dir / "feature_statistics.csv"
    desc.to_csv(desc_path)
    mlflow.log_artifact(str(desc_path), artifact_path="tables")

    # Correlation matrix plot
    fig, ax = plt.subplots(figsize=(14, 10))
    corr = X_train.corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(im)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    corr_path = artifact_dir / "correlation_matrix.png"
    fig.savefig(corr_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(corr_path), artifact_path="plots")

    # Correlation CSV
    corr_csv_path = artifact_dir / "correlation_matrix.csv"
    corr.to_csv(corr_csv_path)
    mlflow.log_artifact(str(corr_csv_path), artifact_path="tables")

    # Log data profile metrics
    for key, val in stats.items():
        mlflow.log_metric(f"data_{key}", val)

    return stats


def _log_classification_report(y_true, y_pred, artifact_dir: Path):
    """Log full classification report as artifact."""
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).T
    path = artifact_dir / "classification_report.csv"
    report_df.to_csv(path)
    mlflow.log_artifact(str(path), artifact_path="tables")

    # Also log per-class metrics
    for label in ["0", "1"]:
        if label in report:
            for metric_name in ["precision", "recall", "f1-score", "support"]:
                mlflow.log_metric(
                    f"val_class{label}_{metric_name.replace('-', '_')}",
                    float(report[label][metric_name]),
                )


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

    # Enable sklearn autolog for maximum information capture
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=True,
        log_datasets=True,
        log_post_training_metrics=True,
        silent=True,
    )

    with mlflow.start_run(run_name="churn-training") as run:
        # Create a temp directory for artifacts
        tmp_dir = Path(tempfile.mkdtemp(prefix="mlflow_artifacts_"))

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1. TAGS — Everything about the run context
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        mlflow.set_tag("mlflow.runName", "churn-training")
        mlflow.set_tag(
            "mlflow.note.content",
            (
                "Full training pipeline: GridSearchCV (72 combinations, 5-fold CV) "
                "→ GradientBoostingClassifier. 13 raw features + 3 engineered. "
                "Includes SHAP analysis, feature importance, learning curves, and "
                "full evaluation suite."
            ),
        )

        # System/environment tags
        mlflow.set_tag("system.python_version", sys.version)
        mlflow.set_tag("system.sklearn_version", sklearn.__version__)
        mlflow.set_tag("system.mlflow_version", mlflow.__version__)
        mlflow.set_tag("system.platform", platform.platform())
        mlflow.set_tag("system.processor", platform.processor())

        # Training configuration tags
        mlflow.set_tag("training.model_type", "GradientBoostingClassifier")
        mlflow.set_tag("training.pipeline", "StandardScaler + GBM")
        mlflow.set_tag("training.tuning_method", "GridSearchCV")
        mlflow.set_tag("training.cv_folds", "5")
        mlflow.set_tag("training.scoring", "f1")
        mlflow.set_tag("training.n_candidates", str(len(PARAM_GRID)))
        mlflow.set_tag(
            "training.feature_engineering", "senior_citizen, total_charges, ticket_rate"
        )

        # Dataset tags
        mlflow.set_tag("dataset.name", "telecom-churn")
        mlflow.set_tag("dataset.path", str(data_path))
        mlflow.set_tag("dataset.n_raw_features", str(len(RAW_FEATURE_COLS)))
        mlflow.set_tag("dataset.n_engineered_features", "3")
        mlflow.set_tag("dataset.n_total_features", str(len(FEATURE_COLS)))

        # GitHub CI/CD tags
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

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 2. LOG DATASET
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        dataset = mlflow.data.pandas_dataset.from_pandas(
            df, source=str(data_path), name="churn", targets="churn"
        )
        mlflow.log_input(dataset, context="training")

        # Log data profile (stats, correlation matrix, feature statistics)
        print("Logging data profile ...")
        _log_data_profile(df, X_train, y_train, tmp_dir)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 3. TRAINING — GridSearchCV
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        pipe = build_model()

        n_candidates = 1
        for v in PARAM_GRID.values():
            n_candidates *= len(v)
        print(
            f"Running hyperparameter search "
            f"(GridSearchCV 5-fold, {n_candidates} candidates) ..."
        )
        grid = GridSearchCV(
            pipe,
            PARAM_GRID,
            cv=5,
            scoring="f1",
            n_jobs=-1,
            verbose=1,
            return_train_score=True,  # Also track train scores
        )
        grid.fit(X_train, y_train)

        model = grid.best_estimator_
        best_params = grid.best_params_
        cv_f1 = float(grid.best_score_)

        print(f"Best CV F1: {cv_f1:.4f}")
        print(f"Best params: {best_params}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 4. FULL EVALUATION METRICS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        val_pred = model.predict(X_val)
        val_proba = model.predict_proba(X_val)[:, 1]
        train_pred = model.predict(X_train)
        train_proba = model.predict_proba(X_train)[:, 1]

        # Validation metrics (comprehensive)
        f1 = float(f1_score(y_val, val_pred))
        accuracy = float(accuracy_score(y_val, val_pred))
        precision = float(precision_score(y_val, val_pred))
        recall = float(recall_score(y_val, val_pred))
        roc_auc = float(roc_auc_score(y_val, val_proba))
        avg_precision = float(average_precision_score(y_val, val_proba))
        balanced_acc = float(balanced_accuracy_score(y_val, val_pred))
        mcc = float(matthews_corrcoef(y_val, val_pred))
        kappa = float(cohen_kappa_score(y_val, val_pred))
        brier = float(brier_score_loss(y_val, val_proba))
        logloss = float(log_loss(y_val, val_proba))

        # Train metrics (for overfitting detection)
        train_f1 = float(f1_score(y_train, train_pred))
        train_accuracy = float(accuracy_score(y_train, train_pred))
        train_roc_auc = float(roc_auc_score(y_train, train_proba))

        data_hash = sha256_file(data_path)
        model_fingerprint = sha256_str(f"{data_hash}:{RANDOM_STATE}:gbm:{best_params}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 5. LOG PARAMETERS (exhaustive)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        mlflow.log_param("model_type", "GradientBoostingClassifier")
        mlflow.log_param("pipeline_steps", "StandardScaler → passthrough → GBM")
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", 0.25)
        mlflow.log_param("stratified_split", True)
        mlflow.log_param("data_hash", data_hash)
        mlflow.log_param("model_fingerprint", model_fingerprint)
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_val_samples", len(X_val))
        mlflow.log_param("n_features_raw", len(RAW_FEATURE_COLS))
        mlflow.log_param("n_features_engineered", 3)
        mlflow.log_param("n_features_total", len(FEATURE_COLS))
        mlflow.log_param("feature_cols", str(FEATURE_COLS))
        mlflow.log_param("raw_feature_cols", str(RAW_FEATURE_COLS))
        mlflow.log_param(
            "engineered_features", "senior_citizen, total_charges, ticket_rate"
        )
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("cv_scoring", "f1")
        n_candidates = 1
        for v in PARAM_GRID.values():
            n_candidates *= len(v)
        mlflow.log_param("grid_n_candidates", n_candidates)
        for k, v in best_params.items():
            mlflow.log_param(k.replace("clf__", ""), v)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 6. LOG METRICS (exhaustive)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Cross-validation metrics
        mlflow.log_metric("cv_f1", cv_f1)

        # Validation metrics
        mlflow.log_metric("val_f1", f1)
        mlflow.log_metric("val_accuracy", accuracy)
        mlflow.log_metric("val_precision", precision)
        mlflow.log_metric("val_recall", recall)
        mlflow.log_metric("val_roc_auc", roc_auc)
        mlflow.log_metric("val_avg_precision", avg_precision)
        mlflow.log_metric("val_balanced_accuracy", balanced_acc)
        mlflow.log_metric("val_mcc", mcc)
        mlflow.log_metric("val_cohen_kappa", kappa)
        mlflow.log_metric("val_brier_score", brier)
        mlflow.log_metric("val_log_loss", logloss)
        mlflow.log_metric("val_samples", len(X_val))

        # Training metrics (overfitting detection)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_roc_auc", train_roc_auc)
        mlflow.log_metric("train_samples", len(X_train))

        # Overfitting gap
        mlflow.log_metric("overfit_gap_f1", round(train_f1 - f1, 4))
        mlflow.log_metric("overfit_gap_accuracy", round(train_accuracy - accuracy, 4))
        mlflow.log_metric("overfit_gap_roc_auc", round(train_roc_auc - roc_auc, 4))

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 7. LOG ARTIFACTS — Plots & Tables
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        print("Generating plots ...")

        _log_confusion_matrix(y_val, val_pred, tmp_dir)
        _log_roc_curve(model, X_val, y_val, tmp_dir)
        _log_precision_recall_curve(model, X_val, y_val, tmp_dir)
        _log_prediction_distribution(val_proba, tmp_dir)
        _log_classification_report(y_val, val_pred, tmp_dir)

        # Feature importance (GBM native)
        transformed_feature_names = FEATURE_COLS  # Same order after pipeline
        _log_feature_importance(model, transformed_feature_names, tmp_dir)

        # GridSearchCV results
        _log_cv_results(grid, tmp_dir)

        # Learning curve & SHAP — skip in CI (adds ~5-8 min on slow runners)
        _is_ci = bool(os.getenv("CI"))

        if not _is_ci:
            print("Computing learning curve ...")
            _log_learning_curve(model, X_train, y_train, tmp_dir)

            print("Computing SHAP values ...")
            _log_shap_summary(model, X_val, transformed_feature_names, tmp_dir)
        else:
            print("⏭️  Skipping learning curve & SHAP in CI (speed optimisation)")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 8. LOG MODEL with signature & input example
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        try:
            from mlflow.models import infer_signature

            signature = infer_signature(X_val, val_pred)
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(3),
                registered_model_name="churn-model",
            )
            print("✅ Model logged and registered in MLflow")
        except Exception as exc:
            print(f"⚠️  mlflow.sklearn.log_model skipped: {exc}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 9. SAVE LOCALLY (backward compat)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
            "val_avg_precision": avg_precision,
            "val_balanced_accuracy": balanced_acc,
            "val_mcc": mcc,
            "val_cohen_kappa": kappa,
            "val_brier_score": brier,
            "val_log_loss": logloss,
            "train_f1": train_f1,
            "train_accuracy": train_accuracy,
            "train_roc_auc": train_roc_auc,
            "overfit_gap_f1": round(train_f1 - f1, 4),
            "feature_cols": FEATURE_COLS,
            "raw_feature_cols": RAW_FEATURE_COLS,
            "feature_means": feature_means,
            "churn_rate": round(float(y.mean()), 4),
            "mlflow_run_id": run.info.run_id,
            "mlflow_experiment_id": run.info.experiment_id,
        }
        write_json(version_dir / "metadata.json", metadata)
        write_json(report_dir / "train_report.json", metadata)

        # Log metadata as artifact
        try:
            mlflow.log_artifact(str(version_dir / "metadata.json"))
        except Exception as exc:
            print(f"⚠️  mlflow.log_artifact skipped: {exc}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 10. SUMMARY
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE — Summary")
        print("=" * 60)
        print(f"  Model version:    {model_fingerprint}")
        print(f"  MLflow run ID:    {run.info.run_id}")
        print(f"  MLflow tracking:  {MLFLOW_TRACKING_URI}")
        print(f"  CV F1:            {cv_f1:.4f}")
        print(f"  Val F1:           {f1:.4f}  |  Train F1: {train_f1:.4f}")
        print(f"  Val ROC-AUC:      {roc_auc:.4f}")
        print(f"  Val MCC:          {mcc:.4f}")
        print(f"  Overfit gap (F1): {train_f1 - f1:.4f}")
        print(f"  Brier score:      {brier:.4f}")
        print(f"  Saved model to:   {model_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
