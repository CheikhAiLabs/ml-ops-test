"""
Centralised, environment-overridable settings via pydantic-settings.

Any value can be overridden by setting an environment variable with the
same name (case-insensitive):

    export MIN_F1=0.85
    export MLFLOW_TRACKING_URI=http://mlflow.internal:5000
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # ---- Paths ----
    project_root: Path = _PROJECT_ROOT
    data_raw: Path = _PROJECT_ROOT / "data" / "raw" / "churn.csv"
    artifacts_dir: Path = _PROJECT_ROOT / "reports"
    models_dir: Path = _PROJECT_ROOT / "models"
    models_latest_dir: Path = _PROJECT_ROOT / "models" / "latest"
    models_versions_dir: Path = _PROJECT_ROOT / "models" / "versions"

    # ---- Training ----
    random_state: int = 42
    min_f1: float = Field(default=0.80, ge=0.0, le=1.0)
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)

    # ---- MLflow ----
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "churn-classifier"
    mlflow_model_name: str = "churn-model"

    # ---- API ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_log_level: str = "info"


settings = Settings()
