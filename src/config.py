"""
Backward-compatible re-exports from pydantic-settings.
All values can be overridden via environment variables or .env file.
"""

from src.settings import settings

PROJECT_ROOT = settings.project_root
DATA_RAW = settings.data_raw
ARTIFACTS_DIR = settings.artifacts_dir
MODELS_DIR = settings.models_dir
MODELS_LATEST_DIR = settings.models_latest_dir
MODELS_VERSIONS_DIR = settings.models_versions_dir

RANDOM_STATE = settings.random_state
MIN_F1 = settings.min_f1
TEST_SIZE = settings.test_size

MLFLOW_TRACKING_URI = settings.mlflow_tracking_uri
MLFLOW_EXPERIMENT_NAME = settings.mlflow_experiment_name
MLFLOW_MODEL_NAME = settings.mlflow_model_name
