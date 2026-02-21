from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "churn.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_LATEST_DIR = MODELS_DIR / "latest"
MODELS_VERSIONS_DIR = MODELS_DIR / "versions"

RANDOM_STATE = 42

# Seuil minimal de qualité pour promotion en "latest"
MIN_F1 = 0.80
