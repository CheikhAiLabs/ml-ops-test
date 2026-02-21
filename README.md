# ChurnGuard — MLOps Churn Prediction Pipeline

End-to-end MLOps project: data validation, experiment tracking, drift detection, quality-gated promotion, containerised inference, monitoring, modern UI, and CI/CD — all with free, open-source tools.

## Architecture

```
data/raw/churn.csv
   │  ← Pandera schema validation
   ▼
src/train.py   ← GradientBoostingClassifier + GridSearchCV (5-fold)
   │              MLflow experiment tracking (params, metrics)
   ▼
src/evaluate.py → reports/eval_report.json
   │
   ▼
src/promote.py  ← Quality gate (F1 ≥ 0.80)
   │
   ▼
models/latest/   → FastAPI + Prometheus metrics
                   → ChurnGuard UI (glassmorphic SPA)
                   → Docker multi-stage build
                   → Grafana dashboards
```

## Tools & Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| **ML Model** | **GradientBoostingClassifier** + **GridSearchCV** | Ensemble model with hyperparameter tuning (5-fold CV) |
| Data validation | **Pandera** | Schema checks on raw data |
| Experiment tracking | **MLflow 3.x** | Params, metrics, artifacts, model registry |
| Drift detection | **Evidently** | Data drift HTML/JSON reports |
| Pipeline orchestration | **DVC** | Reproducible ML pipelines |
| API serving | **FastAPI** + **Prometheus** | Inference + monitoring metrics |
| **Frontend UI** | **ChurnGuard SPA** | Modern glassmorphic prediction interface |
| Monitoring | **Prometheus** + **Grafana** | Real-time metrics dashboards |
| Configuration | **pydantic-settings** | Env-overridable settings |
| Code quality | **Ruff** + **pre-commit** + **mypy** | Lint, format, type-check |
| Testing | **pytest** (20 tests) | Unit, API, behaviour, data tests |
| CI/CD | **GitHub Actions** | Lint, test, train, deploy, Trivy scan |
| Containerisation | **Docker** | Multi-stage, non-root, healthcheck |

## Prerequisites

- Python 3.11, 3.12 or 3.13 (**not** 3.14 — pydantic-core fails to compile)
- Docker + docker compose

## Quick Start

```bash
# 1. Create venv
python3.13 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt

# 3. Run the full pipeline (lint → test → train → eval → promote → deploy → smoke)
make pipeline
```

## Docker Monitoring Stack

Launch all 4 services in one command:

```bash
docker compose up -d --build
```

| Service | URL | Credentials |
|---------|-----|-------------|
| **ChurnGuard UI + API** | http://localhost:8001 | — |
| **MLflow** | http://localhost:5000 | — |
| **Prometheus** | http://localhost:9090 | — |
| **Grafana** | http://localhost:3001 | `admin` / `mlops2024` |

## ChurnGuard Frontend

Modern single-page application accessible at http://localhost:8001 :

- **Prediction** — Formulaire interactif avec sliders, profils types (haut risque, fidèle, etc.), jauge visuelle du risque, recommandations automatiques
- **Modèle** — Métriques en barres de progression (F1, accuracy, precision, recall, ROC AUC), paramètres du modèle, features utilisées
- **À propos** — Architecture et endpoints API

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | ChurnGuard frontend UI |
| GET | `/health` | Readiness + model version |
| POST | `/predict` | Churn prediction (JSON body) |
| GET | `/model-info` | Model metadata (version, F1, params, data hash) |
| GET | `/metrics` | Prometheus metrics (predict_total, latency) |

### Example prediction

```bash
curl -X POST http://localhost:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":28,"tenure_months":6,"monthly_charges":39.9,"contract_type":0,"num_tickets":3}'
```

## MLflow Experiment Tracking

MLflow runs automatically with `docker compose up`. Accessible at http://localhost:5000.

Chaque run enregistre :
- **Params** : model_type, n_estimators, max_depth, learning_rate, subsample, data_hash, random_state
- **Metrics** : cv_f1, val_f1, val_accuracy, val_precision, val_recall, val_roc_auc
- **Tags** : user, source script, git commit

Pour un entraînement local qui log vers le serveur MLflow Docker :

```bash
python -m src.train
# Les runs apparaissent sur http://localhost:5000
```

## Drift Detection

Generate an Evidently drift report:

```bash
python -m src.drift_report
# → reports/drift_report.html  (visual)
# → reports/drift_report.json  (machine-readable)
```

## DVC Pipeline

```bash
dvc repro        # Run/reproduce the ML pipeline
dvc dag          # Visualise DAG
```

## Configuration

All settings are environment-overridable (via `.env` file or exported vars):

```bash
# Override quality gate
export MIN_F1=0.85

# Point MLflow at a remote server
export MLFLOW_TRACKING_URI=http://mlflow.internal:5000

# Change API port
export API_PORT=9000
```

See [src/settings.py](src/settings.py) for the full list.

## Pre-commit Hooks

```bash
pre-commit install          # Set up hooks
pre-commit run --all-files  # Run manually
```

Hooks: ruff check, ruff format, mypy, trailing-whitespace, YAML/JSON validation, large-file guard.

## CI/CD (GitHub Actions)

Push to `main` touching `src/`, `api/`, `data/raw/`, `tests/`, `Dockerfile`, etc. → triggers automatically:

- **CI**: Python 3.11, lint (ruff), 20 pytest tests, train (GBM + GridSearchCV), evaluate, promote, upload artifacts
- **CD**: Docker build, deploy, smoke test all endpoints, **Trivy** security scan
- Manual dispatch available via `workflow_dispatch`

### Trigger the pipeline with a data change

```bash
# Add new rows to the dataset
echo "55,36,65.0,1,2,0" >> data/raw/churn.csv
git add data/raw/churn.csv
git commit -m "data: add new customer record"
git push origin main
# → CI/CD pipeline runs automatically on GitHub Actions
```

## Testing

```bash
pytest -q                              # All 20 tests
pytest tests/test_data_validation.py   # Pandera schema tests (7)
pytest tests/test_api.py               # API endpoint tests (8)
pytest tests/test_model_behavior.py    # Model behaviour tests (4)
pytest tests/test_pipeline_smoke.py    # Pipeline smoke test (1)
```

## Project Structure

```
├── api/main.py              # FastAPI inference server + static UI serving
├── static/
│   ├── index.html           # ChurnGuard SPA (glassmorphic UI)
│   ├── style.css            # Modern dark theme CSS
│   └── app.js               # Frontend logic (predictions, gauges, presets)
├── src/
│   ├── settings.py          # pydantic-settings (env-overridable)
│   ├── config.py            # Backward-compatible re-exports
│   ├── schemas.py           # Pandera data validation schemas
│   ├── features.py          # Feature engineering + data loading
│   ├── train.py             # GradientBoosting + GridSearchCV + MLflow tracking
│   ├── evaluate.py          # Model evaluation
│   ├── promote.py           # Quality-gated promotion (F1 ≥ 0.80)
│   ├── drift_report.py      # Evidently drift detection
│   └── utils.py             # Hashing, I/O helpers
├── tests/
│   ├── test_pipeline_smoke.py
│   ├── test_data_validation.py
│   ├── test_api.py
│   └── test_model_behavior.py
├── monitoring/
│   ├── prometheus/prometheus.yml         # Scrape config
│   └── grafana/
│       ├── provisioning/                 # Auto-provisioned datasources + dashboards
│       └── dashboards/mlops-churn.json   # 7-panel Grafana dashboard
├── data/raw/churn.csv       # Dataset (2000 rows)
├── dvc.yaml                 # DVC pipeline definition
├── .pre-commit-config.yaml  # Pre-commit hooks
├── .github/workflows/ci_cd.yml
├── Dockerfile               # Multi-stage, non-root, healthcheck
├── docker-compose.yml       # 4 services: API, MLflow, Prometheus, Grafana
├── Makefile
└── requirements.txt / requirements-dev.txt
```
