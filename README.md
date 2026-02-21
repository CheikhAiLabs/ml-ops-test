# MLOps Churn Prediction Pipeline

End-to-end MLOps project: data validation, experiment tracking, drift detection, quality-gated promotion, containerised inference, and CI/CD — all with free, open-source tools.

## Architecture

```
data/raw/churn.csv
   │  ← Pandera schema validation
   ▼
src/train.py   ← MLflow experiment tracking
   │
   ▼
src/evaluate.py → reports/eval_report.json
   │
   ▼
src/promote.py  ← Quality gate (F1 ≥ 0.80)
   │
   ▼
models/latest/   → FastAPI + Prometheus metrics
                   → Docker multi-stage build
```

## Tools & Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Data validation | **Pandera** | Schema checks on raw data |
| Experiment tracking | **MLflow** | Params, metrics, artifacts, model registry |
| Drift detection | **Evidently** | Data drift HTML/JSON reports |
| Pipeline orchestration | **DVC** | Reproducible ML pipelines |
| API serving | **FastAPI** + **Prometheus** | Inference + monitoring metrics |
| Configuration | **pydantic-settings** | Env-overridable settings |
| Code quality | **Ruff** + **pre-commit** + **mypy** | Lint, format, type-check |
| Testing | **pytest** + **deepchecks** | Unit, API, behaviour, data tests |
| CI/CD | **GitHub Actions** | Matrix testing + Trivy scan |
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

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Readiness + model version |
| POST | `/predict` | Churn prediction (JSON body) |
| GET | `/model-info` | Model metadata (version, F1, data hash) |
| GET | `/metrics` | Prometheus metrics (predict_total, latency) |

### Example prediction

```bash
curl -X POST http://localhost:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":28,"tenure_months":6,"monthly_charges":39.9,"contract_type":0,"num_tickets":3}'
```

## MLflow Experiment Tracking

After training, launch the MLflow UI:

```bash
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
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

Push to `main` or modify `data/raw/churn.csv` → triggers automatically:

- **CI**: matrix test (Python 3.11 + 3.13), lint, train, evaluate, promote, upload artifacts
- **CD**: Docker build, deploy, smoke test all 4 endpoints, **Trivy** security scan
- Manual dispatch available via `workflow_dispatch`

## Testing

```bash
pytest -q                              # All tests
pytest tests/test_data_validation.py   # Pandera schema tests
pytest tests/test_api.py               # API endpoint tests
pytest tests/test_model_behavior.py    # Model behaviour tests
```

## Project Structure

```
├── api/main.py              # FastAPI inference server
├── src/
│   ├── settings.py          # pydantic-settings (env-overridable)
│   ├── config.py            # Backward-compatible re-exports
│   ├── schemas.py           # Pandera data validation schemas
│   ├── features.py          # Feature engineering + data loading
│   ├── train.py             # Training + MLflow tracking
│   ├── evaluate.py          # Model evaluation
│   ├── promote.py           # Quality-gated promotion (F1 ≥ 0.80)
│   ├── drift_report.py      # Evidently drift detection
│   └── utils.py             # Hashing, I/O helpers
├── tests/
│   ├── test_pipeline_smoke.py
│   ├── test_data_validation.py
│   ├── test_api.py
│   └── test_model_behavior.py
├── data/raw/churn.csv       # Dataset (2000 rows)
├── dvc.yaml                 # DVC pipeline definition
├── .pre-commit-config.yaml  # Pre-commit hooks
├── .github/workflows/ci_cd.yml
├── Dockerfile               # Multi-stage, non-root
├── docker-compose.yml
├── Makefile
└── requirements.txt / requirements-dev.txt
```
