<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20|%203.12%20|%203.13-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.112-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/MLflow-3.10-0194E2?logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.5+-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Multi--stage-2496ED?logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/Grafana-22%20panels-F46800?logo=grafana&logoColor=white" />
  <img src="https://img.shields.io/badge/Prometheus-15%2B%20metrics-E6522C?logo=prometheus&logoColor=white" />
  <img src="https://img.shields.io/badge/DVC-Pipeline-13ADC7?logo=dvc&logoColor=white" />
  <img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white" />
</p>

# 🛡️ ChurnGuard — MLOps Churn Prediction Pipeline

> End-to-end MLOps project for telecom customer churn prediction. From raw data validation to production monitoring, every step — training, evaluation, promotion, inference, explainability, drift detection — is automated, tracked and reproducible using free, open-source tools.

---

## Table of Contents

- [Architecture Overview](#-architecture-overview)
- [Tools & Technologies](#-tools--technologies)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Workflow Step by Step](#-workflow-step-by-step)
  - [Step 1 — Data Validation](#step-1--data-validation-pandera)
  - [Step 2 — Feature Engineering](#step-2--feature-engineering)
  - [Step 3 — Model Training](#step-3--model-training-gradientboosting--gridsearchcv)
  - [Step 4 — Evaluation](#step-4--model-evaluation)
  - [Step 5 — Promotion](#step-5--quality-gate--promotion)
  - [Step 6 — Inference API](#step-6--inference-api-fastapi)
  - [Step 7 — Monitoring](#step-7--production-monitoring-prometheus--grafana)
  - [Step 8 — Drift Detection](#step-8--drift-detection-evidently)
  - [Step 9 — Testing](#step-9--testing-pytest)
  - [Step 10 — CI/CD](#step-10--cicd-github-actions)
- [DVC — Data & Pipeline Versioning](#-dvc--data--pipeline-versioning)
- [MLflow — Experiment Tracking & Model Registry](#-mlflow--experiment-tracking--model-registry)
- [MLflow Tracing — Inference Observability](#-mlflow-tracing--inference-observability)
- [Docker Monitoring Stack](#-docker-monitoring-stack)
- [ChurnGuard Frontend](#-churnguard-frontend)
- [API Endpoints](#-api-endpoints)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)

---

## 🏛️ Architecture Overview

```
📄 data/raw/churn.csv                       ← 2 000 telecom customers (14 columns)
   │
   │  🔍 Pandera schema validation          ← type, range, domain checks per column
   ▼
🔧 src/features.py                          ← 13 raw features + 3 engineered = 16 total
   │
   │  🧠 GradientBoostingClassifier
   │  🔄 GridSearchCV (72 candidates × 5-fold CV)
   │  📊 MLflow full logging (30+ metrics, 10+ plots, SHAP, model registry)
   ▼
📊 src/evaluate.py                          ← F1, accuracy, precision, recall, ROC-AUC
   │
   │  🚦 Quality gate (F1 ≥ 0.72)
   ▼
📦 models/latest/                           ← Promoted model
   │
   ▼
⚡ FastAPI inference API                    ← /predict with explainability
   │  🔭 MLflow Tracing (per-request spans)
   │  📈 Prometheus (15+ metrics)
   │  🎨 ChurnGuard UI (glassmorphic SPA)
   ▼
📈 Grafana (22-panel dashboard)             ← Real-time production monitoring
```

---

## 🧰 Tools & Technologies

Each tool in this project serves a specific role in the MLOps lifecycle. Here they are with their purpose and links to the official documentation.

| Tool | Role in this project | Documentation |
|:-----|:---------------------|:--------------|
| **[scikit-learn](https://scikit-learn.org/stable/)** | Model training (GradientBoostingClassifier), hyperparameter tuning (GridSearchCV), preprocessing (StandardScaler, ColumnTransformer), metrics computation | [scikit-learn docs](https://scikit-learn.org/stable/user_guide.html) |
| **[MLflow](https://mlflow.org/)** | Experiment tracking (params, metrics, artifacts), model registry, dataset logging, inference tracing, autologging | [MLflow docs](https://mlflow.org/docs/latest/index.html) |
| **[SHAP](https://shap.readthedocs.io/)** | Model explainability — computes Shapley values to explain each feature's contribution to predictions | [SHAP docs](https://shap.readthedocs.io/en/latest/) |
| **[FastAPI](https://fastapi.tiangolo.com/)** | REST API for inference, health checks, model metadata, Prometheus metrics endpoint, static frontend serving | [FastAPI docs](https://fastapi.tiangolo.com/) |
| **[Pandera](https://pandera.readthedocs.io/)** | Data validation — enforces types, ranges, allowed values on every column of the dataset before training | [Pandera docs](https://pandera.readthedocs.io/en/stable/) |
| **[Evidently](https://www.evidentlyai.com/)** | Data drift detection — applies statistical tests to compare distributions between reference and current data | [Evidently docs](https://docs.evidentlyai.com/) |
| **[DVC](https://dvc.org/)** | Pipeline orchestration and data versioning — defines reproducible ML stages (validate → train → evaluate → promote → drift) | [DVC docs](https://dvc.org/doc) |
| **[Prometheus](https://prometheus.io/)** | Time-series metrics collection — scrapes the API's `/metrics` endpoint to store request counts, latencies, feature distributions | [Prometheus docs](https://prometheus.io/docs/introduction/overview/) |
| **[Grafana](https://grafana.com/)** | Dashboard visualization — 22-panel auto-provisioned dashboard querying Prometheus data in real time | [Grafana docs](https://grafana.com/docs/grafana/latest/) |
| **[Docker](https://www.docker.com/)** | Containerisation — multi-stage build, non-root user, health check, 4-service compose stack | [Docker docs](https://docs.docker.com/) |
| **[GitHub Actions](https://github.com/features/actions)** | CI/CD — automated lint, test, train, evaluate, promote, build, deploy, security scan on every push | [Actions docs](https://docs.github.com/en/actions) |
| **[pytest](https://docs.pytest.org/)** | Testing framework — 20 tests covering data validation, API endpoints, model behaviour, pipeline smoke | [pytest docs](https://docs.pytest.org/en/stable/) |
| **[Ruff](https://docs.astral.sh/ruff/)** | Linting and formatting — fast Python linter/formatter replacing flake8, isort, black | [Ruff docs](https://docs.astral.sh/ruff/) |
| **[pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)** | Centralized configuration — type-safe, environment-overridable settings with `.env` file support | [pydantic-settings docs](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) |
| **[Trivy](https://trivy.dev/)** | Container security scanner — scans Docker images for known vulnerabilities (CRITICAL/HIGH) in CI/CD | [Trivy docs](https://aquasecurity.github.io/trivy/) |
| **[pandas](https://pandas.pydata.org/)** | Data manipulation — CSV loading, DataFrame operations, feature engineering | [pandas docs](https://pandas.pydata.org/docs/) |
| **[NumPy](https://numpy.org/)** | Numerical computing — array operations underlying sklearn and SHAP computations | [NumPy docs](https://numpy.org/doc/stable/) |
| **[Matplotlib](https://matplotlib.org/)** | Plotting — generates all MLflow artifact plots (confusion matrix, ROC, PR curve, learning curve, SHAP, etc.) | [Matplotlib docs](https://matplotlib.org/stable/contents.html) |

---

## 📋 Prerequisites

- 🐍 **Python 3.11, 3.12 or 3.13** (not 3.14 — pydantic-core fails to compile)
- 🐳 **Docker** + **docker compose** (for the monitoring stack)

---

## 🚀 Quick Start

```bash
# 1️⃣  Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2️⃣  Install dependencies
pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt

# 3️⃣  Run the full pipeline
make pipeline
# → lint → test → train → eval → promote → docker build → smoke test
```

---

## 📖 Workflow Step by Step

### Step 1 — Data Validation (Pandera)

```bash
python -m src.features
```

**What happens:** Before any training, the raw dataset (`data/raw/churn.csv`, 2 000 rows) is validated against a strict [Pandera](https://pandera.readthedocs.io/) schema defined in `src/schemas.py`. This schema enforces:

- **Column types** — each column must be the expected type (`int`, `float`)
- **Value ranges** — e.g. `age` must be between 18 and 100, `monthly_charges` > 0 and < 500
- **Allowed values** — e.g. `gender` ∈ {0, 1}, `contract_type` ∈ {0, 1, 2}, `payment_method` ∈ {0, 1, 2, 3}
- **Strict mode** — no extra columns allowed, catches schema drift early

If any check fails, Pandera raises a `SchemaError` with the exact rows and columns that violated the schema. This prevents corrupted or malformed data from entering the pipeline.

### Step 2 — Feature Engineering

**What happens:** After validation, `src/features.py` enriches the 13 raw columns with 3 derived features, producing a 16-feature input matrix:

| # | Raw Feature | Description |
|:--|:------------|:------------|
| 1 | `gender` | 0 = Female, 1 = Male |
| 2 | `age` | Customer age (18–100) |
| 3 | `partner` | Has partner (0/1) |
| 4 | `dependents` | Has dependents (0/1) |
| 5 | `tenure_months` | Months as customer (0–120) |
| 6 | `monthly_charges` | Monthly bill amount |
| 7 | `contract_type` | 0 = Month-to-month, 1 = One year, 2 = Two years |
| 8 | `payment_method` | 0–3 (electronic check, mailed check, bank transfer, credit card) |
| 9 | `paperless_billing` | Paperless billing enabled (0/1) |
| 10 | `internet_service` | 0 = None, 1 = DSL, 2 = Fiber optic |
| 11 | `online_security` | Online security add-on (0/1) |
| 12 | `tech_support` | Tech support add-on (0/1) |
| 13 | `num_tickets` | Number of support tickets |

| # | Engineered Feature | Formula | Why |
|:--|:-------------------|:--------|:----|
| 14 | `senior_citizen` | `1 if age ≥ 65 else 0` | Captures age-based churn behaviour |
| 15 | `total_charges` | `tenure_months × monthly_charges` | Customer lifetime value proxy |
| 16 | `ticket_rate` | `(num_tickets / max(tenure_months, 1)) × 12` | Annualized support intensity |

### Step 3 — Model Training (GradientBoosting + GridSearchCV)

```bash
python -m src.train
```

**What happens:** This is the core ML step. Here is what occurs under the hood:

1. **Data loading** — loads the validated CSV, engineers features, splits into **80% train / 20% test** (stratified by `churn`)
2. **Preprocessing pipeline** — `ColumnTransformer` applies `StandardScaler` to the 5 numeric features (age, tenure, charges, total_charges, ticket_rate) and passes the 11 binary/categorical features through unchanged
3. **Hyperparameter search** — `GridSearchCV` tries **72 parameter combinations** with **5-fold cross-validation** (360 model fits total):

   | Parameter | Values searched |
   |:----------|:----------------|
   | `n_estimators` | 200, 300, 500 |
   | `max_depth` | 3, 5, 7 |
   | `learning_rate` | 0.05, 0.1 |
   | `subsample` | 0.8, 1.0 |
   | `min_samples_leaf` | 5, 10 |

4. **Best model selection** — the combination with the highest mean cross-validation F1 score wins
5. **Evaluation on holdout** — the best model is evaluated on the 20% test set (never seen during training)

**MLflow logging (comprehensive):** Every training run is automatically tracked in MLflow with:

| Category | What is logged |
|:---------|:---------------|
| **Parameters** | All hyperparameters, data hash, random state, feature count, dataset size |
| **Metrics (30+)** | Validation (F1, accuracy, precision, recall, ROC-AUC, MCC, Cohen's Kappa, Brier score, log loss, balanced accuracy, average precision), Training (F1, accuracy, ROC-AUC), Overfitting gap (train − val), CV stats (mean, std, best, worst F1) |
| **Artifacts (10+ plots)** | Confusion matrix, ROC curve, Precision-Recall curve, feature importance (bar chart), learning curve, SHAP summary (bar + beeswarm), prediction distribution histogram, correlation matrix, data profile CSV |
| **Model registry** | Model registered as `churn-model` with input signature, ready for deployment |
| **Tags** | Python/sklearn/MLflow versions, OS, training config, GitHub CI info |
| **Autolog** | scikit-learn autolog captures additional params and metrics automatically |

6. **Local save** — model saved as `model.joblib` and `metadata.json` in `models/versions/<fingerprint>/` (fingerprint = SHA-256 of data + params)

### Step 4 — Model Evaluation

```bash
MODEL_DIR=$(ls -dt models/versions/* | head -n 1)
python -m src.evaluate --model-dir "$MODEL_DIR"
```

**What happens:**

1. Loads the trained model and metadata from the versioned directory
2. Recreates the same 80/20 stratified split (using the same random state) and evaluates the model on the **test set**
3. Computes **5 metrics**: F1, accuracy, precision, recall, ROC-AUC, plus a full classification report
4. Writes the results to `reports/eval_report.json`
5. Logs all eval metrics to MLflow — either resumes the training run (via `mlflow_run_id` saved in metadata) or creates a new evaluation run

This separate evaluation step serves as an independent check — it ensures the model's reported performance is reproducible and not just an artifact of the training script.

### Step 5 — Quality Gate & Promotion

```bash
python -m src.promote --model-dir "$MODEL_DIR" --eval-report reports/eval_report.json
```

**What happens:**

1. Reads `eval_report.json` and extracts the `test_f1` score
2. Compares it against the **minimum F1 threshold** (`MIN_F1`, default = 0.72)
3. **If F1 < threshold** → the script raises a `RuntimeError` and the pipeline **fails**. No model is deployed. This prevents a regression from reaching production.
4. **If F1 ≥ threshold** → copies `model.joblib` and `metadata.json` to `models/latest/`, making the model available for the inference API

This is a **hard quality gate** — it's the safety net ensuring only models that meet the performance bar get promoted.

### Step 6 — Inference API (FastAPI)

```bash
docker compose up -d inference-api
# or locally: uvicorn api.main:app --host 0.0.0.0 --port 8001
```

**What happens:**

1. **Startup** — the API loads `model.joblib` and `metadata.json` from `models/latest/`. It also computes feature means from the training data (used for explainability).
2. **MLflow tracing** — if `ENABLE_MLFLOW_TRACING=true` (default), the API checks that the MLflow server is reachable and configures per-request tracing under the `churn-inference` experiment.
3. **On each `/predict` request:**
   - Receives 13 raw features as JSON
   - Engineers the 3 derived features (senior_citizen, total_charges, ticket_rate)
   - Runs model inference → returns `churn_prediction` (0 or 1) and `churn_probability` (0.0–1.0)
   - Computes **leave-one-out feature attribution**: for each input feature, replaces it with the training mean, re-runs inference, and measures the probability shift. This tells you *which features drove this specific prediction*.
   - Records Prometheus metrics (request count, latency, feature distributions)
   - Creates an MLflow trace with 3 spans: `feature-engineering`, `model-inference`, `explainability`

### Step 7 — Production Monitoring (Prometheus + Grafana)

```bash
docker compose up -d
# Grafana: http://localhost:3001 (admin / mlops2024)
```

**What happens:**

- **Prometheus** scrapes the API's `/metrics` endpoint every 15 seconds, collecting 15+ metrics:

| Category | Metrics | Purpose |
|:---------|:--------|:--------|
| HTTP | `http_requests_total`, `http_request_duration_seconds` | Traffic volume and latency |
| Predictions | `predict_total`, `predict_churn_total`, `predict_no_churn_total`, `predict_latency_seconds`, `predict_errors_total`, `predict_probability` | Model usage and prediction distribution |
| Features | `feature_age`, `feature_tenure_months`, `feature_monthly_charges`, `feature_num_tickets`, `feature_contract_type_total` | Input data distribution monitoring |
| System | `model_loaded`, `model_info`, `app_start_time_seconds` | API health and model version |

- **Grafana** displays a **22-panel auto-provisioned dashboard** ("ChurnGuard — Production Monitoring") organized in 5 sections:

| Section | Panels | What you see |
|:--------|:-------|:-------------|
| 🟢 Status Bar | 8 | API up/down, uptime, total requests, churn/no-churn counts, error count, churn rate gauge, model version |
| 🚦 Traffic | 3 | Request rate (req/s), HTTP status codes (stacked), error rate (%) |
| ⏱️ Latency | 2 | Inference p50/p95/p99 (ms), HTTP latency by endpoint |
| 🎯 Predictions | 3 | Churn rate over time, average probability, pie chart (churn vs no churn) |
| 📊 Features | 3+ | Age/tenure/charges histograms, contract type pie chart |

### Step 8 — Drift Detection (Evidently)

```bash
python -m src.drift_report
# → reports/drift_report.html  (interactive visual report)
# → reports/drift_report.json  (machine-readable)
```

**What happens:**

1. Loads the full dataset and splits it into a **reference set** (75%) and a **current set** (25%)
2. [Evidently](https://docs.evidentlyai.com/) applies **statistical tests** to every feature (e.g. Kolmogorov-Smirnov for numeric features, chi-squared for categorical) to compare distributions between reference and current
3. Generates an HTML report you can open in a browser — it shows, per feature, whether the distribution has shifted significantly
4. Also generates a JSON report for programmatic consumption

In production, you would run this periodically (e.g. weekly) comparing fresh inference data against training data to detect when the model needs retraining.

### Step 9 — Testing (pytest)

```bash
pytest -q    # Run all 20 tests
```

**What happens:** The test suite validates the entire pipeline across 4 test modules:

| Module | Tests | What it checks |
|:-------|:------|:---------------|
| `test_data_validation.py` | 7 | **Schema enforcement** — verifies that valid data passes Pandera validation and that invalid data (wrong types, out-of-range values, extra columns) is correctly rejected. Tests each validation rule individually. |
| `test_api.py` | 8 | **API correctness** — tests `/health`, `/predict`, `/model-info`, `/metrics` endpoints. Checks response shapes, status codes, content types. Tests error handling (missing fields, invalid values). Verifies Prometheus metrics export. |
| `test_model_behavior.py` | 4 | **Model quality** — loads the promoted model, runs predictions on the test set, and checks: F1 ≥ 0.70, prediction probabilities in [0,1], deterministic outputs (same input → same output), no constant predictions (model actually learned). |
| `test_pipeline_smoke.py` | 1 | **End-to-end smoke test** — runs the *full pipeline* in a temp directory: train → evaluate → promote. Verifies that model files are created, metadata is complete, and the quality gate passes. This catches integration issues. |

### Step 10 — CI/CD (GitHub Actions)

Every push to `main` (touching `src/`, `api/`, `data/`, `tests/`, `Dockerfile`, etc.) triggers the pipeline automatically:

**CI job (Continuous Integration):**
1. Sets up Python 3.11 with pip caching
2. Installs production and dev dependencies
3. **Lint** — runs `ruff check .` to catch style/import issues
4. **Test** — runs all 20 pytest tests
5. **Train** — trains the model with full MLflow logging
6. **Evaluate** — evaluates on the holdout set
7. **Promote** — runs the quality gate and promotes if F1 passes
8. **Upload** — stores `model.joblib`, `metadata.json`, and reports as GitHub artifacts

**CD job (Continuous Deployment):**
1. Downloads the promoted model from CI artifacts
2. **Docker build** — builds the production image
3. **Trivy scan** — scans the Docker image for CRITICAL/HIGH vulnerabilities
4. **Deploy** — starts the inference API container
5. **Smoke test** — hits `/health`, `/predict`, `/model-info`, `/metrics` to verify everything works
6. **Teardown** — cleans up containers

You can also trigger the pipeline manually via `workflow_dispatch`.

---

## 🔄 DVC — Data & Pipeline Versioning

**[DVC (Data Version Control)](https://dvc.org/)** is a version control system designed for ML projects. While Git tracks code, DVC tracks **data files, models, and ML pipelines**.

### What DVC does in this project

DVC serves two roles here:

**1. Pipeline orchestration (`dvc.yaml`)** — defines the ML workflow as a **DAG** (Directed Acyclic Graph) of stages, each with explicit dependencies and outputs:

```yaml
# dvc.yaml — 5 stages
validate → train → evaluate → promote → drift_report
```

| Stage | Command | Dependencies | Outputs |
|:------|:--------|:-------------|:--------|
| `validate` | `python -m src.features` | `churn.csv`, `features.py`, `schemas.py` | — |
| `train` | `python -m src.train` | `churn.csv`, `train.py`, `features.py`, `config.py` | `models/versions/`, `train_report.json` |
| `evaluate` | `python -m src.evaluate` | `evaluate.py`, `churn.csv`, `models/versions/` | `eval_report.json` |
| `promote` | `python -m src.promote` | `promote.py`, `eval_report.json`, `models/versions/` | `models/latest/model.joblib`, `metadata.json` |
| `drift_report` | `python -m src.drift_report` | `churn.csv`, `drift_report.py` | `drift_report.html` |

When you run `dvc repro`, DVC checks which stages have changed inputs (via file hashing) and **only reruns what's necessary**. If only the evaluation code changed, DVC won't retrain — it'll skip straight to evaluation.

```bash
dvc repro          # Run/reproduce the pipeline (only changed stages)
dvc dag            # Visualize the pipeline DAG
dvc metrics show   # Show metrics from reports
dvc params show    # Show tracked parameters
```

**2. Data versioning** — DVC can track large files (datasets, models) that shouldn't live in Git. The project has a `.dvc/` directory and `.dvcignore` file set up for this purpose.

### What is `.dvc/` and how `.dvc` files work

The `.dvc/` directory is DVC's **internal configuration** directory (similar to `.git/` for Git). It contains:

- `.dvc/config` — DVC settings (remote storage, cache configuration, etc.)
- `.dvc/.gitignore` — ensures DVC cache internals don't get committed to Git
- `.dvc/tmp/` — temporary working files

When you track a large file with DVC (e.g. `dvc add data/raw/churn.csv`), here's what happens:

1. DVC **hashes** the file content (MD5) and stores the hash in a small `.dvc` file (e.g. `data/raw/churn.csv.dvc`) — this file is just a few lines of YAML
2. The actual data file is added to `.gitignore` (so Git ignores it)
3. You commit the `.dvc` file to Git (a few bytes) instead of the actual data file (which can be GBs)
4. The actual data is pushed to a **remote storage** (S3, GCS, Azure, SSH, etc.) via `dvc push`
5. Anyone can restore the exact data with `dvc pull` or `dvc checkout`

**Why this matters:** Every Git commit captures the exact dataset version used, making experiments fully reproducible. You can `git checkout` any past commit, run `dvc checkout`, and get the exact data that was used — even if the dataset has changed 100 times since then.

### `.dvcignore`

Works like `.gitignore` but for DVC — tells DVC to skip certain files/directories when scanning for changes. This can speed up `dvc status` and `dvc repro` by ignoring irrelevant files.

---

## 📊 MLflow — Experiment Tracking & Model Registry

Access the MLflow UI at **http://localhost:5000** (started via `docker compose up`).

### What MLflow tracks per training run

| Category | Details |
|:---------|:--------|
| ⚙️ **Parameters** | All GridSearchCV hyperparameters, data hash, random state, feature count, dataset size, split ratio |
| 📏 **Metrics (30+)** | Validation: F1, accuracy, precision, recall, ROC-AUC, balanced accuracy, MCC, Cohen's Kappa, Brier score, log loss, average precision. Training: F1, accuracy, ROC-AUC. Overfitting gap: train−val for F1/accuracy/ROC-AUC. CV: mean F1, std, best, worst, n_candidates, fit time |
| 📈 **Plots & artifacts** | Confusion matrix, ROC curve, Precision-Recall curve, feature importance bar chart, learning curve, SHAP bar + beeswarm, prediction distribution histogram, correlation matrix, feature statistics CSV, CV results CSV, classification report CSV |
| 🏷️ **Tags** | Python version, scikit-learn version, MLflow version, OS, training script, Git commit, GitHub Actions run info |
| 📦 **Model registry** | Model registered as `churn-model` with inferred input/output signature |
| 📊 **Dataset** | Training dataset logged via `mlflow.log_input()` for lineage tracking |

### Model registry

After training, the model is automatically registered in MLflow's **Model Registry** as `churn-model`. Each training run creates a new version. This provides:
- **Version history** — see all model versions with their metrics side by side
- **Lineage** — trace any deployed model back to its training run, dataset, and code

---

## 🔭 MLflow Tracing — Inference Observability

Every `/predict` request generates an **MLflow trace** visible in the **Traces** tab of the MLflow UI.

### Trace structure

```
🔗 churn-prediction (CHAIN)                ← root span
   ├─ inputs: {age: 45, tenure_months: 6, ...}
   ├─ outputs: {churn_prediction: 1, churn_probability: 0.96}
   │
   ├── 🔧 feature-engineering (PARSER)     ← feature computation span
   │   ├─ inputs: raw 13 features
   │   └─ outputs: engineered 16 features
   │
   ├── 🧠 model-inference (LLM)            ← model prediction span
   │   ├─ inputs: {n_features: 16}
   │   └─ outputs: {prediction: 1, probability: 0.96, latency_ms: 4.2}
   │
   └── 📊 explainability (RETRIEVER)       ← attribution span
       └─ outputs: top feature contributions
```

**Configuration:** controlled by `ENABLE_MLFLOW_TRACING` environment variable (default: `true`). At startup, the API health-checks the MLflow server — if unreachable, tracing is silently disabled without impacting predictions.

---

## 🐳 Docker Monitoring Stack

```bash
docker compose up -d --build
```

| Service | Image | Port | Purpose |
|:--------|:------|:-----|:--------|
| 🎨 **inference-api** | Custom build | `8001` | FastAPI API + ChurnGuard UI |
| 📊 **mlflow** | `ghcr.io/mlflow/mlflow:v3.10.0` | `5000` | Experiment tracking + model registry |
| 🔥 **prometheus** | `prom/prometheus:v2.51.0` | `9090` | Metrics collection (30d retention) |
| 📈 **grafana** | `grafana/grafana:10.4.0` | `3001` | Dashboards (admin / mlops2024) |

**Docker image details:** multi-stage build based on `python:3.11-slim`. Stage 1 (builder) installs dependencies; Stage 2 (runtime) copies only what's needed, creates a non-root `appuser`, and includes a `HEALTHCHECK` every 30 seconds. Named volumes persist data across restarts (`mlflow-artifacts`, `prometheus-data`, `grafana-data`).

---

## 🎨 ChurnGuard Frontend

Modern single-page application accessible at **http://localhost:8001**:

- 🎯 **Predict** — Interactive form with sliders and toggle switches, preset profiles (high risk, loyal, etc.), visual risk gauge, automatic recommendations, feature contribution chart
- 🤖 **Model** — Metric progress bars (F1, accuracy, precision, recall, ROC-AUC), model parameters, feature list
- ℹ️ **About** — Architecture overview and API endpoints

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/` | ChurnGuard frontend UI |
| `GET` | `/health` | Readiness check + model version |
| `POST` | `/predict` | Churn prediction with explainability |
| `GET` | `/model-info` | Model metadata (version, metrics, params, data hash) |
| `GET` | `/metrics` | Prometheus metrics (15+ metrics) |

### Example prediction

```bash
curl -X POST http://localhost:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "gender": 0, "age": 28, "partner": 0, "dependents": 0,
    "tenure_months": 6, "monthly_charges": 39.9, "contract_type": 0,
    "payment_method": 2, "paperless_billing": 1, "internet_service": 2,
    "online_security": 0, "tech_support": 0, "num_tickets": 3
  }'
```

---

## ⚙️ Configuration

All settings are centralized in `src/settings.py` using [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) and overridable via environment variables or a `.env` file:

| Setting | Default | Description |
|:--------|:--------|:------------|
| `MIN_F1` | `0.72` | Quality gate threshold |
| `TEST_SIZE` | `0.2` | Train/test split ratio |
| `RANDOM_STATE` | `42` | Reproducibility seed |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `MLFLOW_EXPERIMENT_NAME` | `churn-classifier` | MLflow experiment name |
| `MLFLOW_MODEL_NAME` | `churn-model` | MLflow model registry name |
| `ENABLE_MLFLOW_TRACING` | `true` | Enable/disable inference tracing |
| `API_HOST` | `0.0.0.0` | API listen address |
| `API_PORT` | `8000` | API listen port |

---

## 📁 Project Structure

```
📦 ml-ops-test/
├── ⚡ api/main.py                     # FastAPI inference server + MLflow tracing
├── 🎨 static/
│   ├── index.html                     # ChurnGuard SPA (glassmorphic UI)
│   ├── style.css                      # Modern dark theme CSS
│   └── app.js                         # Frontend logic (predictions, gauges, presets)
├── 🧠 src/
│   ├── settings.py                    # pydantic-settings (env-overridable config)
│   ├── config.py                      # Backward-compatible re-exports
│   ├── schemas.py                     # Pandera data validation schemas
│   ├── features.py                    # Feature engineering + data loading
│   ├── train.py                       # GradientBoosting + GridSearchCV + MLflow (30+ metrics)
│   ├── evaluate.py                    # Model evaluation + MLflow logging
│   ├── promote.py                     # Quality-gated promotion (F1 ≥ 0.72)
│   ├── drift_report.py               # Evidently drift detection
│   └── utils.py                       # Hashing, I/O helpers
├── 🧪 tests/
│   ├── test_data_validation.py        # Pandera schema tests (7)
│   ├── test_api.py                    # API endpoint tests (8)
│   ├── test_model_behavior.py         # Model behaviour tests (4)
│   └── test_pipeline_smoke.py         # Full pipeline smoke test (1)
├── 📈 monitoring/
│   ├── prometheus/prometheus.yml      # Scrape config
│   └── grafana/
│       ├── provisioning/              # Auto-provisioned datasources + dashboards
│       └── dashboards/mlops-churn.json # 22-panel dashboard
├── 📄 data/raw/churn.csv              # Dataset (2 000 rows, 14 columns)
├── 📦 models/
│   ├── latest/                        # Promoted model (model.joblib + metadata.json)
│   └── versions/                      # All trained model versions
├── 📊 reports/                        # Generated reports (train, eval, drift)
├── 🔄 dvc.yaml                        # DVC pipeline definition (5 stages)
├── 🔄 .dvc/                           # DVC internal config (cache, remotes)
├── 🔄 .dvcignore                      # Files DVC should ignore
├── ✨ .pre-commit-config.yaml         # Pre-commit hooks (ruff, mypy)
├── 🚀 .github/workflows/ci_cd.yml    # CI/CD pipeline (lint, test, train, deploy, Trivy)
├── 🐳 Dockerfile                      # Multi-stage, non-root, healthcheck
├── 🐳 docker-compose.yml             # 4 services: API, MLflow, Prometheus, Grafana
├── 🛠️ Makefile                        # Build targets (pipeline, train, test, etc.)
└── 📦 requirements.txt / requirements-dev.txt
```

---

<p align="center">
  Built with ❤️ by <strong>CheikhAiLabs</strong>
</p>
