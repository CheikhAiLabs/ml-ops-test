<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20|%203.12%20|%203.13-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.112-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/MLflow-3.10-0194E2?logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Multi--stage-2496ED?logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/Grafana-22%20panels-F46800?logo=grafana&logoColor=white" />
  <img src="https://img.shields.io/badge/Prometheus-15%2B%20metrics-E6522C?logo=prometheus&logoColor=white" />
  <img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white" />
</p>

# 🛡️ ChurnGuard — MLOps Churn Prediction Pipeline

> End-to-end MLOps project: data validation, experiment tracking, **real-time inference tracing**, drift detection, quality-gated promotion, containerised inference, **production monitoring (22-panel Grafana dashboard, 15+ Prometheus metrics)**, modern UI, and CI/CD — all with free, open-source tools.

---

## 🏛️ Architecture

```
📄 data/raw/churn.csv
   │  ← 🔍 Pandera schema validation
   ▼
🧠 src/train.py    ← GradientBoostingClassifier + GridSearchCV (5-fold)
   │                  MLflow experiment tracking (params, metrics, datasets)
   ▼
📊 src/evaluate.py  → reports/eval_report.json
   │
   ▼
🚦 src/promote.py   ← Quality gate (F1 ≥ 0.78)
   │
   ▼
📦 models/latest/    → ⚡ FastAPI + 15+ Prometheus metrics
                      → 🔭 MLflow Tracing (latency, spans, inputs/outputs)
                      → 🎨 ChurnGuard UI (glassmorphic SPA)
                      → 🐳 Docker multi-stage build
                      → 📈 Grafana 22-panel dashboard
```

---

## 🧰 Tools & Stack

| Layer | Tool | Purpose |
|:------|:-----|:--------|
| 🧠 **ML Model** | **GradientBoostingClassifier** + **GridSearchCV** | Ensemble model with hyperparameter tuning (5-fold CV) |
| 🔍 Data validation | **Pandera** | Schema checks on raw data |
| 📊 Experiment tracking | **MLflow 3.x** | Params, metrics, artifacts, datasets, model registry |
| 🔭 **Inference tracing** | **MLflow Tracing** | Per-request spans with latency, inputs/outputs, errors |
| 📉 Drift detection | **Evidently** | Data drift HTML/JSON reports |
| 🔄 Pipeline orchestration | **DVC** | Reproducible ML pipelines |
| ⚡ API serving | **FastAPI** + **Prometheus** | Inference + 15+ monitoring metrics |
| 🎨 **Frontend UI** | **ChurnGuard SPA** | Modern glassmorphic prediction interface |
| 📈 Monitoring | **Prometheus** + **Grafana** | 22-panel production dashboard |
| ⚙️ Configuration | **pydantic-settings** | Env-overridable settings |
| ✨ Code quality | **Ruff** + **pre-commit** + **mypy** | Lint, format, type-check |
| 🧪 Testing | **pytest** (20 tests) | Unit, API, behaviour, data tests |
| 🚀 CI/CD | **GitHub Actions** | Lint, test, train, deploy, Trivy scan |
| 🐳 Containerisation | **Docker** | Multi-stage, non-root, healthcheck |

---

## 📋 Prerequisites

- 🐍 Python 3.11, 3.12 or 3.13 (**not** 3.14 — pydantic-core fails to compile)
- 🐳 Docker + docker compose

---

## 🚀 Quick Start

```bash
# 1️⃣  Create venv
python3.13 -m venv .venv
source .venv/bin/activate

# 2️⃣  Install dependencies
pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt

# 3️⃣  Run the full pipeline (lint → test → train → eval → promote → deploy → smoke)
make pipeline
```

---

## 🐳 Docker Monitoring Stack

Launch all 4 services in one command:

```bash
docker compose up -d --build
```

| Service | URL | Credentials |
|:--------|:----|:------------|
| 🎨 **ChurnGuard UI + API** | http://localhost:8001 | — |
| 📊 **MLflow** | http://localhost:5000 | — |
| 🔥 **Prometheus** | http://localhost:9090 | — |
| 📈 **Grafana** | http://localhost:3001 | `admin` / `mlops2024` |

---

## 🎨 ChurnGuard Frontend

Modern single-page application accessible at http://localhost:8001 :

- 🎯 **Predict** — Interactive form with sliders, preset profiles (high risk, loyal, etc.), visual risk gauge, automatic recommendations
- 🤖 **Model** — Metric progress bars (F1, accuracy, precision, recall, ROC AUC), model parameters, feature list
- ℹ️ **About** — Architecture and API endpoints

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| 🟢 GET | `/` | ChurnGuard frontend UI |
| 💚 GET | `/health` | Readiness + model version |
| 🔵 POST | `/predict` | Churn prediction (JSON body) |
| 🟡 GET | `/model-info` | Model metadata (version, F1, params, data hash) |
| 🟠 GET | `/metrics` | Prometheus metrics (15+ metrics) |

### 💡 Example prediction

```bash
curl -X POST http://localhost:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":28,"tenure_months":6,"monthly_charges":39.9,"contract_type":0,"num_tickets":3}'
```

---

## 📊 MLflow Experiment Tracking

MLflow runs automatically with `docker compose up`. Accessible at http://localhost:5000.

Each run logs:
- ⚙️ **Params** — `model_type`, `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `data_hash`, `random_state`
- 📏 **Metrics** — `cv_f1`, `val_f1`, `val_accuracy`, `val_precision`, `val_recall`, `val_roc_auc`
- 💾 **Datasets** — Automatic training dataset registration (`mlflow.log_input`)
- 🏷️ **Tags** — user, source script, git commit, CI/CD run info

```bash
python -m src.train
# → Runs appear on http://localhost:5000
```

---

## 🔭 MLflow Tracing (Real-time Observability)

Every `/predict` request automatically generates an **MLflow trace** visible in the **Traces** tab of the MLflow UI (http://localhost:5000).

### 📋 What each trace captures

| Data | Detail |
|:-----|:-------|
| ⏱️ **Real latency** | Actual model inference time (ms) — measured inside the span |
| 📥 **Inputs** | All 5 customer features (age, tenure, charges, contract, tickets) |
| 📤 **Outputs** | `churn_prediction` (0/1) + `churn_probability` (float) |
| 🏷️ **Attributes** | `model_version`, `model_type`, `latency_ms` |
| ✅ **Status** | OK or ERROR |

### 🏗️ Span structure

```
🔗 churn-prediction (CHAIN)         ← root span
   ├─ inputs: {age: 45, tenure_months: 6, ...}
   ├─ outputs: {churn_prediction: 1, churn_probability: 0.96}
   ├─ attributes: model_version, model_type
   │
   └── 🧠 model-inference (LLM)     ← child span
       ├─ inputs: {n_features: 5, n_samples: 1}
       ├─ outputs: {churn_prediction: 1, churn_probability: 0.96}
       └─ attributes: latency_ms: 4.2
```

### ⚙️ Tracing configuration

Controlled by the `ENABLE_MLFLOW_TRACING` environment variable:

```bash
# ✅ Enabled by default in docker-compose
ENABLE_MLFLOW_TRACING=true   # (default)

# ❌ Disabled in CI/CD to avoid network dependencies
ENABLE_MLFLOW_TRACING=false
```

> 💡 At startup, the API checks that the MLflow server is reachable (health check with 2s timeout). If unreachable, tracing is silently disabled without impacting predictions.

---

## 📉 Drift Detection

Generate an Evidently drift report:

```bash
python -m src.drift_report
# → reports/drift_report.html  (visual)
# → reports/drift_report.json  (machine-readable)
```

---

## 🔄 DVC Pipeline

```bash
dvc repro        # Run/reproduce the ML pipeline
dvc dag          # Visualise DAG
```

---

## 🔥 Prometheus Metrics (15+)

The API exposes a `/metrics` endpoint with rich instrumentation:

| Category | Metric | Type | Description |
|:---------|:-------|:-----|:------------|
| 🌐 **HTTP** | `http_requests_total` | Counter | Total requests (method, endpoint, status) |
| | `http_request_duration_seconds` | Histogram | HTTP latency per endpoint |
| 🎯 **Predictions** | `predict_total` | Counter | Total prediction count |
| | `predict_churn_total` | Counter | Predictions where churn=1 |
| | `predict_no_churn_total` | Counter | Predictions where churn=0 |
| | `predict_latency_seconds` | Histogram | Model-only inference latency |
| | `predict_errors_total` | Counter | Failed prediction requests |
| | `predict_probability` | Summary | Probability distribution |
| 📊 **Features** | `feature_age` | Histogram | Age distribution |
| | `feature_tenure_months` | Histogram | Tenure distribution |
| | `feature_monthly_charges` | Histogram | Monthly charges distribution |
| | `feature_num_tickets` | Histogram | Support tickets distribution |
| | `feature_contract_type_total` | Counter | Contract types seen |
| 🖥️ **System** | `model_loaded` | Gauge | Model loaded (0/1) |
| | `model_info` | Info | Version, type, F1 of loaded model |
| | `app_start_time_seconds` | Gauge | App startup timestamp |

---

## 📈 Grafana Dashboard (22 panels)

Dashboard **"ChurnGuard — Production Monitoring"** auto-provisioned, organized in 5 rows:

| Section | Panels | Content |
|:--------|:-------|:--------|
| 🟢 **Status Bar** | 8 | API Status, Uptime, Total Requests, Churn/No Churn counts, Errors, Churn Rate gauge, Model Version |
| 🚦 **Traffic** | 3 | Request Rate (req/s), HTTP Status Codes (stacked), Error Rate (%) |
| ⏱️ **Latency** | 2 | Inference p50/p95/p99 (ms), HTTP Latency by endpoint |
| 🎯 **Predictions** | 3 | Churn Rate over time, Avg Probability, Pie chart Churn vs No Churn |
| 📊 **Feature Distribution** | 3+ | Histograms: Age, Tenure, Monthly Charges — Pie: Contract Type |

> 🔗 Accessible at http://localhost:3001 (`admin` / `mlops2024`). Auto-refresh every 10 seconds.

---

## ⚙️ Configuration

All settings are environment-overridable (via `.env` file or exported vars):

```bash
# 🚦 Override quality gate
export MIN_F1=0.85

# 📊 Point MLflow at a remote server
export MLFLOW_TRACKING_URI=http://mlflow.internal:5000

# 🔭 Enable/disable MLflow tracing
export ENABLE_MLFLOW_TRACING=true

# 🌐 Change API port
export API_PORT=9000
```

See [src/settings.py](src/settings.py) for the full list.

---

## ✨ Pre-commit Hooks

```bash
pre-commit install          # Set up hooks
pre-commit run --all-files  # Run manually
```

Hooks: `ruff check` · `ruff format` · `mypy` · trailing-whitespace · YAML/JSON validation · large-file guard.

---

## 🚀 CI/CD (GitHub Actions)

Push to `main` touching `src/`, `api/`, `data/raw/`, `tests/`, `Dockerfile`, etc. → triggers automatically:

- ✅ **CI** — Python 3.11, lint (ruff), 20 pytest tests, train (GBM + GridSearchCV), evaluate, promote, upload artifacts
- 🐳 **CD** — Docker build, deploy, smoke test all endpoints, **Trivy** security scan
- 🔧 Manual dispatch available via `workflow_dispatch`

### 💡 Trigger the pipeline with a data change

```bash
# Add new rows to the dataset
echo "55,36,65.0,1,2,0" >> data/raw/churn.csv
git add data/raw/churn.csv
git commit -m "data: add new customer record"
git push origin main
# → CI/CD pipeline runs automatically 🚀
```

---

## 🧪 Testing

```bash
pytest -q                              # All 20 tests
pytest tests/test_data_validation.py   # 🔍 Pandera schema tests (7)
pytest tests/test_api.py               # ⚡ API endpoint tests (8)
pytest tests/test_model_behavior.py    # 🧠 Model behaviour tests (4)
pytest tests/test_pipeline_smoke.py    # 💨 Pipeline smoke test (1)
```

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
│   ├── settings.py                    # pydantic-settings (env-overridable)
│   ├── config.py                      # Backward-compatible re-exports
│   ├── schemas.py                     # Pandera data validation schemas
│   ├── features.py                    # Feature engineering + data loading
│   ├── train.py                       # GradientBoosting + GridSearchCV + MLflow
│   ├── evaluate.py                    # Model evaluation
│   ├── promote.py                     # Quality-gated promotion (F1 ≥ 0.78)
│   ├── drift_report.py               # Evidently drift detection
│   └── utils.py                       # Hashing, I/O helpers
├── 🧪 tests/
│   ├── test_pipeline_smoke.py         # Pipeline smoke test
│   ├── test_data_validation.py        # Pandera schema tests
│   ├── test_api.py                    # API endpoint tests
│   └── test_model_behavior.py         # Model behaviour tests
├── 📈 monitoring/
│   ├── prometheus/prometheus.yml      # Scrape config
│   └── grafana/
│       ├── provisioning/              # Auto-provisioned datasources + dashboards
│       └── dashboards/mlops-churn.json # 22-panel Grafana dashboard
├── 📄 data/raw/churn.csv              # Dataset (2000 rows)
├── 🔄 dvc.yaml                        # DVC pipeline definition
├── ✨ .pre-commit-config.yaml         # Pre-commit hooks
├── 🚀 .github/workflows/ci_cd.yml    # CI/CD pipeline
├── 🐳 Dockerfile                      # Multi-stage, non-root, healthcheck
├── 🐳 docker-compose.yml             # 4 services: API, MLflow, Prometheus, Grafana
├── 🛠️ Makefile
└── 📦 requirements.txt / requirements-dev.txt
```

---

<p align="center">
  Built with ❤️ by <strong>CheikhAiLabs</strong>
</p>
