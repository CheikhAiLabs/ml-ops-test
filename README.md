MLOps local pipeline (CI/CD) with data-triggered retrain and local deployment

Prerequisites
- Python 3.11+
- Docker + docker compose
- Optional: GitHub Actions self-hosted runner for local CD

Local run (no CI)
1) python3.13 -m venv .venv
2) source .venv/bin/activate
3) pip install --upgrade pip && pip install -r requirements.txt -r requirements-dev.txt
4) make pipeline

> **Note :** Python 3.14 n'est pas supporté (pydantic-core ne compile pas). Utilisez Python 3.11, 3.12 ou 3.13.

API
- GET  http://localhost:8001/health
- POST http://localhost:8001/predict
  payload:
  {"age":28,"tenure_months":6,"monthly_charges":39.9,"contract_type":0,"num_tickets":3}

Data change simulation
- Edit or append lines in data/raw/churn.csv
- Run: make pipeline
- In CI: pushing the change triggers retrain automatically

GitHub local CD
- Configure a self-hosted runner on your machine
- Push to main
- The workflow will train, evaluate, promote, then deploy via docker compose locally
