MLOps local pipeline (CI/CD) with data-triggered retrain and local deployment

Prerequisites
- Python 3.11+ (3.11, 3.12 ou 3.13)
- Docker + docker compose

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

GitHub CI/CD
- Push to main (ou modifier data/raw/churn.csv) → déclenche automatiquement le workflow
- Job CI : lint, test, train, evaluate, promote, upload artifacts
- Job CD : build Docker, deploy, smoke test
- Tout tourne sur les runners GitHub (ubuntu-latest), pas de self-hosted requis
- Déclenchement manuel possible via workflow_dispatch dans l'onglet Actions
