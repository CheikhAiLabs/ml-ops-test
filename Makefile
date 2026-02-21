PYTHON := python3

.PHONY: venv install install-dev lint test train eval promote api-build api-up api-down smoke pipeline

venv:
	$(PYTHON) -m venv .venv

install:
	. .venv/bin/activate && pip install -r requirements.txt

install-dev:
	. .venv/bin/activate && pip install -r requirements.txt -r requirements-dev.txt

lint:
	. .venv/bin/activate && ruff check .

test:
	. .venv/bin/activate && pytest -q

train:
	. .venv/bin/activate && $(PYTHON) -m src.train

eval:
	@MODEL_DIR=$$(ls -dt models/versions/* | head -n 1) && \
	. .venv/bin/activate && $(PYTHON) -m src.evaluate --model-dir $$MODEL_DIR

promote:
	@MODEL_DIR=$$(ls -dt models/versions/* | head -n 1) && \
	. .venv/bin/activate && $(PYTHON) -m src.evaluate --model-dir $$MODEL_DIR && \
	. .venv/bin/activate && $(PYTHON) -m src.promote --model-dir $$MODEL_DIR --eval-report reports/eval_report.json

api-build:
	docker compose build

api-up:
	docker compose up -d --build

api-down:
	docker compose down

smoke:
	@echo "Waiting for API to start..."
	@sleep 5
	curl -sS http://localhost:8001/health | cat
	curl -sS -X POST http://localhost:8001/predict \
	  -H 'Content-Type: application/json' \
	  -d '{"age":28,"tenure_months":6,"monthly_charges":39.9,"contract_type":0,"num_tickets":3}' | cat

pipeline: lint test train eval promote api-up smoke
