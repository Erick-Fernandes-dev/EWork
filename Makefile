# Makefile
.PHONY: all setup lint format test run-etl run-rag-ingest run-rag-api clean

VENV_ACTIVATE = $(shell poetry env info --path)/bin/activate

all: setup lint test

setup:
	@echo "--> Installing dependencies with Poetry..."
	poetry install

lint:
	@echo "--> Running linter (ruff)..."
	poetry run ruff check .
	@echo "--> Checking formatting (black)..."
	poetry run black --check .
	@echo "--> Checking types (mypy)..."
	poetry run mypy .

format:
	@echo "--> Formatting with ruff and black..."
	poetry run ruff check . --fix
	poetry run black .

test:
	@echo "--> Running tests..."
	poetry run pytest -q

run-etl:
	@echo "--> Running data ETL pipeline..."
	poetry run python -m src.etl.run

run-rag-ingest:
	@echo "--> Running RAG ingestion (PDF to VectorDB)..."
	poetry run python -m src.rag.ingest

run-rag-api:
	@echo "--> Starting RAG API server at http://localhost:8003"
	poetry run uvicorn src.rag.api:app --host 0.0.0.0 --port 8003 --reload

clean:
	@echo "--> Cleaning up generated files..."
	rm -rf data/* silver/* docs/*.pdf db/*
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete