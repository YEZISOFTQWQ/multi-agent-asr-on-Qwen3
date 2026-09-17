.PHONY: install test lint serve init-db

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check src tests

serve:
	multi-agent-asr serve

init-db:
	multi-agent-asr init-db
