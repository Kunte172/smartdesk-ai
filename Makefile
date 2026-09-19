.PHONY: install data train test lint api web

install:
	uv sync --extra dev
	cd apps/web && npm install

data:
	uv run smartdesk generate-sample

train: data
	uv run smartdesk train

test:
	uv run pytest

lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src

api:
	uv run uvicorn smartdesk_ai.api:app --reload

web:
	cd apps/web && npm run dev

