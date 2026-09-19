# SmartDesk AI

Confidence-aware support-ticket triage built as an end-to-end AI engineering project.

SmartDesk predicts a ticket's category and priority, returns separate confidence scores, and
flags uncertain cases for human review. The first release uses an intentionally transparent
TF-IDF + Logistic Regression baseline so future models must demonstrate measurable value.

## What this repository proves

- A validated and reproducible ML data pipeline.
- Honest baseline evaluation before adding a Transformer.
- Versioned FastAPI inference contracts.
- A usable React interface rather than a notebook-only demo.
- Tests, type checking, linting, Docker, and CI.
- Explicit model limitations and human-in-the-loop routing.

## Current status

`v0.1` is a software-complete baseline running on deterministic synthetic Vietnamese tickets.
The synthetic metrics validate the implementation only and must not be presented as real-world
model quality. See [the backlog](docs/BACKLOG.md) for the public-dataset milestone.

## Quick start

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Node.js 22.

```bash
uv sync --extra dev
uv run smartdesk generate-sample
uv run smartdesk train
uv run uvicorn smartdesk_ai.api:app --reload
```

In another terminal:

```bash
cd apps/web
npm install
npm run dev
```

Open the web app at `http://localhost:5173` and API documentation at
`http://localhost:8000/docs`.

## Docker

```bash
docker compose up --build
```

The API is available at `http://localhost:8000`; the web app is available at
`http://localhost:5173`.

## Commands

```bash
uv run smartdesk generate-sample
uv run smartdesk validate --input data/sample/tickets.csv
uv run smartdesk train
uv run smartdesk predict "Tôi không thể đăng nhập và cần hỗ trợ ngay."
uv run pytest
uv run ruff check .
uv run mypy src
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Process and model readiness |
| `GET` | `/v1/model` | Current model metadata |
| `POST` | `/v1/predict` | Predict one ticket |
| `POST` | `/v1/predict/batch` | Predict up to 100 tickets |

Example:

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Ứng dụng bị treo và toàn bộ nhóm không thể làm việc."}'
```

## Project structure

```text
apps/web/              React + TypeScript client
src/smartdesk_ai/      data, training, inference, API, CLI
tests/                 unit and API integration tests
data/                  data contract and generated development sample
artifacts/             generated model and evaluation outputs
docs/                  architecture, model card, backlog
.github/workflows/     continuous integration
```

## Evaluation policy

The project reports macro-F1, accuracy, per-class precision/recall/F1, confidence-based review
rate, and later calibration and latency. Model selection is based on a frozen evaluation set,
not a few hand-picked UI examples.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Baseline evaluation](docs/EVALUATION.md)
- [Model card](docs/MODEL_CARD.md)
- [Backlog](docs/BACKLOG.md)

## License

Source code is released under the MIT License. Third-party datasets retain their own licenses.
