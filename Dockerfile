FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SMARTDESK_MODEL_PATH=/app/artifacts/baseline.joblib

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir . && \
    smartdesk generate-sample --output data/sample/tickets.csv && \
    smartdesk train --input data/sample/tickets.csv \
      --model-output artifacts/baseline.joblib \
      --metrics-output artifacts/metrics.json \
      --predictions-output artifacts/test_predictions.csv

EXPOSE 8000
CMD ["uvicorn", "smartdesk_ai.api:app", "--host", "0.0.0.0", "--port", "8000"]

