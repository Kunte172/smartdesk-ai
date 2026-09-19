from __future__ import annotations

from pathlib import Path

import pytest

from smartdesk_ai.model import save_training_result, train_baseline
from smartdesk_ai.sample_data import generate_sample_dataset


@pytest.fixture(scope="session")
def trained_assets(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("trained-assets")
    dataset_path = root / "tickets.csv"
    model_path = root / "baseline.joblib"
    metrics_path = root / "metrics.json"
    predictions_path = root / "predictions.csv"
    generate_sample_dataset(dataset_path, samples_per_combination=8)
    result = train_baseline(dataset_path)
    save_training_result(
        result,
        model_path=model_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
    )
    return {
        "dataset": dataset_path,
        "model": model_path,
        "metrics": metrics_path,
        "predictions": predictions_path,
    }
