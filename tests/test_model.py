from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from smartdesk_ai.model import BaselineBundle, load_bundle, train_baseline
from smartdesk_ai.service import TicketTriageService


def test_training_produces_metrics_and_predictions(trained_assets: dict[str, Path]) -> None:
    result = train_baseline(trained_assets["dataset"])

    assert result.metrics["dataset"]["rows"] == 192
    assert 0 <= result.metrics["category"]["macro_f1"] <= 1
    assert 0 <= result.metrics["priority"]["macro_f1"] <= 1
    assert len(result.predictions) == 48
    assert isinstance(result.bundle, BaselineBundle)


@pytest.mark.parametrize(
    ("test_size", "threshold", "message"),
    [
        (0.05, 0.45, "test_size"),
        (0.25, 1.0, "review_threshold"),
    ],
)
def test_training_rejects_invalid_configuration(
    trained_assets: dict[str, Path], test_size: float, threshold: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        train_baseline(trained_assets["dataset"], test_size=test_size, review_threshold=threshold)


def test_load_bundle_validates_artifact(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.joblib"
    joblib.dump({"not": "a model"}, invalid)

    with pytest.raises(TypeError, match="BaselineBundle"):
        load_bundle(invalid)


def test_load_bundle_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_bundle(tmp_path / "missing.joblib")


def test_service_predicts_with_confidence(trained_assets: dict[str, Path]) -> None:
    service = TicketTriageService(trained_assets["model"])

    prediction = service.predict(
        "Ứng dụng bị treo khi mở trang chính. KHẨN CẤP: toàn bộ dịch vụ ngừng hoạt động."
    )

    assert prediction.category == "TECHNICAL"
    assert prediction.priority == "URGENT"
    assert 0 <= prediction.confidence <= 1
    assert prediction.model_version == "baseline-0.1.0"
    assert service.ready is True
    assert service.metadata["algorithm"] == "tfidf-logistic-regression"
