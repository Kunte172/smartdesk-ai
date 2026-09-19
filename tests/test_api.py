from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from smartdesk_ai.api import create_app


def test_health_reports_ready_model(trained_assets: dict[str, Path]) -> None:
    with TestClient(create_app(trained_assets["model"])) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_ready"] is True


def test_health_reports_degraded_without_model(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path / "missing.joblib")) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_predict_endpoint(trained_assets: dict[str, Path]) -> None:
    with TestClient(create_app(trained_assets["model"])) as client:
        response = client.post(
            "/v1/predict",
            json={"text": "Tôi bị tính phí hai lần. Mong nhận được phản hồi trong hôm nay."},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "BILLING"
    assert body["priority"] == "MEDIUM"
    assert 0 <= body["confidence"] <= 1


def test_batch_predict_endpoint(trained_assets: dict[str, Path]) -> None:
    with TestClient(create_app(trained_assets["model"])) as client:
        response = client.post(
            "/v1/predict/batch",
            json={
                "items": [
                    {"text": "Tôi không thể đăng nhập. Vấn đề chưa ảnh hưởng công việc."},
                    {"text": "Đơn hàng chưa được giao. Mong được phản hồi trong hôm nay."},
                ]
            },
        )

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_predict_validates_short_text(trained_assets: dict[str, Path]) -> None:
    with TestClient(create_app(trained_assets["model"])) as client:
        response = client.post("/v1/predict", json={"text": "short"})

    assert response.status_code == 422


def test_predict_returns_503_without_model(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path / "missing.joblib")) as client:
        response = client.post("/v1/predict", json={"text": "A valid ticket text"})

    assert response.status_code == 503


def test_model_info(trained_assets: dict[str, Path]) -> None:
    with TestClient(create_app(trained_assets["model"])) as client:
        response = client.get("/v1/model")

    assert response.status_code == 200
    assert response.json()["model_version"] == "baseline-0.1.0"
