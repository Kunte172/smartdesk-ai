from __future__ import annotations

from pathlib import Path

from smartdesk_ai.cli import main


def test_cli_generate_validate_train_and_predict(tmp_path: Path, capsys: object) -> None:
    dataset = tmp_path / "tickets.csv"
    model = tmp_path / "model.joblib"
    metrics = tmp_path / "metrics.json"
    predictions = tmp_path / "predictions.csv"

    assert main(["generate-sample", "--output", str(dataset)]) == 0
    assert main(["validate", "--input", str(dataset)]) == 0
    assert (
        main(
            [
                "train",
                "--input",
                str(dataset),
                "--model-output",
                str(model),
                "--metrics-output",
                str(metrics),
                "--predictions-output",
                str(predictions),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "predict",
                "Tài khoản của tôi đang bị khóa và cần hỗ trợ sớm.",
                "--model",
                str(model),
            ]
        )
        == 0
    )
    assert dataset.exists()
    assert model.exists()
    assert metrics.exists()
    assert predictions.exists()
