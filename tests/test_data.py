from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from smartdesk_ai.data import DatasetValidationError, load_dataset
from smartdesk_ai.sample_data import generate_sample_dataset


def test_generate_and_load_sample_dataset(tmp_path: Path) -> None:
    path = tmp_path / "tickets.csv"
    generated = generate_sample_dataset(path, samples_per_combination=4)

    loaded = load_dataset(path)

    assert len(generated) == 96
    assert len(loaded) == 96
    assert loaded["ticket_id"].is_unique
    assert loaded["category"].nunique() == 6
    assert loaded["priority"].nunique() == 4


def test_generate_rejects_too_few_samples(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 4"):
        generate_sample_dataset(tmp_path / "tickets.csv", samples_per_combination=3)


def test_load_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="does not exist"):
        load_dataset(tmp_path / "missing.csv")


def test_load_rejects_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "tickets.csv"
    pd.DataFrame([{"ticket_id": "1", "text": "A valid ticket text"}]).to_csv(path, index=False)

    with pytest.raises(DatasetValidationError, match="Missing required columns"):
        load_dataset(path)


def test_load_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "tickets.csv"
    pd.DataFrame(
        [
            {
                "ticket_id": "DUP",
                "text": "A valid ticket text number one",
                "category": "GENERAL",
                "priority": "LOW",
            },
            {
                "ticket_id": "DUP",
                "text": "A valid ticket text number two",
                "category": "GENERAL",
                "priority": "LOW",
            },
        ]
    ).to_csv(path, index=False)

    with pytest.raises(DatasetValidationError, match="Duplicate ticket_id"):
        load_dataset(path)


@pytest.mark.parametrize(
    ("column", "value"),
    [("category", "UNKNOWN"), ("priority", "CRITICAL")],
)
def test_load_rejects_invalid_labels(tmp_path: Path, column: str, value: str) -> None:
    path = tmp_path / "tickets.csv"
    row = {
        "ticket_id": "T-1",
        "text": "A sufficiently long ticket text",
        "category": "GENERAL",
        "priority": "LOW",
    }
    row[column] = value
    pd.DataFrame([row]).to_csv(path, index=False)

    with pytest.raises(DatasetValidationError, match=f"Invalid {column}"):
        load_dataset(path)


def test_load_rejects_short_text(tmp_path: Path) -> None:
    path = tmp_path / "tickets.csv"
    pd.DataFrame(
        [{"ticket_id": "T-1", "text": "short", "category": "GENERAL", "priority": "LOW"}]
    ).to_csv(path, index=False)

    with pytest.raises(DatasetValidationError, match="at least 8"):
        load_dataset(path)
