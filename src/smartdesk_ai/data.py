from __future__ import annotations

from pathlib import Path

import pandas as pd

from smartdesk_ai.domain import TicketCategory, TicketPriority

REQUIRED_COLUMNS = ("ticket_id", "text", "category", "priority")


class DatasetValidationError(ValueError):
    """Raised when ticket data violates the training contract."""


def load_dataset(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise DatasetValidationError(f"Dataset does not exist: {source}")

    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise DatasetValidationError(f"Missing required columns: {', '.join(missing)}")

    data = frame.loc[:, REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        data[column] = data[column].str.strip()

    if data.empty:
        raise DatasetValidationError("Dataset must contain at least one ticket")
    if (data["ticket_id"] == "").any():
        raise DatasetValidationError("ticket_id must not be blank")
    if data["ticket_id"].duplicated().any():
        duplicates = data.loc[data["ticket_id"].duplicated(), "ticket_id"].tolist()
        raise DatasetValidationError(f"Duplicate ticket_id values: {duplicates[:5]}")
    if (data["text"].str.len() < 8).any():
        raise DatasetValidationError("Every ticket text must contain at least 8 characters")

    _validate_enum_values(data, "category", TicketCategory)
    _validate_enum_values(data, "priority", TicketPriority)
    return data


def _validate_enum_values(
    data: pd.DataFrame, column: str, enum_type: type[TicketCategory] | type[TicketPriority]
) -> None:
    allowed = {item.value for item in enum_type}
    invalid = sorted(set(data[column]) - allowed)
    if invalid:
        raise DatasetValidationError(f"Invalid {column} values: {', '.join(invalid)}")
