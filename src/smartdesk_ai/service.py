from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smartdesk_ai.model import BaselineBundle, load_bundle


@dataclass(frozen=True)
class TicketPrediction:
    category: str
    priority: str
    category_confidence: float
    priority_confidence: float
    confidence: float
    needs_review: bool
    model_version: str


class TicketTriageService:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._bundle: BaselineBundle | None = None

    @property
    def ready(self) -> bool:
        return self._bundle is not None

    @property
    def metadata(self) -> dict[str, Any]:
        if self._bundle is None:
            return {}
        return dict(self._bundle.metadata)

    def load(self) -> None:
        self._bundle = load_bundle(self.model_path)

    def predict(self, text: str) -> TicketPrediction:
        if self._bundle is None:
            self.load()
        assert self._bundle is not None

        category = str(self._bundle.category_model.predict([text])[0])
        priority = str(self._bundle.priority_model.predict([text])[0])
        category_confidence = float(
            self._bundle.category_model.predict_proba([text]).max(axis=1)[0]
        )
        priority_confidence = float(
            self._bundle.priority_model.predict_proba([text]).max(axis=1)[0]
        )
        confidence = min(category_confidence, priority_confidence)
        threshold = float(self._bundle.metadata["review_threshold"])
        return TicketPrediction(
            category=category,
            priority=priority,
            category_confidence=category_confidence,
            priority_confidence=priority_confidence,
            confidence=confidence,
            needs_review=confidence < threshold,
            model_version=str(self._bundle.metadata["model_version"]),
        )
