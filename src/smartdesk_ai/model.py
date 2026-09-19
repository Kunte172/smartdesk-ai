from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from smartdesk_ai.data import load_dataset


@dataclass(frozen=True)
class BaselineBundle:
    category_model: Pipeline
    priority_model: Pipeline
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TrainingResult:
    bundle: BaselineBundle
    metrics: dict[str, Any]
    predictions: pd.DataFrame


def build_text_pipeline(*, seed: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=12_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=seed,
                ),
            ),
        ]
    )


def train_baseline(
    dataset_path: str | Path,
    *,
    test_size: float = 0.25,
    seed: int = 42,
    review_threshold: float = 0.45,
) -> TrainingResult:
    data = load_dataset(dataset_path)
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4")
    if not 0.0 < review_threshold < 1.0:
        raise ValueError("review_threshold must be between 0 and 1")

    strata = data["category"].astype(str) + "|" + data["priority"].astype(str)
    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=seed,
        stratify=strata,
    )

    category_model = build_text_pipeline(seed=seed)
    priority_model = build_text_pipeline(seed=seed)
    category_model.fit(train["text"], train["category"])
    priority_model.fit(train["text"], train["priority"])

    category_prediction = category_model.predict(test["text"])
    priority_prediction = priority_model.predict(test["text"])
    category_confidence = category_model.predict_proba(test["text"]).max(axis=1)
    priority_confidence = priority_model.predict_proba(test["text"]).max(axis=1)
    combined_confidence = category_confidence.clip(max=priority_confidence)

    predictions = test.loc[:, ["ticket_id", "text", "category", "priority"]].copy()
    predictions["predicted_category"] = category_prediction
    predictions["predicted_priority"] = priority_prediction
    predictions["category_confidence"] = category_confidence.round(6)
    predictions["priority_confidence"] = priority_confidence.round(6)
    predictions["needs_review"] = combined_confidence < review_threshold

    metrics = {
        "dataset": {
            "rows": int(len(data)),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "synthetic": True,
        },
        "category": _classification_metrics(test["category"], category_prediction),
        "priority": _classification_metrics(test["priority"], priority_prediction),
        "review": {
            "threshold": review_threshold,
            "flagged_rate": float(predictions["needs_review"].mean()),
        },
    }
    metadata = {
        "model_version": "baseline-0.1.0",
        "trained_at": datetime.now(UTC).isoformat(),
        "algorithm": "tfidf-logistic-regression",
        "review_threshold": review_threshold,
        "category_labels": sorted(category_model.classes_.tolist()),
        "priority_labels": sorted(priority_model.classes_.tolist()),
        "metrics_summary": {
            "category_macro_f1": metrics["category"]["macro_f1"],
            "priority_macro_f1": metrics["priority"]["macro_f1"],
        },
    }
    bundle = BaselineBundle(
        category_model=category_model,
        priority_model=priority_model,
        metadata=metadata,
    )
    return TrainingResult(bundle=bundle, metrics=metrics, predictions=predictions)


def save_training_result(
    result: TrainingResult,
    *,
    model_path: str | Path,
    metrics_path: str | Path,
    predictions_path: str | Path,
) -> None:
    model_destination = Path(model_path)
    metrics_destination = Path(metrics_path)
    predictions_destination = Path(predictions_path)
    for destination in (model_destination, metrics_destination, predictions_destination):
        destination.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(result.bundle, model_destination)
    metrics_destination.write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result.predictions.to_csv(predictions_destination, index=False)


def load_bundle(path: str | Path) -> BaselineBundle:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Model artifact does not exist: {source}")
    bundle = joblib.load(source)
    if not isinstance(bundle, BaselineBundle):
        raise TypeError("Model artifact is not a SmartDesk BaselineBundle")
    return bundle


def _classification_metrics(y_true: pd.Series, y_prediction: Any) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_prediction)),
        "macro_f1": float(f1_score(y_true, y_prediction, average="macro")),
        "report": classification_report(y_true, y_prediction, output_dict=True, zero_division=0),
    }
