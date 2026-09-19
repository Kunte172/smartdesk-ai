from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(min_length=8, max_length=5_000)


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    category: str
    priority: str
    category_confidence: float = Field(ge=0, le=1)
    priority_confidence: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    needs_review: bool
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_path: str
