from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from smartdesk_ai import __version__
from smartdesk_ai.schemas import (
    BatchPredictionResponse,
    BatchPredictRequest,
    HealthResponse,
    PredictionResponse,
    PredictRequest,
)
from smartdesk_ai.service import TicketTriageService


def create_app(model_path: str | Path | None = None) -> FastAPI:
    resolved_model_path = (
        Path(model_path)
        if model_path is not None
        else Path(os.getenv("SMARTDESK_MODEL_PATH", "artifacts/baseline.joblib"))
    )
    service = TicketTriageService(resolved_model_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if resolved_model_path.exists():
            service.load()
        yield

    app = FastAPI(
        title="SmartDesk AI API",
        description="Confidence-aware ticket category and priority prediction.",
        version=__version__,
        lifespan=lifespan,
    )
    origins = [
        origin.strip()
        for origin in os.getenv("SMARTDESK_CORS_ORIGINS", "http://localhost:5173").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        model_available = service.ready or resolved_model_path.exists()
        return HealthResponse(
            status="ok" if model_available else "degraded",
            model_ready=model_available,
            model_path=str(resolved_model_path),
        )

    @app.get("/v1/model", tags=["model"])
    def model_info() -> dict[str, object]:
        try:
            if not service.ready:
                service.load()
        except (FileNotFoundError, TypeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return service.metadata

    @app.post("/v1/predict", response_model=PredictionResponse, tags=["inference"])
    def predict(request: PredictRequest) -> PredictionResponse:
        try:
            result = service.predict(request.text)
        except (FileNotFoundError, TypeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return PredictionResponse.model_validate(result, from_attributes=True)

    @app.post("/v1/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
    def predict_batch(request: BatchPredictRequest) -> BatchPredictionResponse:
        try:
            predictions = [
                PredictionResponse.model_validate(service.predict(item.text), from_attributes=True)
                for item in request.items
            ]
        except (FileNotFoundError, TypeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return BatchPredictionResponse(predictions=predictions)

    return app


app = create_app()
