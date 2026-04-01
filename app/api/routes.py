"""API routes for chat and health endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.models import ChatRequest, ChatResponse, HealthResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    pipeline = request.app.state.pipeline
    result = pipeline.answer(payload.query, metadata_filters=payload.metadata_filters)
    return ChatResponse.model_validate(result.__dict__)


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    checks = {name: check() for name, check in request.app.state.health_checks.items()}
    status = "ok" if all(checks.values()) else "degraded"
    return HealthResponse(status=status, checks=checks)
