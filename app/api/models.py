"""Pydantic API models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    metadata_filters: dict[str, str] | None = None
    session_id: str | None = None


class SourceResponse(BaseModel):
    chunk_id: str
    chunk_text: str
    relevance_score: float
    shop_name: str | None = None
    mall_name: str | None = None
    floor: str | None = None
    category: str | None = None
    open_time: str | None = None
    close_time: str | None = None
    chunk_type: str | None = None
    parent_chunk_id: str | None = None


class GuardrailsResponse(BaseModel):
    input_flagged: bool
    input_in_scope: bool
    grounding_verified: bool
    confidence: str
    reason: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    guardrails: GuardrailsResponse
    retrieval_debug: dict[str, Any] | None = None
    session_id: str


class HealthResponse(BaseModel):
    status: str
    checks: dict[str, bool]
