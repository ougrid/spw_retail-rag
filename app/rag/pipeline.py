"""End-to-end RAG pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from app.generation.prompts import build_messages
from app.guardrails.input_guard import InputGuard, InputGuardResult
from app.guardrails.output_guard import OutputGuard, OutputGuardResult
from app.retrieval.vector_store import SearchResult

logger = structlog.get_logger(__name__)

FALLBACK_ANSWER = "I do not have that information based on the retrieved context."


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    sources: list[dict[str, Any]]
    guardrails: dict[str, Any]


class RAGPipeline:
    """Coordinate guardrails, retrieval, prompting, and generation."""

    def __init__(
        self,
        embedding_client: Any,
        vector_store: Any,
        llm_client: Any,
        input_guard: InputGuard,
        output_guard: OutputGuard,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._llm_client = llm_client
        self._input_guard = input_guard
        self._output_guard = output_guard
        self._top_k = top_k
        self._score_threshold = score_threshold

    def answer(
        self, query: str, metadata_filters: dict[str, str] | None = None
    ) -> RAGResponse:
        input_result = self._input_guard.evaluate(query)
        if not input_result.allowed:
            return self._blocked_response(input_result)

        query_embedding = self._embedding_client.embed_query(query)
        sources = self._vector_store.search(
            query_embedding,
            limit=self._top_k,
            metadata_filters=metadata_filters,
            score_threshold=self._score_threshold,
        )

        if not sources:
            output_result = self._output_guard.evaluate(FALLBACK_ANSWER, [])
            return self._response(
                answer=FALLBACK_ANSWER,
                sources=[],
                input_result=input_result,
                output_result=output_result,
            )

        messages = build_messages(query, sources)
        raw_answer = self._llm_client.generate(messages)
        output_result = self._output_guard.evaluate(raw_answer, sources)
        final_answer = (
            raw_answer if output_result.grounding_verified else FALLBACK_ANSWER
        )

        response = self._response(
            answer=final_answer,
            sources=sources,
            input_result=input_result,
            output_result=output_result,
        )
        logger.info(
            "rag_pipeline_completed",
            source_count=len(sources),
            grounding_verified=output_result.grounding_verified,
            confidence=output_result.confidence,
        )
        return response

    def _blocked_response(self, input_result: InputGuardResult) -> RAGResponse:
        return RAGResponse(
            answer=input_result.reason,
            sources=[],
            guardrails={
                "input_flagged": input_result.flagged,
                "input_in_scope": input_result.in_scope,
                "grounding_verified": False,
                "confidence": "low",
                "reason": input_result.reason,
            },
        )

    def _response(
        self,
        answer: str,
        sources: list[SearchResult],
        input_result: InputGuardResult,
        output_result: OutputGuardResult,
    ) -> RAGResponse:
        return RAGResponse(
            answer=answer,
            sources=[
                {
                    "chunk_id": source.chunk_id,
                    "chunk_text": source.text,
                    "relevance_score": source.score,
                    **source.metadata,
                }
                for source in sources
            ],
            guardrails={
                "input_flagged": input_result.flagged,
                "input_in_scope": input_result.in_scope,
                "grounding_verified": output_result.grounding_verified,
                "confidence": output_result.confidence,
                "reason": output_result.reason,
            },
        )
