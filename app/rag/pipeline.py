"""End-to-end RAG pipeline orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import structlog

from app.generation.prompts import build_follow_up_rewrite_messages, build_messages
from app.guardrails.input_guard import InputGuard, InputGuardResult
from app.guardrails.output_guard import OutputGuard, OutputGuardResult
from app.rag.query_analyzer import QueryAnalyzer
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector_store import SearchResult
from app.session_memory import ConversationTurn

logger = structlog.get_logger(__name__)

FALLBACK_ANSWER = (
    "I don't have that specific information in my records right now. "
    "I can help you find shops, check opening hours, or suggest stores by category — "
    "what would you like to know?"
)

SHORT_FOLLOW_UP_MAX_CHARS = 24
FOLLOW_UP_MARKERS = {
    "ok",
    "okay",
    "yes",
    "yes please",
    "sure",
    "go",
    "go there",
    "take me there",
    "need",
    "i want that",
    "that one",
    "โอเค",
    "ได้",
    "เอา",
    "ต้องการ",
    "ครับ",
    "ค่ะ",
}


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    sources: list[dict[str, Any]]
    guardrails: dict[str, Any]
    retrieval_debug: dict[str, Any] | None = None


class RAGPipeline:
    """Coordinate guardrails, retrieval, prompting, and generation."""

    def __init__(
        self,
        embedding_client: Any,
        vector_store: Any,
        llm_client: Any,
        input_guard: InputGuard,
        output_guard: OutputGuard,
        query_analyzer: QueryAnalyzer | None = None,
        retriever: HybridRetriever | None = None,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._llm_client = llm_client
        self._input_guard = input_guard
        self._output_guard = output_guard
        self._query_analyzer = query_analyzer or QueryAnalyzer()
        self._retriever = retriever or HybridRetriever(
            vector_store=vector_store,
            query_analyzer=self._query_analyzer,
            top_k=top_k,
        )
        self._top_k = top_k
        self._score_threshold = score_threshold

    def answer(
        self,
        query: str,
        metadata_filters: dict[str, str] | None = None,
        conversation_history: Sequence[ConversationTurn] | None = None,
    ) -> RAGResponse:
        conversation_history = list(conversation_history or [])
        resolved_query = self._resolve_query(query, conversation_history)

        input_result = self._input_guard.evaluate(resolved_query)
        if not input_result.allowed:
            return self._blocked_response(input_result)

        query_embedding = self._embedding_client.embed_query(resolved_query)
        retrieval_result = self._retriever.retrieve(
            resolved_query,
            query_embedding,
            explicit_filters=metadata_filters,
        )
        sources = retrieval_result.sources

        if not sources:
            output_result = self._output_guard.evaluate(FALLBACK_ANSWER, [])
            return self._response(
                answer=FALLBACK_ANSWER,
                sources=[],
                input_result=input_result,
                output_result=output_result,
                retrieval_debug=retrieval_result.debug,
            )

        messages = build_messages(
            query,
            sources,
            conversation_history=conversation_history,
            resolved_query=resolved_query,
        )
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
            retrieval_debug=retrieval_result.debug,
        )
        logger.info(
            "rag_pipeline_completed",
            source_count=len(sources),
            grounding_verified=output_result.grounding_verified,
            confidence=output_result.confidence,
        )
        return response

    def _resolve_query(
        self,
        query: str,
        conversation_history: Sequence[ConversationTurn],
    ) -> str:
        if not conversation_history or not self._needs_follow_up_resolution(query):
            return query

        try:
            rewritten = self._llm_client.generate(
                build_follow_up_rewrite_messages(query, conversation_history)
            ).strip()
        except Exception as error:
            logger.warning("rag_query_rewrite_failed", query=query, error=str(error))
            return query

        if not rewritten:
            return query

        logger.info(
            "rag_query_rewritten",
            original_query=query,
            resolved_query=rewritten,
        )
        return rewritten

    def _needs_follow_up_resolution(self, query: str) -> bool:
        normalized = " ".join(query.strip().lower().split())
        if not normalized:
            return False
        if normalized in FOLLOW_UP_MARKERS:
            return True
        return len(query.strip()) <= SHORT_FOLLOW_UP_MAX_CHARS

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
            retrieval_debug=None,
        )

    def _response(
        self,
        answer: str,
        sources: list[SearchResult],
        input_result: InputGuardResult,
        output_result: OutputGuardResult,
        retrieval_debug: dict[str, Any] | None = None,
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
            retrieval_debug=retrieval_debug,
        )
