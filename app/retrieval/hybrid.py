"""Hybrid lexical-vector retrieval and reranking."""

from __future__ import annotations

from dataclasses import dataclass
import re

import structlog

from app.rag.query_analyzer import QueryAnalysis, QueryAnalyzer
from app.retrieval.vector_store import SearchResult

logger = structlog.get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "buy",
    "can",
    "does",
    "for",
    "i",
    "in",
    "is",
    "me",
    "of",
    "on",
    "the",
    "to",
    "what",
    "where",
}


def _normalize_text(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def _normalize_token(token: str) -> str:
    token = token.lower()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _tokenize(value: str) -> set[str]:
    tokens = {
        _normalize_token(token)
        for token in _TOKEN_PATTERN.findall(value.lower())
    }
    return {token for token in tokens if token and token not in _STOPWORDS}


@dataclass(frozen=True)
class RankedResult:
    result: SearchResult
    hybrid_score: float


class HybridRetriever:
    """Blend vector similarity with lexical and metadata-aware reranking."""

    def __init__(
        self,
        vector_store,
        query_analyzer: QueryAnalyzer,
        top_k: int = 5,
        candidate_multiplier: int = 4,
        minimum_hybrid_score: float = 0.2,
        vector_weight: float = 0.65,
        lexical_weight: float = 0.35,
    ) -> None:
        self._vector_store = vector_store
        self._query_analyzer = query_analyzer
        self._top_k = top_k
        self._candidate_multiplier = candidate_multiplier
        self._minimum_hybrid_score = minimum_hybrid_score
        self._vector_weight = vector_weight
        self._lexical_weight = lexical_weight

    def retrieve(
        self,
        query: str,
        query_vector: list[float],
        explicit_filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        analysis = self._query_analyzer.analyze(query, explicit_filters)
        candidates = self._collect_candidates(query_vector, analysis, explicit_filters)
        if not candidates:
            logger.info("hybrid_retrieval_no_candidates", query=query)
            return []

        ranked = sorted(
            (
                RankedResult(
                    result=result,
                    hybrid_score=self._hybrid_score(
                        query=query,
                        result=result,
                        analysis=analysis,
                        explicit_filters=explicit_filters or {},
                    ),
                )
                for result in candidates.values()
            ),
            key=lambda item: (item.hybrid_score, item.result.score),
            reverse=True,
        )

        filtered = [
            item.result
            for item in ranked
            if item.hybrid_score >= self._minimum_hybrid_score
        ][: self._top_k]

        logger.info(
            "hybrid_retrieval_completed",
            query=query,
            candidate_count=len(candidates),
            returned_count=len(filtered),
            inferred_filters=analysis.inferred_filters,
            explicit_filters=explicit_filters or {},
        )
        return filtered

    def _collect_candidates(
        self,
        query_vector: list[float],
        analysis: QueryAnalysis,
        explicit_filters: dict[str, str] | None,
    ) -> dict[str, SearchResult]:
        candidate_limit = max(self._top_k * self._candidate_multiplier, self._top_k)
        candidates: dict[str, SearchResult] = {}

        for filters in self._build_filter_plans(analysis, explicit_filters or {}):
            response = self._vector_store.search(
                query_vector,
                limit=candidate_limit,
                metadata_filters=filters or None,
                score_threshold=None,
            )
            for result in response:
                existing = candidates.get(result.chunk_id)
                if existing is None or result.score > existing.score:
                    candidates[result.chunk_id] = result
            if response and filters.get("shop_name"):
                break
            if len(candidates) >= candidate_limit:
                break
        return candidates

    def _build_filter_plans(
        self,
        analysis: QueryAnalysis,
        explicit_filters: dict[str, str],
    ) -> list[dict[str, str]]:
        merged = dict(analysis.metadata_filters)
        inferred = dict(analysis.inferred_filters)
        plans: list[dict[str, str]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()

        def add_plan(filters: dict[str, str]) -> None:
            key = tuple(sorted(filters.items()))
            if key in seen:
                return
            seen.add(key)
            plans.append(filters)

        add_plan(merged)

        if inferred:
            if {"shop_name", "mall_name"}.issubset(inferred):
                add_plan(
                    {
                        **explicit_filters,
                        "shop_name": inferred["shop_name"],
                        "mall_name": inferred["mall_name"],
                    }
                )
            if "shop_name" in inferred:
                add_plan({**explicit_filters, "shop_name": inferred["shop_name"]})
            if "mall_name" in inferred:
                add_plan({**explicit_filters, "mall_name": inferred["mall_name"]})
            if "category" in inferred:
                add_plan({**explicit_filters, "category": inferred["category"]})

        if explicit_filters:
            add_plan(dict(explicit_filters))
        else:
            add_plan({})

        return plans

    def _hybrid_score(
        self,
        query: str,
        result: SearchResult,
        analysis: QueryAnalysis,
        explicit_filters: dict[str, str],
    ) -> float:
        query_tokens = _tokenize(query)
        metadata_text = " ".join(str(value) for value in result.metadata.values())
        candidate_tokens = _tokenize(f"{result.text} {metadata_text}")
        overlap = len(query_tokens & candidate_tokens) / max(len(query_tokens), 1)

        normalized_query = _normalize_text(query)
        normalized_shop = _normalize_text(str(result.metadata.get("shop_name", "")))
        normalized_mall = _normalize_text(str(result.metadata.get("mall_name", "")))

        lexical_score = overlap
        if normalized_shop and normalized_shop in normalized_query:
            lexical_score += 0.45
        if normalized_mall and normalized_mall in normalized_query:
            lexical_score += 0.2

        metadata_boost = 0.0
        if result.metadata.get("shop_name") == analysis.inferred_filters.get("shop_name"):
            metadata_boost += 0.2
        if result.metadata.get("mall_name") == analysis.inferred_filters.get("mall_name"):
            metadata_boost += 0.12
        if result.metadata.get("category") == analysis.inferred_filters.get("category"):
            metadata_boost += 0.12

        explicit_match_count = sum(
            1
            for key, value in explicit_filters.items()
            if result.metadata.get(key) == value
        )
        metadata_boost += explicit_match_count * 0.12

        bounded_lexical = min(1.0, lexical_score)
        return (
            self._vector_weight * max(result.score, 0.0)
            + self._lexical_weight * bounded_lexical
            + metadata_boost
        )