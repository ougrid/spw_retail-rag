"""Output guardrails for generated answers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import structlog

from app.retrieval.vector_store import SearchResult

logger = structlog.get_logger(__name__)

TIME_PATTERN = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b")
FLOOR_PATTERN = re.compile(r"\bfloor\s+([A-Za-z0-9]+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class OutputGuardResult:
    grounding_verified: bool
    confidence: str
    reason: str


class OutputGuard:
    """Validate that generated answers stay grounded in retrieved source data."""

    def evaluate(
        self, answer: str, sources: Iterable[SearchResult]
    ) -> OutputGuardResult:
        source_list = list(sources)
        if not source_list:
            return OutputGuardResult(
                False, "low", "No retrieved sources were provided."
            )

        grounded = self._is_grounded(answer, source_list)
        confidence = self._confidence_from_scores(source_list)
        reason = (
            "ok"
            if grounded
            else "Answer contains facts not found in retrieved sources."
        )
        logger.info(
            "output_guard_evaluated",
            grounded=grounded,
            confidence=confidence,
            source_count=len(source_list),
        )
        return OutputGuardResult(
            grounding_verified=grounded, confidence=confidence, reason=reason
        )

    def _is_grounded(self, answer: str, sources: list[SearchResult]) -> bool:
        answer_lower = answer.lower()
        combined_text = " ".join([source.text.lower() for source in sources])
        known_values = {
            str(value).lower()
            for source in sources
            for value in source.metadata.values()
            if str(value).strip()
        }

        time_values = TIME_PATTERN.findall(answer)
        floor_values = FLOOR_PATTERN.findall(answer)

        for time_value in time_values:
            if (
                time_value.lower() not in combined_text
                and time_value.lower() not in known_values
            ):
                return False

        for floor_value in floor_values:
            if (
                floor_value.lower() not in known_values
                and floor_value.lower() not in combined_text
            ):
                return False

        for source in sources:
            shop_name = str(source.metadata.get("shop_name", "")).lower()
            mall_name = str(source.metadata.get("mall_name", "")).lower()
            if (
                shop_name
                and shop_name in answer_lower
                and shop_name not in combined_text
            ):
                return False
            if (
                mall_name
                and mall_name in answer_lower
                and mall_name not in combined_text
            ):
                return False

        return True

    def _confidence_from_scores(self, sources: list[SearchResult]) -> str:
        best_score = max(source.score for source in sources)
        if best_score >= 0.85:
            return "high"
        if best_score >= 0.65:
            return "medium"
        return "low"
