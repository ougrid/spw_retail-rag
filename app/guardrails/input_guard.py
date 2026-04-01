"""Input guardrails for user queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import structlog

logger = structlog.get_logger(__name__)

TOPIC_KEYWORDS = {
    "mall",
    "shop",
    "store",
    "tenant",
    "where",
    "located",
    "location",
    "category",
    "floor",
    "open",
    "opening",
    "close",
    "closing",
    "hour",
    "hours",
    "time",
    "fashion",
    "sports",
    "beauty",
    "book",
    "electronics",
    "cafe",
    "supermarket",
}


@dataclass(frozen=True)
class InputGuardResult:
    allowed: bool
    flagged: bool
    in_scope: bool
    reason: str


class ModerationClient(Protocol):
    def moderate(self, text: str) -> bool:
        """Return True when the input should be blocked."""


class InputGuard:
    """Run moderation and topical checks against a user query."""

    def __init__(self, moderation_client: ModerationClient | None = None) -> None:
        self._moderation_client = moderation_client

    def evaluate(self, query: str) -> InputGuardResult:
        normalized_query = query.strip()
        if not normalized_query:
            return InputGuardResult(False, False, False, "Query cannot be empty.")

        flagged = self._moderation_client.moderate(normalized_query) if self._moderation_client else False
        if flagged:
            logger.warning("input_guard_flagged", query=normalized_query)
            return InputGuardResult(False, True, False, "Query was flagged by content moderation.")

        in_scope = self._is_in_scope(normalized_query)
        if not in_scope:
            logger.info("input_guard_out_of_scope", query=normalized_query)
            return InputGuardResult(False, False, False, "Query is outside mall information scope.")

        logger.info("input_guard_passed", query=normalized_query)
        return InputGuardResult(True, False, True, "ok")

    def _is_in_scope(self, query: str) -> bool:
        lowered = query.lower()
        return any(keyword in lowered for keyword in TOPIC_KEYWORDS)
