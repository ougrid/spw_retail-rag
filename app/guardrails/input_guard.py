"""Input guardrails for user queries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

import structlog

logger = structlog.get_logger(__name__)

# ── Tier-1 keyword fast-pass (free, <1ms) ───────────────────────────
TOPIC_KEYWORDS = {
    # location / mall
    "mall",
    "shop",
    "store",
    "tenant",
    "where",
    "located",
    "location",
    "floor",
    # time
    "open",
    "opening",
    "close",
    "closing",
    "hour",
    "hours",
    "time",
    # categories
    "category",
    "fashion",
    "sports",
    "beauty",
    "book",
    "electronics",
    "cafe",
    "supermarket",
    # shopping intent
    "buy",
    "purchase",
    "want",
    "need",
    "looking for",
    "recommend",
    "find",
    "get",
    "shirt",
    "t-shirt",
    "dress",
    "pants",
    "shoes",
    "bag",
    "jacket",
    "clothes",
    "clothing",
    "outfit",
    "brand",
    "watch",
    "cosmetic",
    "makeup",
    "sneaker",
    "accessory",
    "accessories",
    "jewel",
    "jewelry",
}

THAI_SCOPE_KEYWORDS = {
    # location / mall
    "ร้าน",
    "ห้าง",
    "อยู่ที่ไหน",
    "ที่ไหน",
    "เปิด",
    "ปิด",
    "เวลา",
    "ชั้น",
    "ซื้อ",
    # shopping intent
    "อยาก",
    "อยากได้",
    "หา",
    "แนะนำ",
    "ชุด",
    "เสื้อ",
    "กางเกง",
    "กระเป๋า",
    "รองเท้า",
    "เครื่องสำอาง",
    "สินค้า",
    "ยี่ห้อ",
    "แบรนด์",
    "เสื้อผ้า",
    "นาฬิกา",
    "แฟชั่น",
    "ลำลอง",
    "เที่ยว",
    "ช้อป",
    "ช้อปปิ้ง",
    "ของ",
}

# ── Soft redirect for out-of-scope queries (concierge tone) ─────────
OUT_OF_SCOPE_MESSAGE = (
    "I'm your shopping mall concierge! I can help you find shops, check "
    "opening hours, or suggest stores by category. Could you tell me more "
    "about what you're looking for?"
)

INTENT_CLASSIFICATION_PROMPT = """\
You are a scope classifier for a shopping-mall concierge chatbot.
Decide whether the user's message is related to shopping, retail, malls, stores, \
products, brands, services you'd find in a shopping centre, or asking for product \
recommendations / where to buy something.

Reply with ONLY a JSON object — no markdown fences, no extra text:
{"in_scope": true} or {"in_scope": false}
""".strip()


@dataclass(frozen=True)
class InputGuardResult:
    allowed: bool
    flagged: bool
    in_scope: bool
    reason: str


class ModerationClient(Protocol):
    def moderate(self, text: str) -> bool:
        """Return True when the input should be blocked."""


class IntentClassifier(Protocol):
    """Lightweight LLM call that returns structured scope classification."""

    def classify_intent(self, query: str) -> bool:
        """Return True when the query is in-scope for the mall chatbot."""


class LLMIntentClassifier:
    """Use an LLM to classify whether a query is in-scope."""

    def __init__(self, llm_client: Any) -> None:
        self._llm = llm_client

    def classify_intent(self, query: str) -> bool:
        messages = [
            {"role": "system", "content": INTENT_CLASSIFICATION_PROMPT},
            {"role": "user", "content": query},
        ]
        try:
            raw = self._llm.generate(messages)
            result = json.loads(raw)
            in_scope = result.get("in_scope", False)
            logger.info("llm_intent_classified", query=query, in_scope=in_scope)
            return bool(in_scope)
        except Exception as exc:
            logger.warning("llm_intent_classification_failed", error=str(exc))
            # Fail-open: let ambiguous queries through to retrieval
            return True


class InputGuard:
    """Run moderation, keyword fast-pass, and LLM intent checks."""

    def __init__(
        self,
        moderation_client: ModerationClient | None = None,
        intent_classifier: IntentClassifier | None = None,
    ) -> None:
        self._moderation_client = moderation_client
        self._intent_classifier = intent_classifier

    def evaluate(self, query: str) -> InputGuardResult:
        normalized_query = query.strip()
        if not normalized_query:
            return InputGuardResult(False, False, False, "Query cannot be empty.")

        # Layer 1: Content moderation
        flagged = (
            self._moderation_client.moderate(normalized_query)
            if self._moderation_client
            else False
        )
        if flagged:
            logger.warning("input_guard_flagged", query=normalized_query)
            return InputGuardResult(
                False, True, False, "Query was flagged by content moderation."
            )

        # Tier 1: Keyword fast-pass (free)
        if self._keyword_match(normalized_query):
            logger.info("input_guard_keyword_pass", query=normalized_query)
            return InputGuardResult(True, False, True, "ok")

        # Tier 2: LLM intent classification (only when keywords miss)
        if self._intent_classifier is not None:
            in_scope = self._intent_classifier.classify_intent(normalized_query)
            if in_scope:
                logger.info("input_guard_llm_pass", query=normalized_query)
                return InputGuardResult(True, False, True, "ok")

        # Not in scope — return soft redirect
        logger.info("input_guard_out_of_scope", query=normalized_query)
        return InputGuardResult(False, False, False, OUT_OF_SCOPE_MESSAGE)

    def _keyword_match(self, query: str) -> bool:
        lowered = query.lower()
        return any(keyword in lowered for keyword in TOPIC_KEYWORDS) or any(
            keyword in query for keyword in THAI_SCOPE_KEYWORDS
        )
