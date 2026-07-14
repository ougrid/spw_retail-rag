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

# ── Tier-0 prohibited-item deny-list (checked before the generic ────
# shopping-intent keywords below, so "buy"/"want" can't wave through a
# request for a weapon or explosive)
PROHIBITED_ITEM_KEYWORDS = {
    "gun",
    "guns",
    "firearm",
    "firearms",
    "rifle",
    "pistol",
    "handgun",
    "shotgun",
    "ammo",
    "ammunition",
    "bullet",
    "bullets",
    "weapon",
    "weapons",
    "explosive",
    "explosives",
    "bomb",
    "bombs",
    "grenade",
    "ปืน",
    "อาวุธ",
    "ระเบิด",
    "กระสุน",
}

PROHIBITED_ITEM_MESSAGE = (
    "I can't help with weapons, ammunition, or explosives. I can help you find "
    "shops, check opening hours, or suggest stores by category — what would "
    "you like to know?"
)

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

# ── Tailored redirects per off-topic category, keyed to the LLM's ───
# classification below (falls back to OUT_OF_SCOPE_MESSAGE for anything
# not in this map, e.g. an "other" category or an unrecognized value)
CATEGORY_MESSAGES = {
    "small_talk": (
        "Happy to chat! I'm your shopping mall concierge, though — I can help "
        "you find shops, check opening hours, or suggest stores by category. "
        "What are you in the mood to shop for?"
    ),
    "emotional_support": (
        "I'm sorry you're feeling that way. I'm not able to offer emotional "
        "support, but I'm here if you'd like help finding a shop, checking "
        "hours, or picking out something nice — want a few suggestions?"
    ),
    "general_knowledge": (
        "That's outside what I can help with — I'm your shopping mall "
        "concierge, so I can't answer general knowledge questions. I can "
        "help you find shops, check hours, or suggest stores by category "
        "instead."
    ),
    "prohibited_item": PROHIBITED_ITEM_MESSAGE,
}

INTENT_CLASSIFICATION_PROMPT = """\
You are an intent classifier for a shopping-mall concierge chatbot.
Classify the user's message into exactly one category:
- "shopping": related to shopping, retail, malls, stores, products, brands, \
services you'd find in a shopping centre, or asking for product \
recommendations / where to buy something.
- "prohibited_item": asking to buy or find weapons, ammunition, explosives, \
drugs, or other illegal/dangerous items, even when phrased as a shopping request.
- "small_talk": greetings, chit-chat, jokes, or asking about the bot itself.
- "emotional_support": asking for comfort, venting, or expressing loneliness, \
sadness, or distress.
- "general_knowledge": factual, trivia, news, or political questions unrelated \
to shopping.
- "other": anything else not related to shopping.

Reply with ONLY a JSON object — no markdown fences, no extra text:
{"category": "shopping"}
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
    """Lightweight LLM call that returns a structured scope category."""

    def classify_intent(self, query: str) -> str:
        """Return one of: shopping, prohibited_item, small_talk,
        emotional_support, general_knowledge, other."""


class LLMIntentClassifier:
    """Use an LLM to classify the user's message into a scope category."""

    def __init__(self, llm_client: Any) -> None:
        self._llm = llm_client

    def classify_intent(self, query: str) -> str:
        messages = [
            {"role": "system", "content": INTENT_CLASSIFICATION_PROMPT},
            {"role": "user", "content": query},
        ]
        try:
            raw = self._llm.generate(messages)
            result = json.loads(raw)
            category = str(result.get("category", "other")).strip().lower()
            logger.info("llm_intent_classified", query=query, category=category)
            return category
        except Exception as exc:
            logger.warning("llm_intent_classification_failed", error=str(exc))
            # Fail-open: let ambiguous queries through to retrieval
            return "shopping"


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

        # Tier 0: Prohibited-item deny-list (checked before shopping-intent
        # keywords so "buy"/"want" can't wave through a weapons request)
        if self._prohibited_item_match(normalized_query):
            logger.warning("input_guard_prohibited_item", query=normalized_query)
            return InputGuardResult(False, True, False, PROHIBITED_ITEM_MESSAGE)

        # Tier 1: Keyword fast-pass (free)
        if self._keyword_match(normalized_query):
            logger.info("input_guard_keyword_pass", query=normalized_query)
            return InputGuardResult(True, False, True, "ok")

        # Tier 2: LLM intent classification (only when keywords miss)
        if self._intent_classifier is not None:
            category = self._intent_classifier.classify_intent(normalized_query)
            if category == "shopping":
                logger.info("input_guard_llm_pass", query=normalized_query)
                return InputGuardResult(True, False, True, "ok")

            logger.info(
                "input_guard_llm_reject", query=normalized_query, category=category
            )
            message = CATEGORY_MESSAGES.get(category, OUT_OF_SCOPE_MESSAGE)
            return InputGuardResult(
                False, category == "prohibited_item", False, message
            )

        # Not in scope — return soft redirect (no classifier configured)
        logger.info("input_guard_out_of_scope", query=normalized_query)
        return InputGuardResult(False, False, False, OUT_OF_SCOPE_MESSAGE)

    def _keyword_match(self, query: str) -> bool:
        lowered = query.lower()
        return any(keyword in lowered for keyword in TOPIC_KEYWORDS) or any(
            keyword in query for keyword in THAI_SCOPE_KEYWORDS
        )

    def _prohibited_item_match(self, query: str) -> bool:
        lowered = query.lower()
        return any(keyword in lowered for keyword in PROHIBITED_ITEM_KEYWORDS) or any(
            keyword in query for keyword in PROHIBITED_ITEM_KEYWORDS
        )
