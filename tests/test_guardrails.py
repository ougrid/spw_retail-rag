from types import SimpleNamespace
from unittest.mock import MagicMock

from app.guardrails.input_guard import InputGuard, OUT_OF_SCOPE_MESSAGE
from app.guardrails.openai_moderation import OpenAIModerationClient
from app.guardrails.output_guard import OutputGuard
from app.retrieval.vector_store import SearchResult


class StubModerationClient:
    def __init__(self, flagged: bool):
        self.flagged = flagged

    def moderate(self, text: str) -> bool:
        return self.flagged


def test_input_guard_blocks_flagged_queries():
    guard = InputGuard(moderation_client=StubModerationClient(flagged=True))

    result = guard.evaluate("Where is Nike?")

    assert result.allowed is False
    assert result.flagged is True
    assert result.reason == "Query was flagged by content moderation."


def test_input_guard_blocks_out_of_scope_queries():
    guard = InputGuard()

    result = guard.evaluate("Write me a poem about the ocean")

    assert result.allowed is False
    assert result.in_scope is False
    assert result.reason == OUT_OF_SCOPE_MESSAGE


def test_input_guard_allows_shopping_intent_queries():
    guard = InputGuard()

    result = guard.evaluate("I want a T-shirt")

    assert result.allowed is True
    assert result.in_scope is True


def test_input_guard_allows_thai_shopping_intent():
    guard = InputGuard()

    result = guard.evaluate("อยากได้ชุดลำลองใส่ไปเที่ยว")

    assert result.allowed is True
    assert result.in_scope is True


def test_input_guard_llm_fallback_allows_ambiguous_in_scope_query():
    class StubIntentClassifier:
        def classify_intent(self, query: str) -> bool:
            return True

    guard = InputGuard(intent_classifier=StubIntentClassifier())

    # No keyword match, but LLM says in-scope
    result = guard.evaluate("มีอะไรน่าสนใจบ้าง")

    assert result.allowed is True
    assert result.in_scope is True


def test_input_guard_llm_fallback_rejects_off_topic_query():
    class StubIntentClassifier:
        def classify_intent(self, query: str) -> bool:
            return False

    guard = InputGuard(intent_classifier=StubIntentClassifier())

    result = guard.evaluate("What is the meaning of life?")

    assert result.allowed is False
    assert result.in_scope is False


def test_input_guard_allows_shop_queries():
    guard = InputGuard()

    result = guard.evaluate("What sports shops are on floor 1?")

    assert result.allowed is True
    assert result.in_scope is True


def test_input_guard_allows_thai_shop_queries():
    guard = InputGuard()

    result = guard.evaluate("ร้าน Nike อยู่ที่ไหน")

    assert result.allowed is True
    assert result.in_scope is True


def test_output_guard_marks_grounded_answer_with_high_confidence():
    guard = OutputGuard()
    sources = [
        SearchResult(
            chunk_id="shop-1-summary",
            text="Nike is a Sports shop located on floor 1 of ICONSIAM. Open from 10:00 to 22:00.",
            score=0.92,
            metadata={
                "shop_name": "Nike",
                "mall_name": "ICONSIAM",
                "floor": "1",
                "category": "Sports",
                "open_time": "10:00",
                "close_time": "22:00",
            },
        )
    ]

    result = guard.evaluate(
        "Nike is on floor 1 of ICONSIAM and opens at 10:00.", sources
    )

    assert result.grounding_verified is True
    assert result.confidence == "high"


def test_output_guard_flags_answer_with_unknown_time():
    guard = OutputGuard()
    sources = [
        SearchResult(
            chunk_id="shop-1-summary",
            text="Nike is a Sports shop located on floor 1 of ICONSIAM. Open from 10:00 to 22:00.",
            score=0.92,
            metadata={
                "shop_name": "Nike",
                "mall_name": "ICONSIAM",
                "floor": "1",
                "open_time": "10:00",
                "close_time": "22:00",
            },
        )
    ]

    result = guard.evaluate("Nike opens at 09:00.", sources)

    assert result.grounding_verified is False
    assert result.reason == "Answer contains facts not found in retrieved sources."


def test_output_guard_returns_low_confidence_without_sources():
    guard = OutputGuard()

    result = guard.evaluate("I do not know.", [])

    assert result.grounding_verified is False
    assert result.confidence == "low"


def test_openai_moderation_client_flags_blocked_input():
    stub_client = MagicMock()
    stub_client.moderations.create.return_value = SimpleNamespace(
        results=[SimpleNamespace(flagged=True)]
    )

    client = OpenAIModerationClient.__new__(OpenAIModerationClient)
    client._client = stub_client
    client._model = "omni-moderation-latest"

    assert client.moderate("harmful input") is True
