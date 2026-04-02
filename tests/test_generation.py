from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.generation.llm import LLMClient
from app.generation.prompts import SYSTEM_PROMPT, build_context_block, build_messages
from app.retrieval.vector_store import SearchResult


def _sample_sources():
    return [
        SearchResult(
            chunk_id="shop-1-summary",
            text="Nike is a Sports shop located on floor 1 of ICONSIAM. Open from 10:00 to 22:00.",
            score=0.92,
            metadata={
                "shop_name": "Nike",
                "mall_name": "ICONSIAM",
                "floor": "1",
                "category": "Sports",
            },
        )
    ]


def test_build_context_block_includes_source_metadata_and_text():
    context_block = build_context_block(_sample_sources())

    assert "[Source 1]" in context_block
    assert "Shop: Nike" in context_block
    assert "Mall: ICONSIAM" in context_block
    assert "Content: Nike is a Sports shop" in context_block


def test_build_messages_includes_system_prompt_and_query_context():
    messages = build_messages("Where is Nike?", _sample_sources())

    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "concierge" in SYSTEM_PROMPT.lower()
    assert messages[1]["content"]
    assert "Where is Nike?" in messages[1]["content"]


def test_llm_client_returns_completion_text():
    stub_client = MagicMock()
    stub_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="Nike is on floor 1."))
        ]
    )

    client = LLMClient.__new__(LLMClient)
    client._client = stub_client
    client._model = "gpt-4o-mini"
    client._temperature = 0.1
    client._max_tokens = 128
    client._max_retries = 1
    client._retry_delay_seconds = 0

    result = client.generate(build_messages("Where is Nike?", _sample_sources()))

    assert result == "Nike is on floor 1."


def test_llm_client_raises_after_retry_exhaustion():
    stub_client = MagicMock()
    stub_client.chat.completions.create.side_effect = RuntimeError("transient failure")

    client = LLMClient.__new__(LLMClient)
    client._client = stub_client
    client._model = "gpt-4o-mini"
    client._temperature = 0.1
    client._max_tokens = 128
    client._max_retries = 1
    client._retry_delay_seconds = 0

    with pytest.raises(RuntimeError, match="failed after retries"):
        client.generate(build_messages("Where is Nike?", _sample_sources()))
