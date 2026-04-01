from app.guardrails.input_guard import InputGuard
from app.guardrails.output_guard import OutputGuard
from app.rag.pipeline import FALLBACK_ANSWER, RAGPipeline
from app.retrieval.vector_store import SearchResult


class StubEmbeddingClient:
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


class StubVectorStore:
    def __init__(self, results):
        self.results = results

    def search(self, query_vector, limit, metadata_filters=None, score_threshold=None):
        return self.results


class StubLLMClient:
    def __init__(self, answer: str):
        self.answer_text = answer

    def generate(self, messages):
        return self.answer_text


def _sample_source() -> SearchResult:
    return SearchResult(
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


def test_pipeline_returns_blocked_response_for_out_of_scope_query():
    pipeline = RAGPipeline(
        embedding_client=StubEmbeddingClient(),
        vector_store=StubVectorStore([]),
        llm_client=StubLLMClient("unused"),
        input_guard=InputGuard(),
        output_guard=OutputGuard(),
    )

    response = pipeline.answer("Write a sonnet about the moon")

    assert response.sources == []
    assert response.guardrails["input_in_scope"] is False


def test_pipeline_returns_fallback_when_no_sources_are_found():
    pipeline = RAGPipeline(
        embedding_client=StubEmbeddingClient(),
        vector_store=StubVectorStore([]),
        llm_client=StubLLMClient("unused"),
        input_guard=InputGuard(),
        output_guard=OutputGuard(),
    )

    response = pipeline.answer("What sports shops are in ICONSIAM?")

    assert response.answer == FALLBACK_ANSWER
    assert response.guardrails["confidence"] == "low"


def test_pipeline_returns_grounded_answer_with_sources():
    pipeline = RAGPipeline(
        embedding_client=StubEmbeddingClient(),
        vector_store=StubVectorStore([_sample_source()]),
        llm_client=StubLLMClient("Nike is on floor 1 of ICONSIAM and opens at 10:00."),
        input_guard=InputGuard(),
        output_guard=OutputGuard(),
    )

    response = pipeline.answer("Where is Nike in ICONSIAM?")

    assert response.answer == "Nike is on floor 1 of ICONSIAM and opens at 10:00."
    assert response.sources[0]["shop_name"] == "Nike"
    assert response.guardrails["grounding_verified"] is True


def test_pipeline_replaces_ungrounded_answer_with_fallback():
    pipeline = RAGPipeline(
        embedding_client=StubEmbeddingClient(),
        vector_store=StubVectorStore([_sample_source()]),
        llm_client=StubLLMClient("Nike opens at 09:00."),
        input_guard=InputGuard(),
        output_guard=OutputGuard(),
    )

    response = pipeline.answer("What time does Nike open?")

    assert response.answer == FALLBACK_ANSWER
    assert response.guardrails["grounding_verified"] is False
