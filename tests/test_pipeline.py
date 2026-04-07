from app.guardrails.input_guard import InputGuard
from app.guardrails.output_guard import OutputGuard
from app.rag.pipeline import FALLBACK_ANSWER, RAGPipeline
from app.retrieval.vector_store import SearchResult
from app.session_memory import ConversationTurn


class StubEmbeddingClient:
    def __init__(self) -> None:
        self.queries = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1, 0.2]


class StubVectorStore:
    def __init__(self, results):
        self.results = results

    def search(self, query_vector, limit, metadata_filters=None, score_threshold=None):
        return self.results


class StubLLMClient:
    def __init__(self, answer: str, rewrites=None):
        self.answer_text = answer
        self.rewrites = list(rewrites or [])
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        if self.rewrites:
            return self.rewrites.pop(0)
        return self.answer_text


class StubRetriever:
    def __init__(self, sources, debug=None):
        self.sources = sources
        self.debug = debug or {"inferred_filters": {}, "candidates": []}
        self.queries = []

    def retrieve(self, query, query_vector, explicit_filters=None):
        self.queries.append(query)
        return type(
            "HybridResult",
            (),
            {"sources": self.sources, "debug": self.debug},
        )()


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
    assert "shopping mall concierge" in response.answer


def test_pipeline_returns_fallback_when_no_sources_are_found():
    pipeline = RAGPipeline(
        embedding_client=StubEmbeddingClient(),
        vector_store=StubVectorStore([]),
        llm_client=StubLLMClient("unused"),
        input_guard=InputGuard(),
        output_guard=OutputGuard(),
        retriever=StubRetriever(
            [], debug={"inferred_filters": {"mall_name": "ICONSIAM"}, "candidates": []}
        ),
    )

    response = pipeline.answer("What sports shops are in ICONSIAM?")

    assert response.answer == FALLBACK_ANSWER
    assert response.guardrails["confidence"] == "low"
    assert response.retrieval_debug["inferred_filters"] == {"mall_name": "ICONSIAM"}


def test_pipeline_returns_grounded_answer_with_sources():
    pipeline = RAGPipeline(
        embedding_client=StubEmbeddingClient(),
        vector_store=StubVectorStore([_sample_source()]),
        llm_client=StubLLMClient("Nike is on floor 1 of ICONSIAM and opens at 10:00."),
        input_guard=InputGuard(),
        output_guard=OutputGuard(),
        retriever=StubRetriever(
            [_sample_source()],
            debug={
                "inferred_filters": {"shop_name": "Nike"},
                "candidates": [{"selected": True}],
            },
        ),
    )

    response = pipeline.answer("Where is Nike in ICONSIAM?")

    assert response.answer == "Nike is on floor 1 of ICONSIAM and opens at 10:00."
    assert response.sources[0]["shop_name"] == "Nike"
    assert response.guardrails["grounding_verified"] is True
    assert response.retrieval_debug["inferred_filters"] == {"shop_name": "Nike"}


def test_pipeline_replaces_ungrounded_answer_with_fallback():
    pipeline = RAGPipeline(
        embedding_client=StubEmbeddingClient(),
        vector_store=StubVectorStore([_sample_source()]),
        llm_client=StubLLMClient("Nike opens at 09:00."),
        input_guard=InputGuard(),
        output_guard=OutputGuard(),
        retriever=StubRetriever(
            [_sample_source()],
            debug={
                "inferred_filters": {"shop_name": "Nike"},
                "candidates": [{"selected": True}],
            },
        ),
    )

    response = pipeline.answer("What time does Nike open?")

    assert response.answer == FALLBACK_ANSWER
    assert response.guardrails["grounding_verified"] is False


def test_pipeline_rewrites_short_follow_up_using_conversation_history():
    embedding_client = StubEmbeddingClient()
    llm_client = StubLLMClient(
        answer="Zara is on floor 1 of Siam Center and is open from 10:00 to 22:00.",
        rewrites=["How do I get to Zara in Siam Center?"],
    )
    retriever = StubRetriever(
        [_sample_source()],
        debug={"inferred_filters": {"shop_name": "Zara"}, "candidates": []},
    )
    pipeline = RAGPipeline(
        embedding_client=embedding_client,
        vector_store=StubVectorStore([_sample_source()]),
        llm_client=llm_client,
        input_guard=InputGuard(),
        output_guard=OutputGuard(),
        retriever=retriever,
    )

    response = pipeline.answer(
        "ต้องการ",
        conversation_history=[
            ConversationTurn(role="user", content="โอเค ไป Zara"),
            ConversationTurn(
                role="assistant",
                content="Zara ตั้งอยู่ที่ชั้น 1 ของ Siam Center ค่ะ คุณต้องการให้ฉันช่วยแนะนำเส้นทางไปที่นั่นไหม?",
            ),
        ],
    )

    assert embedding_client.queries[0] == "How do I get to Zara in Siam Center?"
    assert retriever.queries[0] == "How do I get to Zara in Siam Center?"
    assert (
        response.answer
        == "Zara is on floor 1 of Siam Center and is open from 10:00 to 22:00."
    )
