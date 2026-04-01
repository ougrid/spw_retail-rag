from fastapi.testclient import TestClient

from app.main import create_app
from app.rag.pipeline import RAGResponse


class StubPipeline:
    def answer(self, query: str, metadata_filters=None) -> RAGResponse:
        return RAGResponse(
            answer="Nike is on floor 1 of ICONSIAM.",
            sources=[
                {
                    "chunk_id": "shop-1-summary",
                    "chunk_text": "Nike is a Sports shop located on floor 1 of ICONSIAM.",
                    "relevance_score": 0.92,
                    "shop_name": "Nike",
                    "mall_name": "ICONSIAM",
                    "floor": "1",
                    "category": "Sports",
                    "open_time": "10:00",
                    "close_time": "22:00",
                    "chunk_type": "summary",
                    "parent_chunk_id": "",
                }
            ],
            guardrails={
                "input_flagged": False,
                "input_in_scope": True,
                "grounding_verified": True,
                "confidence": "high",
                "reason": "ok",
            },
        )


def test_chat_endpoint_returns_answer_sources_and_guardrails():
    app = create_app(
        pipeline=StubPipeline(),
        health_checks={"qdrant": lambda: True, "openai": lambda: True},
    )
    client = TestClient(app)

    response = client.post("/chat", json={"query": "Where is Nike?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Nike is on floor 1 of ICONSIAM."
    assert body["sources"][0]["shop_name"] == "Nike"
    assert body["guardrails"]["grounding_verified"] is True


def test_health_endpoint_returns_status_and_checks():
    app = create_app(
        pipeline=StubPipeline(),
        health_checks={"qdrant": lambda: True, "openai": lambda: False},
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "checks": {"qdrant": True, "openai": False},
    }


def test_request_id_middleware_sets_response_header():
    app = create_app(
        pipeline=StubPipeline(),
        health_checks={"qdrant": lambda: True, "openai": True},
    )
    client = TestClient(app)

    response = client.post("/chat", json={"query": "Where is Nike?"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
