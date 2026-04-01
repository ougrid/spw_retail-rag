from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client.http import models

from app.ingestion.chunker import ChunkDocument
from app.retrieval.embeddings import EmbeddingClient
from app.retrieval.vector_store import QdrantVectorStore, SearchResult, _build_filter


class StubOpenAIClient:
    def __init__(self):
        self.embeddings = MagicMock()


def test_embedding_client_batches_texts():
    stub_client = StubOpenAIClient()
    stub_client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(embedding=[0.1, 0.2]),
            SimpleNamespace(embedding=[0.3, 0.4]),
        ]
    )

    client = EmbeddingClient.__new__(EmbeddingClient)
    client._client = stub_client
    client._model = "text-embedding-3-small"
    client._dimensions = 2

    embeddings = client.embed_texts(["shop one", "shop two"])

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    stub_client.embeddings.create.assert_called_once()


def test_embedding_client_returns_single_query_embedding():
    client = EmbeddingClient.__new__(EmbeddingClient)
    client.embed_texts = MagicMock(return_value=[[0.5, 0.6]])

    embedding = client.embed_query("where is nike")

    assert embedding == [0.5, 0.6]


def test_build_filter_creates_qdrant_filter_from_metadata():
    query_filter = _build_filter({"mall_name": "ICONSIAM", "category": "Sports"})

    assert isinstance(query_filter, models.Filter)
    assert len(query_filter.must) == 2
    assert query_filter.must[0].key == "mall_name"
    assert query_filter.must[0].match.value == "ICONSIAM"


def test_ensure_collection_creates_missing_collection():
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False
    store = QdrantVectorStore("localhost", 6333, "mall_shops", 1536, client=mock_client)

    store.ensure_collection()

    mock_client.create_collection.assert_called_once()


def test_upsert_documents_pushes_chunk_payloads_to_qdrant():
    mock_client = MagicMock()
    store = QdrantVectorStore("localhost", 6333, "mall_shops", 2, client=mock_client)
    documents = [
        ChunkDocument(
            chunk_id="shop-1-summary",
            text="Nike is a sports shop.",
            metadata={"mall_name": "ICONSIAM", "chunk_type": "summary"},
        )
    ]

    store.upsert_documents(documents, [[0.1, 0.2]])

    mock_client.upsert.assert_called_once()
    points = mock_client.upsert.call_args.kwargs["points"]
    assert points[0].payload["chunk_id"] == "shop-1-summary"
    assert points[0].payload["mall_name"] == "ICONSIAM"


def test_search_returns_normalized_results():
    mock_client = MagicMock()
    mock_client.search.return_value = [
        SimpleNamespace(
            payload={
                "chunk_id": "shop-1-summary",
                "text": "Nike is a sports shop.",
                "mall_name": "ICONSIAM",
                "chunk_type": "summary",
            },
            score=0.91,
        )
    ]
    store = QdrantVectorStore("localhost", 6333, "mall_shops", 2, client=mock_client)

    results = store.search([0.1, 0.2], limit=3, metadata_filters={"mall_name": "ICONSIAM"})

    assert results == [
        SearchResult(
            chunk_id="shop-1-summary",
            text="Nike is a sports shop.",
            score=0.91,
            metadata={"mall_name": "ICONSIAM", "chunk_type": "summary"},
        )
    ]


def test_upsert_documents_requires_matching_document_and_embedding_counts():
    store = QdrantVectorStore("localhost", 6333, "mall_shops", 2, client=MagicMock())

    with pytest.raises(ValueError, match="counts must match"):
        store.upsert_documents([], [[0.1, 0.2]])


def test_health_check_returns_false_when_qdrant_is_unavailable():
    mock_client = MagicMock()
    mock_client.get_collections.side_effect = RuntimeError("connection failed")
    store = QdrantVectorStore("localhost", 6333, "mall_shops", 2, client=mock_client)

    assert store.health_check() is False
