"""Qdrant vector store helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.ingestion.chunker import ChunkDocument

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """Normalized search result returned from Qdrant."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class QdrantVectorStore:
    """Wrapper around Qdrant collection management and search."""

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        dimensions: int,
        client: QdrantClient | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.dimensions = dimensions
        self.client = client or QdrantClient(host=host, port=port)

    def ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        existing = self.client.collection_exists(self.collection_name)
        if existing:
            logger.info(
                "qdrant_collection_exists", collection_name=self.collection_name
            )
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.dimensions, distance=models.Distance.COSINE
            ),
        )
        logger.info("qdrant_collection_created", collection_name=self.collection_name)

    def recreate_collection(self) -> None:
        """Drop and recreate the collection."""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.dimensions, distance=models.Distance.COSINE
            ),
        )
        logger.info("qdrant_collection_recreated", collection_name=self.collection_name)

    def upsert_documents(
        self, documents: list[ChunkDocument], embeddings: list[list[float]]
    ) -> None:
        """Upsert chunk documents and embeddings into Qdrant."""
        if len(documents) != len(embeddings):
            raise ValueError("Document and embedding counts must match")

        points = []
        for document, embedding in zip(documents, embeddings, strict=True):
            points.append(
                models.PointStruct(
                    id=str(uuid4()),
                    vector=embedding,
                    payload={
                        "chunk_id": document.chunk_id,
                        "text": document.text,
                        **document.metadata,
                    },
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(
            "qdrant_documents_upserted",
            collection_name=self.collection_name,
            count=len(points),
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        metadata_filters: dict[str, str] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Search Qdrant using a query vector and optional metadata filters."""
        query_filter = _build_filter(metadata_filters)
        response = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )

        results = [
            SearchResult(
                chunk_id=item.payload["chunk_id"],
                text=item.payload["text"],
                score=item.score,
                metadata={
                    key: value
                    for key, value in item.payload.items()
                    if key not in {"chunk_id", "text"}
                },
            )
            for item in response
        ]
        logger.info(
            "qdrant_search_completed",
            collection_name=self.collection_name,
            result_count=len(results),
        )
        return results

    def health_check(self) -> bool:
        """Return whether Qdrant is reachable."""
        try:
            self.client.get_collections()
        except Exception:
            logger.exception(
                "qdrant_health_check_failed", collection_name=self.collection_name
            )
            return False
        logger.info("qdrant_health_check_passed", collection_name=self.collection_name)
        return True


def _build_filter(metadata_filters: dict[str, str] | None) -> models.Filter | None:
    if not metadata_filters:
        return None

    conditions = [
        models.FieldCondition(key=key, match=models.MatchValue(value=value))
        for key, value in metadata_filters.items()
    ]
    return models.Filter(must=conditions)
