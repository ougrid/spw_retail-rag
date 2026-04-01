"""Embedding utilities backed by the OpenAI API."""

from __future__ import annotations

from collections.abc import Sequence

import structlog
from openai import OpenAI

logger = structlog.get_logger(__name__)


class EmbeddingClient:
    """Thin wrapper around the OpenAI embeddings API."""

    def __init__(self, api_key: str, model: str, dimensions: int) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []

        response = self._client.embeddings.create(
            model=self._model,
            input=list(texts),
            dimensions=self._dimensions,
        )
        embeddings = [item.embedding for item in response.data]
        logger.info("embeddings_created", count=len(embeddings), model=self._model)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query string."""
        embeddings = self.embed_texts([text])
        if not embeddings:
            raise ValueError("Expected one embedding for query input")
        return embeddings[0]
