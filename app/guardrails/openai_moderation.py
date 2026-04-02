"""OpenAI-backed moderation client for input guardrails."""

from __future__ import annotations

import structlog
from openai import OpenAI

logger = structlog.get_logger(__name__)


class OpenAIModerationClient:
    """Thin wrapper around the OpenAI moderation API."""

    def __init__(
        self,
        api_key: str,
        model: str = "omni-moderation-latest",
        client: OpenAI | None = None,
    ) -> None:
        self._client = client or OpenAI(api_key=api_key)
        self._model = model

    def moderate(self, text: str) -> bool:
        response = self._client.moderations.create(model=self._model, input=text)
        flagged = any(result.flagged for result in response.results)
        logger.info(
            "moderation_evaluated",
            model=self._model,
            flagged=flagged,
        )
        return flagged
