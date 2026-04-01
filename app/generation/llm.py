"""LLM client wrapper for grounded answer generation."""

from __future__ import annotations

import time
from collections.abc import Sequence

import structlog
from openai import OpenAI

logger = structlog.get_logger(__name__)


class LLMClient:
    """Thin retrying wrapper around the OpenAI chat completions API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        max_retries: int = 2,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._retry_delay_seconds = retry_delay_seconds

    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        """Generate a completion from chat messages."""
        attempt = 0
        last_error: Exception | None = None

        while attempt <= self._max_retries:
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=list(messages),
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                content = response.choices[0].message.content or ""
                logger.info("llm_completion_created", model=self._model, attempt=attempt + 1)
                return content.strip()
            except Exception as error:
                last_error = error
                logger.warning(
                    "llm_completion_failed",
                    model=self._model,
                    attempt=attempt + 1,
                    error=str(error),
                )
                if attempt == self._max_retries:
                    break
                time.sleep(self._retry_delay_seconds)
                attempt += 1

        raise RuntimeError("LLM completion failed after retries") from last_error
