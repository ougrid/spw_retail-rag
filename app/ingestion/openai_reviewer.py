"""LLM-backed reviewer for normalization cluster suggestions."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence

import structlog
from openai import OpenAI

from app.ingestion.normalizer import ClusterSuggestion, NameReviewer

logger = structlog.get_logger(__name__)

REVIEW_SYSTEM_PROMPT = """
You review mall name normalization clusters.
Decide whether each cluster contains variants of the same mall name.
Choose a clean, user-facing canonical name for approved clusters.
Prefer the official branded form when obvious, such as ICONSIAM in uppercase.
Return JSON only.
""".strip()


def _default_review(clusters: Sequence[ClusterSuggestion]) -> dict[str, list[str]]:
    return {
        cluster.canonical_name: list(cluster.variants)
        for cluster in clusters
        if len(cluster.variants) > 1
    }


def _extract_json_object(content: str) -> str:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Reviewer response did not contain a JSON object")
    return content[start : end + 1]


class OpenAINameReviewer(NameReviewer):
    """Use an OpenAI chat model to review and canonicalize cluster suggestions."""

    def __init__(
        self,
        api_key: str,
        model: str,
        client: OpenAI | None = None,
        max_retries: int = 2,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        self._client = client or OpenAI(api_key=api_key)
        self._model = model
        self._max_retries = max_retries
        self._retry_delay_seconds = retry_delay_seconds

    def review(self, clusters: Sequence[ClusterSuggestion]) -> dict[str, list[str]]:
        if not clusters:
            return {}

        user_prompt = self._build_user_prompt(clusters)
        attempt = 0
        last_error: Exception | None = None

        while attempt <= self._max_retries:
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
                reviewed = self._parse_response(content, clusters)
                logger.info(
                    "normalization_clusters_reviewed_by_llm",
                    model=self._model,
                    cluster_count=len(clusters),
                    reviewed_count=len(reviewed),
                    attempt=attempt + 1,
                )
                return reviewed
            except Exception as error:
                last_error = error
                logger.warning(
                    "normalization_llm_review_failed",
                    model=self._model,
                    attempt=attempt + 1,
                    error=str(error),
                )
                if attempt == self._max_retries:
                    break
                time.sleep(self._retry_delay_seconds)
                attempt += 1

        logger.warning(
            "normalization_llm_review_fallback",
            model=self._model,
            cluster_count=len(clusters),
            error=str(last_error) if last_error else "unknown",
        )
        return _default_review(clusters)

    def _build_user_prompt(self, clusters: Sequence[ClusterSuggestion]) -> str:
        cluster_payload = [
            {
                "proposed_canonical_name": cluster.canonical_name,
                "variants": list(cluster.variants),
            }
            for cluster in clusters
        ]
        return (
            "Review the following mall-name clusters and decide which should be approved.\n"
            "Return a JSON object in this exact shape:\n"
            '{"clusters":[{"approved":true,"canonical_name":"...","variants":["...","..."]}]}'
            "\nDo not include commentary.\n\n"
            f"Clusters:\n{json.dumps(cluster_payload, ensure_ascii=False, indent=2)}"
        )

    def _parse_response(
        self,
        content: str,
        clusters: Sequence[ClusterSuggestion],
    ) -> dict[str, list[str]]:
        payload = json.loads(_extract_json_object(content))
        items = payload.get("clusters", [])
        if not isinstance(items, list):
            raise ValueError("Reviewer JSON must contain a 'clusters' list")

        known_variants = {
            variant
            for cluster in clusters
            for variant in cluster.variants
        }
        reviewed: dict[str, list[str]] = {}

        for item in items:
            if not isinstance(item, dict):
                continue
            if not bool(item.get("approved", False)):
                continue

            canonical_name = str(item.get("canonical_name", "")).strip()
            variants = [
                str(variant).strip()
                for variant in item.get("variants", [])
                if str(variant).strip() in known_variants
            ]
            variants = list(dict.fromkeys(variants))
            if canonical_name and len(variants) > 1:
                reviewed[canonical_name] = variants

        if not reviewed:
            raise ValueError("Reviewer did not return any valid approved clusters")
        return reviewed