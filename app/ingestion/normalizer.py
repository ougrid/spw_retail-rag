"""Intelligent name normalization utilities for ingestion."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Protocol, Sequence

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ClusterSuggestion:
    """Suggested canonical name and its grouped variants."""

    canonical_name: str
    variants: tuple[str, ...]


class NameReviewer(Protocol):
    """Protocol for an LLM-backed or rule-based cluster reviewer."""

    def review(self, clusters: Sequence[ClusterSuggestion]) -> dict[str, list[str]]:
        """Return canonical name to variants mapping."""


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _text_similarity(left: str, right: str) -> float:
    left_key = _normalize_key(left)
    right_key = _normalize_key(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0
    return SequenceMatcher(a=left_key, b=right_key).ratio()


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _canonical_name(variants: Sequence[str]) -> str:
    if not variants:
        raise ValueError("Cannot select a canonical name from an empty variant list")
    return variants[0]


def cluster_names(
    names: Sequence[str],
    similarity_threshold: float = 0.75,
    embeddings: dict[str, Sequence[float]] | None = None,
) -> list[ClusterSuggestion]:
    """Cluster semantically similar names into candidate normalization groups."""
    unique_names = [
        name.strip() for name in dict.fromkeys(names) if name and str(name).strip()
    ]
    if not unique_names:
        return []

    parents = {name: name for name in unique_names}

    def find(name: str) -> str:
        parent = parents[name]
        while parent != parents[parent]:
            parents[parent] = parents[parents[parent]]
            parent = parents[parent]
        parents[name] = parent
        return parent

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for index, left in enumerate(unique_names):
        for right in unique_names[index + 1 :]:
            if embeddings and left in embeddings and right in embeddings:
                similarity = _cosine_similarity(embeddings[left], embeddings[right])
            else:
                similarity = _text_similarity(left, right)
            if similarity >= similarity_threshold:
                union(left, right)

    grouped: dict[str, list[str]] = {}
    for name in unique_names:
        grouped.setdefault(find(name), []).append(name)

    suggestions = []
    for variants in grouped.values():
        canonical = _canonical_name(variants)
        suggestions.append(
            ClusterSuggestion(
                canonical_name=canonical, variants=tuple(sorted(variants))
            )
        )

    suggestions.sort(key=lambda suggestion: suggestion.canonical_name.lower())
    logger.info(
        "names_clustered", clusters=len(suggestions), input_count=len(unique_names)
    )
    return suggestions


def review_clusters(
    clusters: Sequence[ClusterSuggestion],
    reviewer: NameReviewer | None = None,
) -> dict[str, list[str]]:
    """Review cluster suggestions and return canonical-to-variants mapping."""
    if reviewer is not None:
        try:
            reviewed = reviewer.review(clusters)
            logger.info("clusters_reviewed", reviewed_count=len(reviewed))
            return reviewed
        except Exception as error:
            logger.warning("clusters_review_failed", error=str(error))

    reviewed = {
        cluster.canonical_name: list(cluster.variants)
        for cluster in clusters
        if len(cluster.variants) > 1
    }
    logger.info("clusters_reviewed", reviewed_count=len(reviewed), reviewer="default")
    return reviewed


def flatten_mappings(canonical_to_variants: dict[str, Sequence[str]]) -> dict[str, str]:
    """Convert canonical-to-variants mapping into a variant-to-canonical lookup."""
    flattened: dict[str, str] = {}
    for canonical, variants in canonical_to_variants.items():
        flattened[canonical] = canonical
        for variant in variants:
            flattened[variant] = canonical
    return flattened


def apply_name_mappings(
    df: pd.DataFrame,
    mappings: dict[str, str],
    column: str = "mall_name",
) -> pd.DataFrame:
    """Apply approved name mappings to a dataframe column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in dataframe")

    normalized_df = df.copy()
    normalized_df[column] = normalized_df[column].map(
        lambda value: mappings.get(value, value)
    )
    logger.info("name_mappings_applied", column=column, mapping_count=len(mappings))
    return normalized_df


def detect_unknown_names(names: Sequence[str], mappings: dict[str, str]) -> list[str]:
    """Return names that are not covered by the current mapping store."""
    unknown = sorted({name for name in names if name and name not in mappings})
    logger.info("unknown_names_detected", unknown_count=len(unknown))
    return unknown


def load_name_mappings(file_path: str | Path) -> dict[str, str]:
    """Load persisted variant-to-canonical mappings from disk."""
    path = Path(file_path)
    if not path.exists():
        logger.info("name_mappings_missing", path=str(path))
        return {}

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Name mappings file must contain a JSON object")

    mappings = {str(key): str(value) for key, value in data.items()}
    logger.info("name_mappings_loaded", path=str(path), mapping_count=len(mappings))
    return mappings


def save_name_mappings(mappings: dict[str, str], file_path: str | Path) -> None:
    """Persist variant-to-canonical mappings to disk."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(mappings.items())), handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    logger.info("name_mappings_saved", path=str(path), mapping_count=len(mappings))
