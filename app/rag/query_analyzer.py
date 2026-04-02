"""Query analysis helpers for metadata-aware retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd
import structlog

from app.ingestion.cleaner import clean_shop_data
from app.ingestion.loader import load_csv
from app.ingestion.normalizer import apply_name_mappings, load_name_mappings

logger = structlog.get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

PRODUCT_CATEGORY_KEYWORDS = {
    "Sports": {
        "shoe",
        "shoes",
        "sneaker",
        "sneakers",
        "footwear",
        "athletic",
        "sport",
        "sports",
    },
    "Jewelry": {"watch", "watches", "jewelry", "jewellery", "luxury"},
    "Electronics": {
        "electronics",
        "electronic",
        "gadget",
        "gadgets",
        "phone",
        "phones",
        "laptop",
        "laptops",
        "computer",
        "computers",
    },
    "Beauty": {"beauty", "cosmetic", "cosmetics", "makeup", "skincare"},
    "Cafe": {"cafe", "coffee", "pastry", "pastries", "drink", "drinks"},
    "Books": {"book", "books", "magazine", "magazines", "read"},
    "Supermarket": {"supermarket", "grocery", "groceries", "market"},
    "Fashion": {"fashion", "clothes", "clothing", "apparel", "wear"},
}


def _normalize_text(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


@dataclass(frozen=True)
class QueryAnalysis:
    metadata_filters: dict[str, str]
    inferred_filters: dict[str, str]


class QueryAnalyzer:
    """Infer metadata filters from natural-language retail queries."""

    def __init__(
        self,
        mall_aliases: dict[str, str] | None = None,
        shop_aliases: dict[str, str] | None = None,
        category_keywords: dict[str, set[str]] | None = None,
    ) -> None:
        self._mall_aliases = mall_aliases or {}
        self._shop_aliases = shop_aliases or {}
        self._category_keywords = category_keywords or PRODUCT_CATEGORY_KEYWORDS

    @classmethod
    def from_paths(cls, data_csv_path: str, name_mappings_path: str) -> "QueryAnalyzer":
        raw_df = load_csv(data_csv_path)
        cleaned_df = clean_shop_data(raw_df)
        mappings = load_name_mappings(name_mappings_path)
        normalized_df = apply_name_mappings(cleaned_df, mappings)
        return cls(
            mall_aliases=_build_mall_aliases(cleaned_df, normalized_df, mappings),
            shop_aliases=_build_shop_aliases(normalized_df),
        )

    def analyze(
        self, query: str, explicit_filters: dict[str, str] | None = None
    ) -> QueryAnalysis:
        merged_filters = dict(explicit_filters or {})
        inferred_filters: dict[str, str] = {}

        if not query.strip():
            return QueryAnalysis(
                metadata_filters=merged_filters,
                inferred_filters=inferred_filters,
            )

        if "mall_name" not in merged_filters:
            mall_name = self._match_alias(query, self._mall_aliases)
            if mall_name:
                inferred_filters["mall_name"] = mall_name
                merged_filters["mall_name"] = mall_name

        if "shop_name" not in merged_filters:
            shop_name = self._match_alias(query, self._shop_aliases)
            if shop_name:
                inferred_filters["shop_name"] = shop_name
                merged_filters["shop_name"] = shop_name

        if "category" not in merged_filters:
            category = self._match_category(query)
            if category:
                inferred_filters["category"] = category
                merged_filters["category"] = category

        logger.info(
            "query_analyzed",
            query=query,
            explicit_filters=explicit_filters or {},
            inferred_filters=inferred_filters,
            merged_filters=merged_filters,
        )
        return QueryAnalysis(
            metadata_filters=merged_filters,
            inferred_filters=inferred_filters,
        )

    def _match_alias(self, query: str, aliases: dict[str, str]) -> str | None:
        normalized_query = _normalize_text(query)
        matches = [
            (len(alias), canonical)
            for alias, canonical in aliases.items()
            if alias and alias in normalized_query
        ]
        if not matches:
            return None
        matches.sort(reverse=True)
        return matches[0][1]

    def _match_category(self, query: str) -> str | None:
        query_tokens = set(_TOKEN_PATTERN.findall(query.lower()))
        for category, keywords in self._category_keywords.items():
            if query_tokens & keywords:
                return category
        return None


def _build_mall_aliases(
    cleaned_df: pd.DataFrame,
    normalized_df: pd.DataFrame,
    mappings: dict[str, str],
) -> dict[str, str]:
    aliases: dict[str, str] = {}

    for raw_name in cleaned_df["mall_name"].dropna().unique().tolist():
        canonical = mappings.get(str(raw_name), str(raw_name))
        aliases[_normalize_text(str(raw_name))] = canonical

    for canonical_name in normalized_df["mall_name"].dropna().unique().tolist():
        aliases[_normalize_text(str(canonical_name))] = str(canonical_name)

    return aliases


def _build_shop_aliases(normalized_df: pd.DataFrame) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for shop_name in normalized_df["shop_name"].dropna().unique().tolist():
        aliases[_normalize_text(str(shop_name))] = str(shop_name)
    return aliases