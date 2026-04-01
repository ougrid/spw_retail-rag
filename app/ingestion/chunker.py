"""Chunking utilities for shop documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ChunkConfig:
    """Configuration for document chunk generation."""

    strategy: Literal["single", "hierarchical"] = "single"
    max_chunk_tokens: int = 256
    overlap_tokens: int = 50
    include_metadata_in_text: bool = True


@dataclass(frozen=True)
class ChunkDocument:
    """A single document chunk ready for embedding."""

    chunk_id: str
    text: str
    metadata: dict[str, str]


def _token_windows(tokens: list[str], max_tokens: int, overlap_tokens: int) -> list[list[str]]:
    if not tokens:
        return []

    step = max(1, max_tokens - overlap_tokens)
    windows: list[list[str]] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + max_tokens]
        if not window:
            continue
        windows.append(window)
        if start + max_tokens >= len(tokens):
            break
    return windows


def build_shop_summary_text(record: pd.Series, include_metadata_in_text: bool = True) -> str:
    """Build the canonical summary text for a shop record."""
    summary = (
        f"{record['shop_name']} is a {record['category']} shop located on floor {record['floor']} "
        f"of {record['mall_name']}. {record['description']} "
        f"Open from {record['open_time']} to {record['close_time']}."
    )
    if not include_metadata_in_text:
        return record["description"]
    return " ".join(summary.split())


def _base_metadata(record: pd.Series) -> dict[str, str]:
    return {
        "mall_name": str(record["mall_name"]),
        "shop_name": str(record["shop_name"]),
        "category": str(record["category"]),
        "floor": str(record["floor"]),
        "open_time": str(record["open_time"]),
        "close_time": str(record["close_time"]),
    }


def chunk_shop_records(df: pd.DataFrame, config: ChunkConfig | None = None) -> list[ChunkDocument]:
    """Convert shop records into single or hierarchical chunks."""
    chunk_config = config or ChunkConfig()
    documents: list[ChunkDocument] = []

    for row_index, (_, record) in enumerate(df.iterrows()):
        base_metadata = _base_metadata(record)
        parent_chunk_id = f"shop-{row_index}-summary"
        summary_text = build_shop_summary_text(record, chunk_config.include_metadata_in_text)

        summary_doc = ChunkDocument(
            chunk_id=parent_chunk_id,
            text=summary_text,
            metadata={**base_metadata, "chunk_type": "summary", "parent_chunk_id": ""},
        )
        documents.append(summary_doc)

        if chunk_config.strategy == "single":
            continue

        description_tokens = str(record["description"]).split()
        detail_windows = _token_windows(
            description_tokens,
            max_tokens=chunk_config.max_chunk_tokens,
            overlap_tokens=chunk_config.overlap_tokens,
        )

        for detail_index, window in enumerate(detail_windows):
            detail_text = " ".join(window).strip()
            if not detail_text or detail_text == str(record["description"]).strip():
                continue
            documents.append(
                ChunkDocument(
                    chunk_id=f"shop-{row_index}-detail-{detail_index}",
                    text=detail_text,
                    metadata={
                        **base_metadata,
                        "chunk_type": "detail",
                        "parent_chunk_id": parent_chunk_id,
                    },
                )
            )

    logger.info(
        "shop_records_chunked",
        strategy=chunk_config.strategy,
        input_rows=len(df),
        output_chunks=len(documents),
    )
    return documents
