"""End-to-end ingestion pipeline for shop data."""

from __future__ import annotations

import argparse
import json

import structlog

from app.config import get_settings
from app.ingestion.chunker import ChunkConfig, chunk_shop_records
from app.ingestion.cleaner import clean_shop_data
from app.ingestion.loader import load_csv
from app.ingestion.normalizer import (
    apply_name_mappings,
    cluster_names,
    detect_unknown_names,
    flatten_mappings,
    load_name_mappings,
    review_clusters,
    save_name_mappings,
)
from app.retrieval.embeddings import EmbeddingClient
from app.retrieval.vector_store import QdrantVectorStore

logger = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest shop data into Qdrant.")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run in automated mode using stored mappings and default suggestion review.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Qdrant collection before ingestion.",
    )
    return parser.parse_args()


def build_normalized_dataframe(auto_mode: bool) -> tuple[object, dict[str, str], list[str]]:
    settings = get_settings()
    raw_df = load_csv(settings.data_csv_path)
    cleaned_df = clean_shop_data(raw_df)

    existing_mappings = load_name_mappings(settings.name_mappings_path)
    normalized_df = apply_name_mappings(cleaned_df, existing_mappings)

    unique_names = sorted(normalized_df["mall_name"].dropna().unique().tolist())
    unknown_names = detect_unknown_names(unique_names, existing_mappings)

    if unknown_names:
        suggestions = cluster_names(unknown_names)
        reviewed = review_clusters(suggestions)
        suggested_mappings = flatten_mappings(reviewed)
        merged_mappings = {**existing_mappings, **suggested_mappings}
        save_name_mappings(merged_mappings, settings.name_mappings_path)
        normalized_df = apply_name_mappings(cleaned_df, merged_mappings)
        existing_mappings = merged_mappings

        if not auto_mode:
            logger.warning(
                "normalization_review_recommended",
                unknown_names=unknown_names,
                suggestion_count=len(suggested_mappings),
            )
            print(
                json.dumps(
                    {
                        "message": "New or unmapped names were detected. Review these suggestions in the admin UI when available.",
                        "unknown_names": unknown_names,
                        "suggested_mappings": suggested_mappings,
                    },
                    indent=2,
                )
            )

    return normalized_df, existing_mappings, unknown_names


def main() -> None:
    args = parse_args()
    settings = get_settings()

    normalized_df, mappings, unknown_names = build_normalized_dataframe(auto_mode=args.auto)
    documents = chunk_shop_records(
        normalized_df,
        ChunkConfig(
            strategy=settings.chunk_strategy,
            max_chunk_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        ),
    )

    embedding_client = EmbeddingClient(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )
    vector_store = QdrantVectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection_name,
        dimensions=settings.embedding_dimensions,
    )

    if args.recreate:
        vector_store.recreate_collection()
    else:
        vector_store.ensure_collection()

    embeddings = embedding_client.embed_texts([document.text for document in documents])
    vector_store.upsert_documents(documents, embeddings)

    logger.info(
        "ingestion_completed",
        rows=len(normalized_df),
        chunks=len(documents),
        mapping_count=len(mappings),
        unknown_name_count=len(unknown_names),
        auto_mode=args.auto,
        recreate=args.recreate,
    )


if __name__ == "__main__":
    main()
