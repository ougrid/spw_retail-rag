import pandas as pd

from app.ingestion.chunker import ChunkConfig, chunk_shop_records


def _sample_dataframe(description: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mall_name": ["ICONSIAM"],
            "shop_name": ["Nike"],
            "category": ["Sports"],
            "floor": ["1"],
            "description": [description],
            "open_time": ["10:00"],
            "close_time": ["22:00"],
        }
    )


def test_single_strategy_produces_one_summary_chunk_per_shop():
    df = _sample_dataframe("Athletic footwear and apparel.")

    documents = chunk_shop_records(df, ChunkConfig(strategy="single"))

    assert len(documents) == 1
    assert documents[0].metadata["chunk_type"] == "summary"
    assert documents[0].metadata["mall_name"] == "ICONSIAM"
    assert "Nike is a Sports shop" in documents[0].text


def test_hierarchical_strategy_adds_detail_chunks_for_long_descriptions():
    description = " ".join(f"token{i}" for i in range(12))
    df = _sample_dataframe(description)

    documents = chunk_shop_records(
        df,
        ChunkConfig(strategy="hierarchical", max_chunk_tokens=5, overlap_tokens=2),
    )

    assert len(documents) > 1
    assert documents[0].metadata["chunk_type"] == "summary"
    detail_documents = [
        doc for doc in documents if doc.metadata["chunk_type"] == "detail"
    ]
    assert detail_documents
    assert all(
        doc.metadata["parent_chunk_id"] == documents[0].chunk_id
        for doc in detail_documents
    )


def test_hierarchical_strategy_preserves_metadata_on_detail_chunks():
    description = " ".join(f"token{i}" for i in range(8))
    df = _sample_dataframe(description)

    documents = chunk_shop_records(
        df,
        ChunkConfig(strategy="hierarchical", max_chunk_tokens=4, overlap_tokens=1),
    )

    detail_document = next(
        doc for doc in documents if doc.metadata["chunk_type"] == "detail"
    )
    assert detail_document.metadata["shop_name"] == "Nike"
    assert detail_document.metadata["category"] == "Sports"
    assert detail_document.metadata["floor"] == "1"


def test_single_strategy_can_exclude_metadata_from_text():
    df = _sample_dataframe("Athletic footwear and apparel.")

    documents = chunk_shop_records(
        df,
        ChunkConfig(strategy="single", include_metadata_in_text=False),
    )

    assert documents[0].text == "Athletic footwear and apparel."
