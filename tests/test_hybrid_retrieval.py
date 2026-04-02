from app.rag.query_analyzer import QueryAnalyzer
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector_store import SearchResult


def _source(
    chunk_id: str,
    text: str,
    score: float,
    shop_name: str,
    mall_name: str,
    category: str,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        text=text,
        score=score,
        metadata={
            "shop_name": shop_name,
            "mall_name": mall_name,
            "category": category,
        },
    )


class StubVectorStore:
    def __init__(self, responses_by_filter):
        self._responses_by_filter = responses_by_filter
        self.calls = []

    def search(self, query_vector, limit, metadata_filters=None, score_threshold=None):
        self.calls.append(
            {
                "limit": limit,
                "metadata_filters": metadata_filters,
                "score_threshold": score_threshold,
            }
        )
        key = tuple(sorted((metadata_filters or {}).items()))
        return self._responses_by_filter.get(key, [])


def test_hybrid_retriever_uses_inferred_filters_and_reranks_candidates():
    nike = _source(
        "nike",
        "Nike is a Sports shop located on floor 1 of ICONSIAM. Athletic footwear and apparel.",
        0.46,
        "Nike",
        "ICONSIAM",
        "Sports",
    )
    adidas = _source(
        "adidas",
        "Adidas is a Sports shop located on floor 2 of Siam Center. Sports apparel and shoes.",
        0.44,
        "Adidas",
        "Siam Center",
        "Sports",
    )
    vector_store = StubVectorStore(
        {
            tuple(sorted({"shop_name": "Nike", "category": "Sports"}.items())): [nike],
            tuple(sorted({"category": "Sports"}.items())): [nike, adidas],
            tuple(): [nike, adidas],
        }
    )
    retriever = HybridRetriever(
        vector_store=vector_store,
        query_analyzer=QueryAnalyzer(shop_aliases={"nike": "Nike"}),
        top_k=2,
    )

    retrieval = retriever.retrieve("Where can I buy Nike shoes?", [0.1, 0.2])
    results = retrieval.sources

    assert [result.metadata["shop_name"] for result in results] == ["Nike"]
    assert vector_store.calls[0]["metadata_filters"] == {
        "shop_name": "Nike",
        "category": "Sports",
    }
    assert retrieval.debug["inferred_filters"] == {
        "shop_name": "Nike",
        "category": "Sports",
    }
    assert retrieval.debug["candidates"][0]["selected"] is True


def test_hybrid_retriever_falls_back_to_less_restrictive_filter_plan():
    nike = _source(
        "nike",
        "Nike is a Sports shop located on floor 1 of ICONSIAM. Athletic footwear and apparel.",
        0.46,
        "Nike",
        "ICONSIAM",
        "Sports",
    )
    vector_store = StubVectorStore(
        {
            tuple(sorted({"shop_name": "Nike"}.items())): [nike],
            tuple(): [nike],
        }
    )
    retriever = HybridRetriever(
        vector_store=vector_store,
        query_analyzer=QueryAnalyzer(shop_aliases={"nike": "Nike"}),
        top_k=1,
    )

    retrieval = retriever.retrieve("Where can I buy Nike shoes?", [0.1, 0.2])
    results = retrieval.sources

    assert [result.metadata["shop_name"] for result in results] == ["Nike"]
    assert vector_store.calls[0]["metadata_filters"] == {
        "shop_name": "Nike",
        "category": "Sports",
    }
    assert vector_store.calls[1]["metadata_filters"] == {"shop_name": "Nike"}


def test_hybrid_retriever_applies_minimum_score_cutoff():
    weak = _source(
        "weak",
        "Generic shop information with no lexical overlap.",
        0.05,
        "Unknown",
        "Unknown Mall",
        "Misc",
    )
    vector_store = StubVectorStore({tuple(): [weak]})
    retriever = HybridRetriever(
        vector_store=vector_store,
        query_analyzer=QueryAnalyzer(category_keywords={}),
        top_k=1,
        minimum_hybrid_score=0.2,
    )

    retrieval = retriever.retrieve("Where can I buy shoes?", [0.1, 0.2])
    results = retrieval.sources

    assert results == []
    assert retrieval.debug["candidate_count"] == 1