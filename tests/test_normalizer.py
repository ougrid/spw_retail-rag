import pandas as pd

from app.ingestion.normalizer import (
    ClusterSuggestion,
    apply_name_mappings,
    cluster_names,
    detect_unknown_names,
    flatten_mappings,
    load_name_mappings,
    review_clusters,
    save_name_mappings,
)


class StubReviewer:
    def review(self, clusters):
        return {
            "ICONSIAM": ["Icon Siam", "icon-siam", "ICONSIAM"],
            "Siam Center": ["Siam Center", "Siam-Center"],
        }


def test_cluster_names_groups_known_variants_by_text_similarity():
    suggestions = cluster_names(
        ["Icon Siam", "icon-siam", "ICONSIAM", "Siam Center", "Siam-Center"],
        similarity_threshold=0.75,
    )

    grouped = {suggestion.canonical_name: set(suggestion.variants) for suggestion in suggestions}

    assert {"Icon Siam", "icon-siam", "ICONSIAM"} in grouped.values()
    assert {"Siam Center", "Siam-Center"} in grouped.values()


def test_cluster_names_supports_embedding_similarity_override():
    suggestions = cluster_names(
        ["Alpha Plaza", "Beta Plaza"],
        similarity_threshold=0.99,
        embeddings={
            "Alpha Plaza": [1.0, 0.0],
            "Beta Plaza": [1.0, 0.0],
        },
    )

    assert suggestions == [
        ClusterSuggestion(canonical_name="Alpha Plaza", variants=("Alpha Plaza", "Beta Plaza"))
    ]


def test_review_clusters_uses_external_reviewer_when_available():
    clusters = [ClusterSuggestion(canonical_name="Icon Siam", variants=("ICONSIAM", "Icon Siam", "icon-siam"))]

    reviewed = review_clusters(clusters, reviewer=StubReviewer())

    assert reviewed["ICONSIAM"] == ["Icon Siam", "icon-siam", "ICONSIAM"]


def test_flatten_save_and_load_mappings_round_trip(tmp_path):
    canonical_to_variants = {
        "ICONSIAM": ["Icon Siam", "icon-siam", "ICONSIAM"],
        "Siam Center": ["Siam Center", "Siam-Center"],
    }
    mappings = flatten_mappings(canonical_to_variants)
    mapping_path = tmp_path / "name_mappings.json"

    save_name_mappings(mappings, mapping_path)
    loaded = load_name_mappings(mapping_path)

    assert loaded == mappings
    assert loaded["icon-siam"] == "ICONSIAM"
    assert loaded["Siam-Center"] == "Siam Center"


def test_apply_name_mappings_updates_dataframe_values():
    df = pd.DataFrame({"mall_name": ["Icon Siam", "Siam-Center", "Siam Paragon"]})
    mappings = {
        "Icon Siam": "ICONSIAM",
        "Siam-Center": "Siam Center",
    }

    normalized_df = apply_name_mappings(df, mappings)

    assert normalized_df["mall_name"].tolist() == ["ICONSIAM", "Siam Center", "Siam Paragon"]


def test_detect_unknown_names_returns_unmapped_values():
    mappings = {
        "Icon Siam": "ICONSIAM",
        "ICONSIAM": "ICONSIAM",
    }

    unknown = detect_unknown_names(["Icon Siam", "Siam-Center", "ICONSIAM", "Siam Paragon"], mappings)

    assert unknown == ["Siam Paragon", "Siam-Center"]
