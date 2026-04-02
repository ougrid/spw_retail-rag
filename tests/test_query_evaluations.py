import json
from pathlib import Path

from app.guardrails.input_guard import InputGuard
from app.rag.query_analyzer import QueryAnalyzer


def _load_cases():
    fixture_path = Path(__file__).parent / "fixtures" / "query_evaluation_cases.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_query_evaluation_fixtures_cover_supported_retail_queries():
    analyzer = QueryAnalyzer.from_paths("data/shops.csv", "data/name_mappings.json")
    guard = InputGuard()

    for case in _load_cases():
        query = case["query"]
        expected_in_scope = case["expected_in_scope"]
        expected_filters = case["expected_filters"]

        input_result = guard.evaluate(query)
        analysis = analyzer.analyze(query)

        assert input_result.in_scope is expected_in_scope, case["name"]
        for key, value in expected_filters.items():
            assert analysis.inferred_filters.get(key) == value, case["name"]