from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.retrieval.local_search import LocalSearchOptions, local_text_search
from personal_lifelog_rag.retrieval.query_router import route_query


def test_vlm_caption_is_searchable_as_vlm_evidence(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_vlm_record(repository)

    report = local_text_search(repository, LocalSearchOptions(query="ラーメン", intent="food_activity", limit=10))
    result = report["results"][0]

    assert result["date"] == "2024-12-24"
    assert result["vlm_match_count"] == 1
    assert "vlm" in result["evidence_types"]
    assert "ラーメン" in result["vlm_samples"][0]["caption"]


def test_qa_food_photo_query_routes_to_specific_food_term_and_vlm_evidence(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_vlm_record(repository)

    result = route_query(repository, "ラーメンを食べた写真はいつ？", limit=5)

    assert result.intent == "specific_food_search"
    assert result.routing == "multimodal-search"
    assert result.results[0]["date"] == "2024-12-24"
    assert "画像解析では" in result.answer


def _seed_vlm_record(repository: LifelogRepository) -> None:
    repository.add_media_item(
        id="media_vlm_search",
        file_path="/local/photos/ramen.jpg",
        file_name="ramen.jpg",
        file_hash="hash-vlm-search",
        media_type="image",
        captured_at="2024-12-24T19:00:00+09:00",
    )
    repository.upsert_media_vlm(
        media_id="media_vlm_search",
        caption="ラーメンの可能性がある料理写真",
        short_caption="ラーメン写真の可能性",
        food_cues=["ramen_possible"],
        vlm_engine="unit_test_vlm",
        model_name="unit-test-vlm",
        status="success",
        confidence=0.8,
    )
