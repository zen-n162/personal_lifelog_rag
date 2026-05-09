from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.ui.multimodal_search_service import mark_search_result_for_review, multimodal_search_for_ui
from personal_lifelog_rag.vlm.review_service import VlmReviewFilters, list_vlm_review_items


def test_search_result_review_action_is_visible_in_vlm_review_queue(tmp_path: Path) -> None:
    repository = _seed_repository(tmp_path)

    mark_search_result_for_review(repository, "media_review_link", "accepted")
    items = list_vlm_review_items(repository, VlmReviewFilters(review_status="accepted"))
    payload = multimodal_search_for_ui(repository, query="ステージの写真", backend="vlm_sql", limit=5)

    assert items[0]["media_id"] == "media_review_link"
    assert payload["rows"][0][14] == "accepted"
    assert "gps_lon" not in str(items)


def _seed_repository(tmp_path: Path) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_media_item(
        id="media_review_link",
        file_path=str(tmp_path / "stage.jpg"),
        file_name="stage.jpg",
        file_hash="hash-review-link",
        media_type="image",
        captured_at="2024-12-14T18:00:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
    )
    repository.upsert_media_vlm(
        media_id="media_review_link",
        caption="Performance stage possible",
        short_caption="Stage possible",
        scene_tags=["stage_possible"],
        activity_tags=["performance_possible"],
        vlm_engine="qwen3_vl_transformers",
        status="success",
    )
    return repository
