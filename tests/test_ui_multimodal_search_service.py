from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.ui.multimodal_search_service import (
    detail_values,
    mark_search_result_for_review,
    multimodal_search_for_ui,
    search_result_detail_for_ui,
)


def test_multimodal_search_service_returns_table_and_detail(tmp_path: Path) -> None:
    repository = _seed_search_repository(tmp_path)

    payload = multimodal_search_for_ui(repository, query="ご飯を食べた写真", backend="vlm_sql", limit=10)
    detail = search_result_detail_for_ui(repository, "media_ui_search")

    assert payload["rows"][0][1] == "media_ui_search"
    assert payload["rows"][0][6] in {"medium", "strong"}
    assert "gps_lat" not in str(payload)
    assert "Meal candidate" in detail["caption"]
    assert "MENU" in detail["ocr_text"]
    assert len(detail_values(detail)) == 8


def test_multimodal_search_service_respects_hidden_and_not_searchable(tmp_path: Path) -> None:
    repository = _seed_search_repository(tmp_path)
    mark_search_result_for_review(repository, "media_ui_search", "not_searchable")

    payload = multimodal_search_for_ui(repository, query="ご飯を食べた写真", backend="vlm_sql", limit=10)

    assert payload["rows"] == []


def test_multimodal_search_quick_review_action_updates_detail(tmp_path: Path) -> None:
    repository = _seed_search_repository(tmp_path)

    detail = mark_search_result_for_review(repository, "media_ui_search", "wrong")

    assert detail["review_status"] == "wrong"
    payload = multimodal_search_for_ui(repository, query="ご飯を食べた写真", backend="vlm_sql", limit=10)
    assert payload["rows"] == []


def _seed_search_repository(tmp_path: Path) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_media_item(
        id="media_ui_search",
        file_path=str(tmp_path / "food.jpg"),
        file_name="food.jpg",
        file_hash="hash-ui-search",
        media_type="image",
        captured_at="2024-12-24T15:53:00+09:00",
        thumbnail_path=str(tmp_path / "thumb.jpg"),
        gps_lat=35.0,
        gps_lon=139.0,
    )
    repository.upsert_media_vlm(
        media_id="media_ui_search",
        caption="Meal candidate on a table",
        short_caption="Meal candidate",
        food_cues=["meal_possible", "rice_possible"],
        vlm_engine="qwen3_vl_transformers",
        status="success",
    )
    repository.upsert_media_ocr(media_id="media_ui_search", ocr_text="MENU", ocr_text_redacted="MENU", status="success")
    event_id = repository.add_event(
        id="event_ui_search",
        date="2024-12-24",
        start_time="15:50:00",
        end_time="16:10:00",
        title="食事・カフェの可能性",
        summary="候補",
        confidence=0.6,
    )
    repository.add_event_evidence(event_id=event_id, evidence_type="photo", evidence_id="media_ui_search")
    repository.add_event_evidence(event_id=event_id, evidence_type="vlm", evidence_id="media_ui_search")
    return repository
