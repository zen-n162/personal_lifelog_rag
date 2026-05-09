from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.timeline.event_builder import build_events


def test_build_events_uses_vlm_analysis_json_as_weak_photo_text(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_media_item(
        id="media_vlm_event",
        file_path="/local/photos/vlm_event.jpg",
        file_name="vlm_event.jpg",
        file_hash="hash-vlm-event",
        media_type="image",
        captured_at="2024-12-24T12:00:00+09:00",
    )
    repository.upsert_media_vlm(
        media_id="media_vlm_event",
        caption="ラーメンの可能性がある料理写真",
        short_caption="ラーメン写真の可能性",
        food_cues=["ramen_possible"],
        vlm_engine="unit_test_vlm",
        model_name="unit-test-vlm",
        status="success",
        confidence=0.8,
    )

    build_events(repository, start_date="2024-12-24")

    events = repository.list_events(start_date="2024-12-24", end_date="2024-12-24")
    assert events[0]["title"] == "食事・カフェの可能性"
    assert "VLM" in (events[0]["summary"] or "")
