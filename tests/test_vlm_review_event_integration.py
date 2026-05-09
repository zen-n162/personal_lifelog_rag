from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.timeline.event_builder import build_events


def test_not_event_usable_vlm_result_is_not_used_by_build_events(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_media_item(
        id="media_not_event_usable",
        file_path="/local/photos/not_event_usable.jpg",
        file_name="not_event_usable.jpg",
        file_hash="hash-not-event-usable",
        media_type="image",
        captured_at="2024-12-24T12:00:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
    )
    repository.upsert_media_vlm(
        media_id="media_not_event_usable",
        caption="ラーメンの可能性がある料理写真",
        short_caption="ラーメン候補",
        food_cues=["ramen_possible"],
        vlm_engine="fake",
        status="success",
        confidence=0.9,
    )
    repository.upsert_media_vlm_override(
        media_id="media_not_event_usable",
        review_status="rejected",
        is_event_usable=False,
    )

    build_events(repository, start_date="2024-12-24", force=True)
    event = repository.list_events(start_date="2024-12-24", end_date="2024-12-24", include_hidden=True)[0]
    evidence_types = {row["evidence_type"] for row in repository.list_event_evidence(event["id"])}

    assert event["title"] != "食事・カフェの可能性"
    assert "vlm" not in evidence_types
