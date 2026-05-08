from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.ui.event_review import (
    event_detail,
    event_review_overview,
    save_event_review_override,
)


def test_event_review_overview_returns_counts_and_rows(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_review_records(repository)

    overview = event_review_overview(repository, "2024-12-24")

    assert overview["event_count"] == 1
    assert overview["photo_count"] == 1
    assert overview["line_count"] == 1
    assert overview["event_evidence_count"] == 2
    assert overview["rows"][0][0] == "event_review"


def test_event_detail_returns_limited_redacted_evidence(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_review_records(repository)

    detail = event_detail(repository, "event_review", line_limit=10, photo_limit=20)

    assert detail["evidence_summary"] == "evidence total=2, line=1, photo=1"
    assert len(detail["line_evidence"]) == 1
    assert len(detail["line_evidence"][0]["text"]) < 120
    assert "とても長いLINE本文" in detail["line_evidence"][0]["text"]
    assert len(detail["photo_evidence"]) == 1
    assert detail["photo_evidence"][0]["gps"] == "GPSあり"


def test_event_review_save_override_updates_effective_event(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_review_records(repository)

    save_event_review_override(
        repository,
        "event_review",
        title="UI修正タイトル",
        summary="UI修正要約",
        location_name="UI場所",
        tags="旅行, 食事",
        is_verified=True,
        is_pinned=True,
    )
    event = repository.get_event("event_review", include_hidden=True)

    assert event is not None
    assert event["title"] == "UI修正タイトル"
    assert event["summary"] == "UI修正要約"
    assert event["location_name"] == "UI場所"
    assert event["is_verified"] == 1
    assert event["is_pinned"] == 1
    assert "旅行" in (event["tags_json"] or "")


def _seed_review_records(repository: LifelogRepository) -> None:
    repository.add_event(
        id="event_review",
        date="2024-12-24",
        start_time="17:30:00",
        end_time="18:00:00",
        title="自動イベント",
        summary="自動要約",
        location_name="自動場所",
        confidence=0.7,
    )
    repository.add_line_message(
        id="line_review",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T17:30:00+09:00",
        sender="自分",
        text="これはとても長いLINE本文です。" * 20,
        message_type="text",
    )
    repository.add_media_item(
        id="media_review",
        file_path="/local/photos/review.jpg",
        file_name="review.jpg",
        file_hash="hash-review",
        captured_at="2024-12-24T17:40:00+09:00",
        gps_lat=10.0,
        gps_lon=20.0,
        thumbnail_path="/local/thumbnails/review.jpg",
    )
    repository.add_event_evidence(
        event_id="event_review",
        evidence_type="line",
        evidence_id="line_review",
        weight=0.9,
    )
    repository.add_event_evidence(
        event_id="event_review",
        evidence_type="photo",
        evidence_id="media_review",
        weight=0.8,
    )
