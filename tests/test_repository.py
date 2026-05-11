from __future__ import annotations

import sqlite3
from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.db.schema import TABLE_NAMES


def test_initialize_creates_required_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)

    repository.initialize()

    with sqlite3.connect(db_path) as connection:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert set(TABLE_NAMES).issubset(table_names)


def test_repository_inserts_and_counts_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)

    media_item_id = repository.add_media_item(
        id="media_dummy",
        file_path="/local/photos/img001.jpg",
        file_name="img001.jpg",
        file_hash="hash-img001",
        media_type="image",
        captured_at="2024-12-24T12:00:00+09:00",
        fallback_captured_at="2024-12-24T12:00:00+09:00",
        gps_lat=35.6895,
        gps_lon=139.6917,
        camera_model="Example Camera",
        width=640,
        height=480,
        thumbnail_path="data/thumbnails/hash-img001.jpg",
    )
    line_message_id = repository.add_line_message(
        id="line_msg_dummy",
        chat_id="line_chat_dummy",
        source_file="chat.txt",
        sent_at="2024-12-24T13:00:00+09:00",
        sender="Me",
        text="Shinjuku station",
        message_type="text",
    )
    event_id = repository.add_event(
        id="event_dummy",
        date="2024-12-24",
        start_time="12:00:00",
        title="Went to Shinjuku",
        summary="Visited Shinjuku",
        location_name="Shinjuku",
    )
    repository.add_event_evidence(
        event_id=event_id,
        evidence_type="media_item",
        evidence_id=media_item_id,
        weight=0.8,
    )
    repository.add_event_evidence(
        event_id=event_id,
        evidence_type="line_message",
        evidence_id=line_message_id,
        weight=0.9,
    )

    assert repository.stats() == {
        "media_items": 1,
        "line_messages": 1,
        "events": 1,
        "event_evidence": 2,
        "line_call_events": 0,
        "media_ocr": 0,
        "media_vlm": 0,
        "media_vlm_overrides": 0,
        "media_embeddings": 0,
        "analysis_jobs": 0,
        "analysis_job_items": 0,
        "location_points": 0,
        "place_clusters": 0,
        "places": 0,
        "event_places": 0,
        "media_places": 0,
        "face_detections": 0,
        "face_detection_runs": 0,
        "face_embeddings": 0,
        "face_clusters": 0,
        "face_cluster_members": 0,
        "persons": 0,
        "person_face_clusters": 0,
        "person_aliases": 0,
        "line_speaker_links": 0,
        "person_line_mentions": 0,
        "media_people": 0,
        "event_people": 0,
        "person_event_notes": 0,
        "privacy_actions": 0,
    }


def test_media_items_are_deduplicated_by_file_hash(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)

    first_id = repository.add_media_item(
        file_path="/local/photos/a.jpg",
        file_name="a.jpg",
        file_hash="same-hash",
        media_type="image",
        fallback_captured_at="2024-12-24T12:00:00+09:00",
    )
    second_id = repository.add_media_item(
        file_path="/local/photos/b.jpg",
        file_name="b.jpg",
        file_hash="same-hash",
        media_type="image",
        fallback_captured_at="2024-12-24T12:00:00+09:00",
    )

    assert first_id == second_id
    assert repository.stats()["media_items"] == 1
