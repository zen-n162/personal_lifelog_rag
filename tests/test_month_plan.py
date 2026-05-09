from __future__ import annotations

from pathlib import Path

import pytest

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.embeddings.repository import MediaEmbeddingRepository
from personal_lifelog_rag.rollout.monthly_rollout import month_plan, parse_month


def test_month_plan_returns_target_counts(tmp_path: Path) -> None:
    repository = _seed_month_repository(tmp_path)

    plan = month_plan(repository, month="2025-02", limit=100)

    assert plan["start_date"] == "2025-02-01"
    assert plan["end_date"] == "2025-02-28"
    assert plan["counts"]["photos_count"] == 2
    assert plan["counts"]["gps_photos_count"] == 1
    assert plan["counts"]["line_messages_count"] == 1
    assert plan["counts"]["call_events_count"] == 1
    assert plan["counts"]["events_count"] == 1
    assert plan["counts"]["media_vlm"]["success"] == 1
    assert plan["counts"]["media_embeddings"]["success_media"] == 1
    assert plan["recommended_limits"]["vlm_limit"] == 1
    assert plan["recommended_limits"]["embedding_limit"] == 1


def test_parse_month_rejects_invalid_format() -> None:
    with pytest.raises(ValueError):
        parse_month("2025-2")
    with pytest.raises(ValueError):
        parse_month("2025-13")


def _seed_month_repository(tmp_path: Path) -> LifelogRepository:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    for index in range(1, 3):
        repository.add_media_item(
            id=f"media_feb_{index}",
            file_path=str(tmp_path / f"feb_{index}.jpg"),
            file_name=f"feb_{index}.jpg",
            file_hash=f"hash-feb-{index}",
            media_type="image",
            captured_at=f"2025-02-0{index}T10:00:00+09:00",
            gps_lat=35.0 if index == 1 else None,
            gps_lon=139.0 if index == 1 else None,
        )
    repository.add_line_message(id="line_feb", sent_at="2025-02-01T12:00:00+09:00", sender="S", text="hello")
    repository.upsert_line_call_events(
        [
            {
                "message_id": "line_feb",
                "chat_id": "chat",
                "sent_at": "2025-02-01T12:00:00+09:00",
                "sender": "S",
                "call_status": "completed",
                "duration_sec": 60,
                "raw_text_short": "call",
            }
        ]
    )
    repository.add_event(
        id="event_feb",
        date="2025-02-01",
        start_time="2025-02-01T10:00:00+09:00",
        end_time="2025-02-01T11:00:00+09:00",
        title="写真撮影の記録",
        summary="summary",
        confidence=0.7,
    )
    repository.upsert_media_vlm(media_id="media_feb_1", caption="caption", status="success", vlm_engine="qwen3_vl_transformers")
    MediaEmbeddingRepository(db_path).upsert_embedding(
        media_id="media_feb_1",
        embedding_type="image",
        embedding_model="qwen",
        vector=[0.1, 0.2],
        status="success",
    )
    return repository

