from __future__ import annotations

import json
from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.reporting.collectors import collect_report_data
from personal_lifelog_rag.reporting.schemas import ReportOptions


def test_collect_report_data_includes_core_stats_and_eval_run(tmp_path: Path) -> None:
    repository = _report_repository(tmp_path)
    eval_run = tmp_path / "eval.json"
    eval_run.write_text(
        json.dumps(
            {
                "run_id": "eval_dummy",
                "summary": {"cases": 1, "passed": 1, "failed": 0, "skipped": 0},
                "by_type": {"date_qa": {"total": 1, "passed": 1, "failed": 0, "skipped": 0}},
                "ranking_metrics": {"top1_accuracy": 1.0},
                "safety_metrics": {"forbidden_phrase_violations": 0},
            }
        ),
        encoding="utf-8",
    )

    data = collect_report_data(
        repository,
        ReportOptions(start_date="2024-12-24", end_date="2024-12-24", eval_run=eval_run),
    )

    assert data["db_summary"]["media_items"] == 1
    assert data["event_stats"]["total_events"] == 1
    assert data["ocr_stats"]["total_media_ocr"] == 1
    assert data["vlm_stats"]["total_media_vlm"] == 1
    assert data["embedding_stats"]["total"] == 1
    assert data["call_stats"]["total"] == 1
    assert data["private_eval"]["summary"]["passed"] == 1


def _report_repository(tmp_path: Path) -> LifelogRepository:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"dummy")
    repository.add_media_item(
        id="media_report",
        file_path=str(image_path),
        file_name="photo.jpg",
        file_hash="hash-report",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
    )
    repository.add_line_message(
        id="line_report",
        chat_id="chat",
        source_file="chat.txt",
        sent_at="2024-12-24T10:05:00+09:00",
        sender="sender",
        text="dummy call",
    )
    repository.add_event(
        id="event_report",
        date="2024-12-24",
        start_time="10:00:00",
        end_time="11:00:00",
        title="新宿に行った可能性",
        summary="dummy",
        confidence=0.6,
    )
    repository.add_event_evidence(event_id="event_report", evidence_type="photo", evidence_id="media_report", weight=0.8)
    repository.add_event_evidence(event_id="event_report", evidence_type="line", evidence_id="line_report", weight=0.8)
    repository.upsert_media_ocr(
        media_id="media_report",
        ocr_text="新宿",
        ocr_text_redacted="新宿",
        ocr_engine="fake",
        status="success",
        analysis_version="ocr_v1",
    )
    repository.upsert_media_vlm(
        media_id="media_report",
        caption="料理の可能性",
        short_caption="料理",
        scene_tags=["restaurant"],
        object_tags=[],
        activity_tags=["meal_possible"],
        location_cues=[],
        food_cues=["meal_possible"],
        safety_flags=[],
        vlm_engine="fake",
        model_name="fake",
        prompt_version="lifelog_structured_tags_v1",
        confidence=0.8,
        status="success",
        analysis_version="vlm_v1",
    )
    repository.upsert_line_call_events(
        [
            {
                "message_id": "line_report",
                "chat_id": "chat",
                "sent_at": "2024-12-24T10:05:00+09:00",
                "sender": "sender",
                "call_status": "completed",
                "duration_sec": 60,
                "raw_text_short": "call",
            }
        ]
    )
    from personal_lifelog_rag.embeddings.repository import MediaEmbeddingRepository

    MediaEmbeddingRepository(db_path).upsert_embedding(
        media_id="media_report",
        embedding_type="image",
        embedding_model="fake",
        vector=[1.0, 0.0],
        status="success",
    )
    return repository

