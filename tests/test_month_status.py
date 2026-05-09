from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.rollout.monthly_rollout import month_status


def test_month_status_returns_statistics_and_artifacts(tmp_path: Path) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_media_item(
        id="media_status",
        file_path=str(tmp_path / "status.jpg"),
        file_name="status.jpg",
        file_hash="hash-status",
        media_type="image",
        captured_at="2025-02-05T10:00:00+09:00",
    )
    repository.upsert_media_vlm(media_id="media_status", caption="caption", status="failed", vlm_engine="qwen3_vl_transformers")
    repository.upsert_media_ocr(media_id="media_status", ocr_text="OCR", status="success")
    repository.add_event(
        id="event_status",
        date="2025-02-05",
        start_time="2025-02-05T10:00:00+09:00",
        end_time="2025-02-05T11:00:00+09:00",
        title="event",
        summary="summary",
        confidence=0.5,
    )
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "lifelog_2025-02.md").write_text("# report", encoding="utf-8")
    eval_dir = tmp_path / "eval_outputs"
    eval_dir.mkdir()
    (eval_dir / "eval_202502.json").write_text("{}", encoding="utf-8")

    status = month_status(repository, month="2025-02", reports_dir=reports_dir, eval_outputs_dir=eval_dir)

    assert status["vlm"]["failed"] == 1
    assert status["ocr"]["success"] == 1
    assert status["events_count"] == 1
    assert status["report_exists"] is True
    assert status["eval_run_exists"] is True

