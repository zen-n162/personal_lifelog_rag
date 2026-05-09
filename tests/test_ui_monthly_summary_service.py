from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.ui.monthly_summary_service import monthly_summary_for_ui


def test_monthly_summary_service_returns_ui_tables(tmp_path: Path) -> None:
    repository = _seed_monthly_repository(tmp_path)

    payload = monthly_summary_for_ui(repository, month="2025-01")

    assert "2025-01" in payload["summary_text"]
    assert ["events", 1] in payload["metrics"]
    assert payload["representative_day_rows"][0][0] == "2025-01-05"
    assert payload["representative_event_rows"][0][2] == "食事・カフェの可能性"
    assert "gps_lat" not in str(payload)


def _seed_monthly_repository(tmp_path: Path) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_media_item(
        id="media_month_ui",
        file_path=str(tmp_path / "food.jpg"),
        file_name="food.jpg",
        file_hash="hash-month-ui",
        media_type="image",
        captured_at="2025-01-05T12:10:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
    )
    repository.upsert_media_vlm(media_id="media_month_ui", caption="meal possible", food_cues=["meal_possible"], status="success")
    repository.upsert_media_ocr(media_id="media_month_ui", ocr_text="MENU", status="success")
    repository.add_line_message(id="line_month_ui", sent_at="2025-01-05T12:00:00+09:00", sender="S", text="ご飯")
    event_id = repository.add_event(
        id="event_month_ui",
        date="2025-01-05",
        start_time="12:00:00",
        end_time="13:00:00",
        title="食事・カフェの可能性",
        summary="画像解析による候補",
        confidence=0.7,
    )
    repository.add_event_evidence(event_id=event_id, evidence_type="line", evidence_id="line_month_ui")
    repository.add_event_evidence(event_id=event_id, evidence_type="photo", evidence_id="media_month_ui")
    repository.add_event_evidence(event_id=event_id, evidence_type="vlm", evidence_id="media_month_ui")
    repository.add_event_evidence(event_id=event_id, evidence_type="ocr", evidence_id="media_month_ui")
    return repository
