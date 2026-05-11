from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.places.location_store import create_place
from personal_lifelog_rag.reporting.report_builder import build_report
from personal_lifelog_rag.reporting.schemas import ReportOptions


def test_private_place_display_name_is_redacted_from_public_report_data(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    event_id = repository.add_event(
        id="event_private_place",
        date="2025-01-10",
        start_time="09:00:00",
        title="位置情報付き写真の記録",
        summary="private place test",
        location_name="具体的な自宅名",
        confidence=0.7,
    )
    create_place(
        repository,
        place_id="place_home",
        display_name="具体的な自宅名",
        category="home",
        privacy_level="private",
    )
    with connect(repository.db_path) as connection:
        connection.execute(
            "INSERT INTO event_places (event_id, place_id, source, confidence) VALUES (?, ?, 'manual', 0.9)",
            (event_id, "place_home"),
        )
        connection.commit()

    report = build_report(
        repository,
        ReportOptions(start_date="2025-01-01", end_date="2025-01-31", mode="public", include_examples=False),
    )
    payload = str(report["data"]) + report["markdown"]

    assert "具体的な自宅名" not in payload
    assert "latitude" not in payload.lower()
    assert "longitude" not in payload.lower()
