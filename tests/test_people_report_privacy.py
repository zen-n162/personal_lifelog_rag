from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.faces.person_service import create_person
from personal_lifelog_rag.people.integration import format_event_people, list_event_people
from personal_lifelog_rag.reporting.report_builder import build_report, write_report
from personal_lifelog_rag.reporting.schemas import ReportOptions


def test_public_report_does_not_include_people_display_name_or_face_ids(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    person = create_person(repository, name="Private People Report Name", public_name="人物A", privacy_level="public_alias")
    repository.add_event(id="event_people_report", date="2025-05-01", start_time="10:00:00", title="Dummy")
    with connect(repository.db_path) as connection:
        connection.execute(
            """
            INSERT INTO event_people (
                event_id, person_id, source, confidence, evidence_count, media_count, line_count
            )
            VALUES ('event_people_report', ?, 'manual', 1.0, 1, 0, 0)
            """,
            [person["id"]],
        )
        connection.commit()

    report = build_report(repository, ReportOptions(mode="public"))
    result = write_report(report, output_path=tmp_path / "public_report.md", save_json=True)
    text = result.markdown_path.read_text(encoding="utf-8")

    assert "Private People Report Name" not in text
    assert "event_people_report" not in text
    assert "face_" not in text


def test_private_event_people_display_can_show_manual_name(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    person = create_person(repository, name="Private Event Detail Name", privacy_level="private")
    repository.add_event(id="event_people_private", date="2025-05-01", start_time="10:00:00", title="Dummy")
    with connect(repository.db_path) as connection:
        connection.execute(
            """
            INSERT INTO event_people (
                event_id, person_id, source, confidence, evidence_count, media_count, line_count
            )
            VALUES ('event_people_private', ?, 'manual', 1.0, 1, 0, 0)
            """,
            [person["id"]],
        )
        connection.commit()

    rows = list_event_people(repository, event_id="event_people_private", public_mode=False)
    text = format_event_people(rows)
    assert "Private Event Detail Name" in text
