from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.faces.person_service import create_person
from personal_lifelog_rag.privacy_controls import person_delete
from personal_lifelog_rag.reporting.report_builder import build_report, write_report
from personal_lifelog_rag.reporting.schemas import ReportOptions


def test_public_report_omits_soft_deleted_person_display_name(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    person = create_person(repository, name="Private Deleted Report Name", public_name="人物A", privacy_level="public_alias")
    repository.add_event(id="event_deleted_person", date="2025-01-01", start_time="10:00:00", title="Dummy")
    with connect(repository.db_path) as connection:
        connection.execute(
            """
            INSERT INTO event_people (event_id, person_id, source, confidence, evidence_count)
            VALUES ('event_deleted_person', ?, 'manual', 1.0, 1)
            """,
            (person["id"],),
        )
        connection.commit()
    person_delete(repository, person_id=person["id"], soft=True, dry_run=False, yes=True)

    report = build_report(repository, ReportOptions(mode="public"))
    result = write_report(report, output_path=tmp_path / "report.md", save_json=True)
    text = result.markdown_path.read_text(encoding="utf-8")

    assert "Private Deleted Report Name" not in text
    assert "face_cluster" not in text
