from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.faces.person_service import create_person
from personal_lifelog_rag.people.integration import format_people_stats, people_stats


def test_people_stats_counts_sources_and_public_labels(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    person = create_person(repository, name="Private Stats Name", public_name="人物A", privacy_level="public_alias")
    repository.add_media_item(
        id="media_stats_person",
        file_path="/local/fake/media_stats_person.jpg",
        file_name="media_stats_person.jpg",
        file_hash="hash-stats-person",
        media_type="image",
        captured_at="2025-04-01T10:00:00+09:00",
    )
    repository.add_event(id="event_stats_person", date="2025-04-01", start_time="10:00:00", title="Dummy")
    with connect(repository.db_path) as connection:
        connection.execute(
            """
            INSERT INTO media_people (media_id, person_id, source, confidence, verified_by_user)
            VALUES ('media_stats_person', ?, 'manual', 1.0, 1)
            """,
            [person["id"]],
        )
        connection.execute(
            """
            INSERT INTO event_people (
                event_id, person_id, source, confidence, evidence_count, media_count, line_count
            )
            VALUES ('event_stats_person', ?, 'manual', 1.0, 1, 1, 0)
            """,
            [person["id"]],
        )
        connection.commit()

    report = people_stats(repository, start_date="2025-04-01", end_date="2025-04-30", public_mode=True)
    assert report["persons_total"] == 1
    assert report["media_people_count"] == 1
    assert report["event_people_count"] == 1
    assert report["top_persons"][0]["person_label"] == "人物A"
    text = format_people_stats(report)
    assert "人物A" in text
    assert "Private Stats Name" not in text
