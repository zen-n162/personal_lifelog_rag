from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.faces.person_service import create_person
from personal_lifelog_rag.line.person_links import link_line_speaker_to_person
from personal_lifelog_rag.people.integration import build_event_people, list_event_people


def test_line_speaker_links_build_event_people(tmp_path) -> None:
    repository = _seed_event_with_line(tmp_path)

    dry = build_event_people(repository, start_date="2025-02-01", end_date="2025-02-28", dry_run=True)
    assert dry["line_speaker_candidates"] == 1
    assert dry["would_insert"] == 1
    assert repository.stats()["event_people"] == 0

    report = build_event_people(repository, start_date="2025-02-01", end_date="2025-02-28", dry_run=False, yes=True)
    assert report["inserted"] == 1
    rows = list_event_people(repository, date_value="2025-02-14")
    assert rows[0]["event_id"] == "event_people_line"
    assert rows[0]["source"] == "line_speaker"
    assert rows[0]["line_count"] == 1
    assert rows[0]["confidence"] == 0.70


def test_face_and_line_same_event_builds_combined_event_people(tmp_path) -> None:
    repository = _seed_event_with_line(tmp_path)
    repository.add_media_item(
        id="media_event_face",
        file_path="/local/fake/media_event_face.jpg",
        file_name="media_event_face.jpg",
        file_hash="hash-event-face",
        media_type="image",
        captured_at="2025-02-14T12:00:00+09:00",
    )
    repository.add_event_evidence(event_id="event_people_line", evidence_type="photo", evidence_id="media_event_face")
    person_id = repository._fetch_all("SELECT id FROM persons", [])[0]["id"]
    from personal_lifelog_rag.db.repository import connect

    with connect(repository.db_path) as connection:
        connection.execute(
            """
            INSERT INTO media_people (
                media_id, person_id, source, confidence, face_id,
                face_cluster_id, verified_by_user
            )
            VALUES ('media_event_face', ?, 'face_cluster', 0.85, NULL, NULL, 1)
            """,
            [person_id],
        )
        connection.commit()

    report = build_event_people(repository, start_date="2025-02-01", end_date="2025-02-28", dry_run=False, yes=True)
    assert report["source_counts"] == {"combined": 1, "face": 1, "line_speaker": 1}
    rows = list_event_people(repository, date_value="2025-02-14")
    by_source = {row["source"]: row for row in rows}
    assert by_source["combined"]["confidence"] == 0.90
    assert by_source["combined"]["media_count"] == 1
    assert by_source["combined"]["line_count"] == 1


def _seed_event_with_line(tmp_path) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    person = create_person(repository, name="人物テストEvent", public_name="人物A", privacy_level="public_alias")
    repository.add_line_message(
        id="line_event_people",
        chat_id="chat_event_people",
        source_file="dummy_line.txt",
        sent_at="2025-02-14T09:00:00+09:00",
        sender="SpeakerEvent",
        text="short dummy",
    )
    link_line_speaker_to_person(
        repository,
        chat_id="chat_event_people",
        speaker_name="SpeakerEvent",
        person_id=person["id"],
        yes=True,
    )
    repository.add_event(
        id="event_people_line",
        date="2025-02-14",
        start_time="09:00:00",
        title="Dummy event",
    )
    repository.add_event_evidence(event_id="event_people_line", evidence_type="line", evidence_id="line_event_people")
    return repository
