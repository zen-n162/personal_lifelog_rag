from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.faces.person_service import add_person_alias, create_person
from personal_lifelog_rag.privacy_controls import person_delete, person_detach
from personal_lifelog_rag.retrieval.query_router import route_query


def test_person_detach_removes_links_but_keeps_person(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    person = _seed_person_links(repository)

    dry = person_detach(repository, person_id=person["id"], dry_run=True)
    assert dry["would_detach"]["line_speaker_links"] == 1

    result = person_detach(repository, person_id=person["id"], dry_run=False, yes=True)
    assert result["detached"]["media_people"] == 1
    with connect(repository.db_path) as connection:
        person_exists = connection.execute("SELECT 1 FROM persons WHERE id = ?", (person["id"],)).fetchone()
        line_links = connection.execute("SELECT COUNT(*) FROM line_speaker_links WHERE person_id = ?", (person["id"],)).fetchone()[0]
    assert person_exists is not None
    assert line_links == 0


def test_soft_deleted_person_is_excluded_from_qa(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    person = _seed_person_links(repository)

    before = route_query(repository, "人物テストAとLINEした日は？")
    assert "人物テストA" in before.answer

    person_delete(repository, person_id=person["id"], soft=True, dry_run=False, yes=True)
    after = route_query(repository, "人物テストAとLINEした日は？")

    assert "人物テストA" not in after.answer


def _seed_person_links(repository: LifelogRepository) -> dict:
    person = create_person(repository, name="人物テストA", public_name="人物A", privacy_level="public_alias")
    add_person_alias(repository, person_id=person["id"], alias="人物テストA")
    media_id = repository.add_media_item(
        id="media_person_privacy",
        file_path="/tmp/person_privacy.jpg",
        file_name="person_privacy.jpg",
        captured_at="2025-01-01T10:00:00",
    )
    repository.add_event(id="event_person_privacy", date="2025-01-01", start_time="10:00:00", title="Dummy")
    with connect(repository.db_path) as connection:
        connection.execute(
            """
            INSERT INTO line_messages (id, chat_id, source_file, sent_at, sender, text, message_type)
            VALUES ('line_person_privacy', 'chat_person_privacy', 'dummy', '2025-01-01T11:00:00', '人物テストA', 'dummy', 'text')
            """
        )
        connection.execute(
            """
            INSERT INTO line_speaker_links (id, chat_id, speaker_name, person_id, verified_by_user, verified_at)
            VALUES ('speaker_person_privacy', 'chat_person_privacy', '人物テストA', ?, 1, '2025-01-01T00:00:00')
            """,
            (person["id"],),
        )
        connection.execute(
            """
            INSERT INTO media_people (media_id, person_id, source, confidence, verified_by_user)
            VALUES (?, ?, 'manual', 1.0, 1)
            """,
            (media_id, person["id"]),
        )
        connection.execute(
            """
            INSERT INTO event_people (event_id, person_id, source, confidence, evidence_count)
            VALUES ('event_person_privacy', ?, 'manual', 1.0, 1)
            """,
            (person["id"],),
        )
        connection.commit()
    return person
