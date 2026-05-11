from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.faces.person_service import create_person
from personal_lifelog_rag.retrieval.query_router import route_query


def test_person_photo_search_uses_manual_verified_media_people(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    person = create_person(repository, name="人物A", public_name="人物A", privacy_level="public_alias")
    repository.add_media_item(
        id="media_person_a",
        file_path="/local/fake/person_a.jpg",
        file_name="person_a.jpg",
        file_hash="hash-person-a",
        captured_at="2025-01-05T12:00:00+09:00",
    )
    with connect(repository.db_path) as connection:
        connection.execute(
            """
            INSERT INTO media_people (media_id, person_id, source, confidence, verified_by_user)
            VALUES ('media_person_a', ?, 'manual', 1.0, 1)
            """,
            [person["id"]],
        )
        connection.commit()

    result = route_query(repository, "人物Aが写っている写真はいつ？")

    assert result.intent == "person_photo_search"
    assert result.routing == "person-place-qa"
    assert result.results[0]["media_id"] == "media_person_a"
    assert result.results[0]["evidence_strength"] == "medium"
    assert result.metadata["resolved_person_id"] == person["id"]
    assert "写っている可能性" in result.answer


def test_unverified_person_is_not_used_for_person_photo_search(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    person = create_person(repository, name="人物未確認", privacy_level="private")
    with connect(repository.db_path) as connection:
        connection.execute("UPDATE persons SET manual_verified = 0 WHERE id = ?", [person["id"]])
        connection.commit()

    result = route_query(repository, "人物未確認が写っている写真はいつ？")

    assert result.intent == "person_photo_search"
    assert result.results == []
    assert "手動確認済みperson" in result.answer
