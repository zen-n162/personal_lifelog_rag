from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository


def test_person_tables_are_created(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    stats = repository.stats()

    assert stats["persons"] == 0
    assert stats["person_face_clusters"] == 0
    assert stats["person_aliases"] == 0
    assert stats["media_people"] == 0
    assert stats["event_people"] == 0
    assert stats["person_event_notes"] == 0
