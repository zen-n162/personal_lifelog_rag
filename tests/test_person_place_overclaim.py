from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.faces.person_service import create_person
from personal_lifelog_rag.retrieval.query_router import route_query


def test_person_place_qa_does_not_emit_relationship_overclaim_terms(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    person = create_person(repository, name="人物A", public_name="人物A", privacy_level="public_alias")
    repository.add_event(id="event_person", date="2025-01-01", start_time="09:00:00", title="イベント候補")
    with connect(repository.db_path) as connection:
        connection.execute(
            "INSERT INTO event_people (event_id, person_id, source, confidence, evidence_count) VALUES ('event_person', ?, 'line_speaker', 0.7, 1)",
            [person["id"]],
        )
        connection.commit()

    result = route_query(repository, "人物Aと一緒だった可能性がある日は？")

    forbidden = ["恋人", "家族", "友人", "親密", "確実に一緒"]
    assert not any(term in result.answer for term in forbidden)
    assert result.metadata["overclaim_flags"] == []
    assert "関係性は推定していません" in result.answer
