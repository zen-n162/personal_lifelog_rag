from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.faces.person_service import create_person
from personal_lifelog_rag.places.location_store import create_place
from personal_lifelog_rag.retrieval.query_router import route_query


def test_person_place_activity_search_combines_manual_person_place_event(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    person = create_person(repository, name="人物A", public_name="人物A", privacy_level="public_alias")
    create_place(
        repository,
        place_id="place_shinjuku",
        display_name="新宿駅周辺",
        public_name="新宿周辺",
        category="station",
        privacy_level="public_label",
        aliases=["新宿"],
        manual_verified=True,
    )
    repository.add_event(
        id="event_person_place_food",
        date="2025-01-12",
        start_time="12:00:00",
        title="ご飯と移動の候補",
        summary="新宿周辺で食事をした可能性。",
        confidence=0.8,
    )
    with connect(repository.db_path) as connection:
        connection.execute(
            """
            INSERT INTO event_people (
                event_id, person_id, source, confidence, evidence_count, media_count, line_count
            )
            VALUES ('event_person_place_food', ?, 'combined', 0.9, 2, 1, 1)
            """,
            [person["id"]],
        )
        connection.execute(
            "INSERT INTO event_places (event_id, place_id, source, confidence) VALUES ('event_person_place_food', 'place_shinjuku', 'manual', 0.95)"
        )
        connection.commit()

    result = route_query(repository, "人物Aと新宿でご飯を食べた日は？")

    assert result.intent == "person_place_search"
    assert result.results[0]["event_id"] == "event_person_place_food"
    assert result.results[0]["evidence_strength"] == "strong"
    assert result.metadata["resolved_person_id"] == person["id"]
    assert result.metadata["resolved_place_id"] == "place_shinjuku"
    assert "可能性" in result.answer
    assert "確実に一緒" not in result.answer
