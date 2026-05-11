from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.faces.person_service import create_person
from personal_lifelog_rag.places.location_store import create_place
from personal_lifelog_rag.retrieval.query_router import route_query


def test_public_mode_hides_private_person_name_and_private_place(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    person = create_person(repository, name="Private Real Person", public_name="人物A", privacy_level="public_alias")
    create_place(
        repository,
        place_id="place_private",
        display_name="Private Lab Name",
        category="lab",
        privacy_level="private",
        aliases=["秘密場所"],
        manual_verified=True,
    )
    repository.add_event(
        id="event_private_person_place",
        date="2025-01-20",
        start_time="10:00:00",
        title="private test event",
        summary="private summary",
        confidence=0.7,
    )
    with connect(repository.db_path) as connection:
        connection.execute(
            "INSERT INTO event_people (event_id, person_id, source, confidence, evidence_count) VALUES ('event_private_person_place', ?, 'manual', 1.0, 1)",
            [person["id"]],
        )
        connection.execute(
            "INSERT INTO event_places (event_id, place_id, source, confidence) VALUES ('event_private_person_place', 'place_private', 'manual', 1.0)"
        )
        connection.commit()

    result = route_query(repository, "人物Aと秘密場所に行ったのはいつ？", public_mode=True)

    assert "Private Real Person" not in result.answer
    assert "Private Lab Name" not in result.answer
    assert "人物A" in result.answer
    assert "正確なGPS" not in result.answer or "表示していません" in result.answer
    assert result.metadata["privacy_mode"] == "public"


def test_monthly_place_summary_uses_public_safe_labels(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_event(id="event_month_place", date="2025-01-21", start_time="10:00:00", title="場所候補")
    create_place(
        repository,
        place_id="place_public",
        display_name="Private Station Name",
        public_name="場所A",
        category="station",
        privacy_level="public_label",
        aliases=["場所A"],
        manual_verified=True,
    )
    with connect(repository.db_path) as connection:
        connection.execute(
            "INSERT INTO event_places (event_id, place_id, source, confidence) VALUES ('event_month_place', 'place_public', 'manual', 1.0)"
        )
        connection.commit()

    result = route_query(repository, "2025年1月に行った場所は？", public_mode=True)

    assert result.intent == "monthly_place_summary"
    assert "場所A" in result.answer
    assert "Private Station Name" not in result.answer
    assert result.metadata["source_counts"]["event_places"] == 1
