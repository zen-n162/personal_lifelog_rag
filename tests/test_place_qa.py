from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.places.location_store import create_place
from personal_lifelog_rag.retrieval.query_router import route_query


def test_place_visit_search_uses_manual_place_label_without_gps(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_event(
        id="event_place_a",
        date="2025-01-10",
        start_time="18:00:00",
        title="場所Aに行った可能性",
        summary="手動場所ラベルに基づくイベント。",
        confidence=0.8,
    )
    create_place(
        repository,
        place_id="place_a",
        display_name="場所テストA",
        public_name="場所A",
        category="station",
        privacy_level="public_label",
        aliases=["場所A"],
        manual_verified=True,
    )
    with connect(repository.db_path) as connection:
        connection.execute(
            "INSERT INTO event_places (event_id, place_id, source, confidence) VALUES ('event_place_a', 'place_a', 'manual', 0.95)"
        )
        connection.commit()

    result = route_query(repository, "場所Aに行ったのはいつ？")

    assert result.intent == "place_visit_search"
    assert result.routing == "person-place-qa"
    assert result.results[0]["date"] == "2025-01-10"
    assert result.metadata["resolved_place_id"] == "place_a"
    assert "正確なGPS座標は表示していません" in result.answer


def test_place_photo_search_uses_media_places(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_media_item(
        id="media_place_a",
        file_path="/local/fake/place_a.jpg",
        file_name="place_a.jpg",
        file_hash="hash-place-a",
        captured_at="2025-01-11T10:00:00+09:00",
    )
    create_place(repository, place_id="place_a", display_name="場所A", category="station", aliases=["場所A"], manual_verified=True)
    with connect(repository.db_path) as connection:
        connection.execute(
            "INSERT INTO media_places (media_id, place_id, source, confidence) VALUES ('media_place_a', 'place_a', 'manual', 0.9)"
        )
        connection.commit()

    result = route_query(repository, "場所Aの写真はいつ？")

    assert result.intent == "place_photo_search"
    assert result.results[0]["media_id"] == "media_place_a"
    assert "exact" not in result.answer.lower()
