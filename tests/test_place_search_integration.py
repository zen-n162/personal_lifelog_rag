from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.places.location_store import create_place
from personal_lifelog_rag.retrieval.local_search import LocalSearchOptions, local_text_search
from personal_lifelog_rag.vlm.schemas import ImageSearchOptions
from personal_lifelog_rag.vlm.vlm_service import image_search


def test_manual_verified_place_alias_is_searchable_in_qa_local_search(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    event_id = repository.add_event(
        id="event_place",
        date="2025-01-10",
        start_time="12:00:00",
        title="写真と移動の記録",
        summary="GPSクラスタから手動確認された場所候補。",
        confidence=0.72,
    )
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
    with connect(repository.db_path) as connection:
        connection.execute(
            "INSERT INTO event_places (event_id, place_id, source, confidence) VALUES (?, ?, 'manual', 0.95)",
            (event_id, "place_shinjuku"),
        )
        connection.commit()

    report = local_text_search(repository, LocalSearchOptions(query="新宿に行ったのはいつ？", limit=5, intent="place_visit"))

    assert report["results"]
    assert report["results"][0]["date"] == "2025-01-10"
    assert "place" in report["results"][0]["evidence_types"]


def test_place_label_can_retrieve_photos_in_image_search(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_media_item(
        id="media_place",
        file_path="/local/fake/place.jpg",
        file_name="place.jpg",
        file_hash="hash-place",
        captured_at="2025-01-10T12:00:00+09:00",
    )
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
    with connect(repository.db_path) as connection:
        connection.execute(
            "INSERT INTO media_places (media_id, place_id, source, confidence) VALUES (?, ?, 'manual', 0.95)",
            ("media_place", "place_shinjuku"),
        )
        connection.commit()

    report = image_search(repository, ImageSearchOptions(query="新宿駅周辺の写真", limit=5))

    assert report["results"]
    assert report["results"][0]["media_id"] == "media_place"
    assert "place" in report["results"][0]["evidence_types"]
