from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.places.location_store import (
    assign_db_places,
    build_location_points_from_media,
    cluster_location_points,
    create_place,
)


def test_assign_db_places_creates_media_and_event_place_links(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    media_id = repository.add_media_item(
        id="media_place",
        file_path="/local/fake/place.jpg",
        file_name="place.jpg",
        file_hash="hash-place",
        captured_at="2025-01-02T10:00:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
    )
    for index, offset in enumerate((0.0001, 0.0002), start=2):
        repository.add_media_item(
            id=f"media_place_{index}",
            file_path=f"/local/fake/place_{index}.jpg",
            file_name=f"place_{index}.jpg",
            file_hash=f"hash-place-{index}",
            captured_at=f"2025-01-0{index}T10:00:00+09:00",
            gps_lat=35.0 + offset,
            gps_lon=139.0 + offset,
        )
    event_id = repository.add_event(
        id="event_place",
        date="2025-01-02",
        start_time="10:00:00",
        title="GPS付き写真イベント",
        confidence=0.7,
    )
    repository.add_event_evidence(event_id=event_id, evidence_type="photo", evidence_id=media_id)
    build_location_points_from_media(repository, dry_run=False)
    cluster_report = cluster_location_points(repository, eps_meters=100, min_samples=3, dry_run=False)
    cluster_id = cluster_report.clusters[0]["id"]
    create_place(
        repository,
        place_id="place_test",
        display_name="テスト場所",
        public_name="テスト周辺",
        category="station",
        privacy_level="public_label",
        cluster_id=cluster_id,
    )

    dry = assign_db_places(repository, start_date="2025-01-01", end_date="2025-01-31", dry_run=True)
    report = assign_db_places(repository, start_date="2025-01-01", end_date="2025-01-31", dry_run=False)
    event = repository.get_event(event_id, include_hidden=True)

    assert dry.media_links == 3
    assert dry.event_links == 1
    assert report.media_links == 3
    assert report.event_links == 1
    assert event is not None
    assert event["location_name"] == "テスト場所"
    with connect(repository.db_path) as connection:
        media_place_count = connection.execute("SELECT COUNT(*) FROM media_places").fetchone()[0]
        event_place_count = connection.execute("SELECT COUNT(*) FROM event_places").fetchone()[0]
    assert media_place_count == 3
    assert event_place_count == 1
