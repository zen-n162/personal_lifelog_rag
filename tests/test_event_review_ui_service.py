from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.places.place_dictionary import load_place_dictionary
from personal_lifelog_rag.ui.event_review import event_detail
from personal_lifelog_rag.ui.event_review_service import ReviewQueueFilters, review_queue


def test_event_detail_does_not_expose_exact_gps(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_gps_event(repository)

    detail = event_detail(repository, "event_gps")
    serialized = str(detail)

    assert "35.123456" not in serialized
    assert "139.654321" not in serialized
    assert detail["photo_evidence"][0]["gps"] == "GPSあり"


def test_review_queue_service_does_not_require_places_yaml(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_gps_event(repository)

    places = load_place_dictionary(tmp_path / "missing_places.yaml", required=False)
    report = review_queue(repository, ReviewQueueFilters(date="2024-12-24"))

    assert places == []
    assert report["rows"][0]["location_name"] == "安全な場所名"


def _seed_gps_event(repository: LifelogRepository) -> None:
    repository.add_event(
        id="event_gps",
        date="2024-12-24",
        start_time="18:00:00",
        end_time="19:00:00",
        title="位置情報付き写真の記録",
        summary="GPS付き写真があります",
        location_name="安全な場所名",
        confidence=0.6,
    )
    repository.add_media_item(
        id="media_gps",
        file_path="/local/private/photo.jpg",
        file_name="photo.jpg",
        file_hash="hash-ui-service-gps",
        captured_at="2024-12-24T18:10:00+09:00",
        gps_lat=35.123456,
        gps_lon=139.654321,
    )
    repository.add_event_evidence(event_id="event_gps", evidence_type="photo", evidence_id="media_gps")
