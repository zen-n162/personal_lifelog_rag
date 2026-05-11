from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.places.location_store import build_location_points_from_media, cluster_location_points, create_place, link_place_cluster
from personal_lifelog_rag.ui.place_review_service import place_cluster_detail_for_ui, place_review_queue_for_ui


def test_place_review_queue_and_detail_hide_exact_coordinates_by_default(tmp_path) -> None:
    repository = _seed_cluster(tmp_path)
    cluster_id = _cluster_id(repository)
    create_place(
        repository,
        place_id="place_station",
        display_name="実名の駅",
        public_name="駅周辺",
        category="station",
        privacy_level="public_label",
        manual_verified=True,
    )
    link_place_cluster(repository, place_id="place_station", cluster_id=cluster_id)

    queue = place_review_queue_for_ui(repository, status="all", limit=10)
    detail = place_cluster_detail_for_ui(repository, cluster_id)

    assert queue["table"]
    assert "駅周辺" in str(queue["table"][0])
    assert "centroid_lat" not in detail["summary"]
    assert "139." not in detail["summary"]
    assert detail["linked_place_id"] == "place_station"
    assert detail["gallery"] == []


def _seed_cluster(tmp_path) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    for index, offset in enumerate((0.0, 0.0001, 0.0002), start=1):
        repository.add_media_item(
            id=f"media_{index}",
            file_path=f"/local/fake/{index}.jpg",
            file_name=f"{index}.jpg",
            file_hash=f"hash-{index}",
            captured_at=f"2025-01-0{index}T10:00:00+09:00",
            gps_lat=35.0 + offset,
            gps_lon=139.0 + offset,
        )
    build_location_points_from_media(repository, dry_run=False)
    cluster_location_points(repository, eps_meters=100, min_samples=3, dry_run=False)
    return repository


def _cluster_id(repository: LifelogRepository) -> str:
    rows = repository._fetch_all("SELECT id FROM place_clusters ORDER BY id", [])
    return str(rows[0]["id"])
