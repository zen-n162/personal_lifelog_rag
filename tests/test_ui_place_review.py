from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.places.location_store import build_location_points_from_media, cluster_location_points
from personal_lifelog_rag.ui.place_review_service import place_cluster_detail_for_ui, place_review_queue_for_ui


def test_ui_place_review_service_returns_rows_without_exact_lat_lon(tmp_path) -> None:
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

    payload = place_review_queue_for_ui(repository, status="all", limit=10)
    cluster_id = payload["cluster_ids"][0]
    detail = place_cluster_detail_for_ui(repository, cluster_id)

    assert payload["table"]
    assert "35." not in str(payload["table"])
    assert "139." not in str(payload["table"])
    assert "35." not in detail["summary"]
    assert "139." not in detail["summary"]
