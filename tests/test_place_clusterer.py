from __future__ import annotations

from personal_lifelog_rag.places.clusterer import (
    cluster_place_candidates,
    format_place_clusters,
    place_clusters_to_yaml,
)
from personal_lifelog_rag.places.schemas import Place


def test_cluster_places_groups_nearby_gps_photos() -> None:
    media_items = [
        _media("a", "2024-12-01T10:00:00+09:00", 10.0000, 20.0000),
        _media("b", "2024-12-01T10:01:00+09:00", 10.0003, 20.0003),
        _media("c", "2024-12-02T10:02:00+09:00", 10.0004, 20.0004),
        _media("far", "2024-12-02T12:00:00+09:00", 11.0000, 20.0000),
    ]
    places = [
        Place(
            id="known_area",
            name="登録済み",
            display_name="登録済み",
            lat=10.0001,
            lon=20.0001,
            radius_m=200.0,
        )
    ]

    clusters = cluster_place_candidates(media_items, radius_m=200.0, min_points=3, places=places)

    assert len(clusters) == 1
    assert clusters[0].photo_count == 3
    assert clusters[0].date_start == "2024-12-01"
    assert clusters[0].date_end == "2024-12-02"
    assert clusters[0].nearest_registered_place is not None
    assert clusters[0].nearest_registered_place.place_id == "known_area"
    assert "candidate_place_001" in format_place_clusters(clusters)
    assert "show_exact_location: false" in place_clusters_to_yaml(clusters)


def test_cluster_format_hides_center_for_sensitive_registered_place() -> None:
    media_items = [
        _media("a", "2024-12-01T10:00:00+09:00", 10.0000, 20.0000),
        _media("b", "2024-12-01T10:01:00+09:00", 10.0001, 20.0001),
        _media("c", "2024-12-01T10:02:00+09:00", 10.0002, 20.0002),
    ]
    places = [
        Place(
            id="sensitive_area",
            name="センシティブ",
            display_name="センシティブ",
            lat=10.0001,
            lon=20.0001,
            radius_m=200.0,
            privacy_level="sensitive",
            show_exact_location=False,
        )
    ]

    output = format_place_clusters(
        cluster_place_candidates(media_items, radius_m=200.0, min_points=3, places=places)
    )

    assert "center: 非表示" in output
    assert "10.000" not in output


def _media(media_id: str, captured_at: str, lat: float, lon: float) -> dict[str, object]:
    return {
        "id": media_id,
        "file_name": f"{media_id}.jpg",
        "captured_at": captured_at,
        "gps_lat": lat,
        "gps_lon": lon,
    }
