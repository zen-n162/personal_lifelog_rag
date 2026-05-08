from __future__ import annotations

from personal_lifelog_rag.places.geo import (
    haversine_meters,
    privacy_safe_lat_lon,
    rounded_lat_lon,
    valid_lat_lon,
)


def test_haversine_distance_is_reasonable_for_small_offset() -> None:
    distance = haversine_meters(0.0, 0.0, 0.0, 0.001)

    assert 100.0 < distance < 120.0


def test_lat_lon_validation() -> None:
    assert valid_lat_lon(10.0, 20.0)
    assert not valid_lat_lon(91.0, 20.0)
    assert not valid_lat_lon(10.0, 181.0)


def test_privacy_safe_lat_lon_hides_sensitive_exact_location() -> None:
    assert privacy_safe_lat_lon(10.123456, 20.123456, show_exact_location=False, privacy_level="sensitive") == "非表示"
    assert rounded_lat_lon(10.123456, 20.123456, decimals=3) == "10.123, 20.123"

