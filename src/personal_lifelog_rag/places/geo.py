"""Small local geospatial helpers.

No external geocoding or online API is used here. Coordinates are only compared
locally against user-provided place definitions.
"""

from __future__ import annotations

import math
from typing import Any


EARTH_RADIUS_M = 6_371_000.0


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance between two GPS points in meters."""

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def valid_lat_lon(lat: Any, lon: Any) -> bool:
    parsed = parse_lat_lon(lat, lon)
    return parsed is not None


def parse_lat_lon(lat: Any, lon: Any) -> tuple[float, float] | None:
    try:
        parsed_lat = float(lat)
        parsed_lon = float(lon)
    except (TypeError, ValueError):
        return None
    if not (-90.0 <= parsed_lat <= 90.0):
        return None
    if not (-180.0 <= parsed_lon <= 180.0):
        return None
    return parsed_lat, parsed_lon


def rounded_coordinate(value: float | None, *, decimals: int = 3) -> str:
    if value is None:
        return "unknown"
    return f"{value:.{decimals}f}"


def rounded_lat_lon(lat: float | None, lon: float | None, *, decimals: int = 3) -> str:
    return f"{rounded_coordinate(lat, decimals=decimals)}, {rounded_coordinate(lon, decimals=decimals)}"


def privacy_safe_lat_lon(
    lat: float | None,
    lon: float | None,
    *,
    show_exact_location: bool,
    privacy_level: str = "normal",
) -> str:
    """Return a coordinate label that respects local privacy settings."""

    if privacy_level == "sensitive" and not show_exact_location:
        return "非表示"
    decimals = 6 if show_exact_location else 3
    return rounded_lat_lon(lat, lon, decimals=decimals)

