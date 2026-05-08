"""Match GPS points to a user-provided local place dictionary."""

from __future__ import annotations

from personal_lifelog_rag.places.geo import haversine_meters, parse_lat_lon
from personal_lifelog_rag.places.schemas import Place, PlaceMatch


def match_place(lat: object, lon: object, places: list[Place]) -> PlaceMatch | None:
    parsed = parse_lat_lon(lat, lon)
    if parsed is None:
        return None
    point_lat, point_lon = parsed
    candidates: list[PlaceMatch] = []
    for place in places:
        distance_m = haversine_meters(point_lat, point_lon, place.lat, place.lon)
        if distance_m <= place.radius_m:
            candidates.append(
                PlaceMatch(
                    place_id=place.id,
                    display_name=place.display_name,
                    distance_m=distance_m,
                    privacy_level=place.privacy_level,
                    show_exact_location=place.show_exact_location,
                    category=place.category,
                )
            )
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: candidate.distance_m)


def summarize_place_matches(matches: list[PlaceMatch]) -> list[dict[str, object]]:
    counts: dict[str, dict[str, object]] = {}
    for match in matches:
        row = counts.setdefault(
            match.place_id,
            {
                "place_id": match.place_id,
                "display_name": match.display_name,
                "privacy_level": match.privacy_level,
                "show_exact_location": match.show_exact_location,
                "count": 0,
                "min_distance_m": round(match.distance_m, 1),
            },
        )
        row["count"] = int(row["count"]) + 1
        row["min_distance_m"] = min(float(row["min_distance_m"]), round(match.distance_m, 1))
    return sorted(counts.values(), key=lambda row: (-int(row["count"]), str(row["display_name"])))

