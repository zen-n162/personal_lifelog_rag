"""Privacy-safe display helpers for local place dictionaries."""

from __future__ import annotations

from personal_lifelog_rag.places.geo import privacy_safe_lat_lon
from personal_lifelog_rag.places.schemas import Place


def place_display_preview(places: list[Place]) -> list[dict[str, object]]:
    """Return how place entries will be shown in answers and UI surfaces."""

    rows: list[dict[str, object]] = []
    for place in places:
        coordinate_display = privacy_safe_lat_lon(
            place.lat,
            place.lon,
            show_exact_location=place.show_exact_location,
            privacy_level=place.privacy_level,
        )
        exact_hidden = not place.show_exact_location or place.privacy_level == "sensitive"
        rows.append(
            {
                "id": place.id,
                "display_name": place.display_name,
                "category": place.category,
                "privacy_level": place.privacy_level,
                "show_exact_location": place.show_exact_location,
                "coordinate_display": coordinate_display,
                "exact_coordinate_display": "hidden" if exact_hidden else "shown",
            }
        )
    return rows


def format_place_display_preview(rows: list[dict[str, object]]) -> str:
    lines = ["Place display preview"]
    if not rows:
        lines.append("- none")
        return "\n".join(lines)
    for row in rows:
        lines.extend(
            [
                "",
                f"- {row['id']}",
                f"  display: {row['display_name']}",
                f"  privacy: {row['privacy_level']}",
                f"  exact coordinate display: {row['exact_coordinate_display']}",
                f"  coordinate preview: {row['coordinate_display']}",
            ]
        )
    return "\n".join(lines)
