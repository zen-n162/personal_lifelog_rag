"""Local GPS clustering for place suggestions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any

from personal_lifelog_rag.places.geo import haversine_meters, parse_lat_lon, privacy_safe_lat_lon
from personal_lifelog_rag.places.matcher import match_place
from personal_lifelog_rag.places.schemas import Place, PlaceCluster


@dataclass
class _ClusterAccumulator:
    points: list[dict[str, Any]] = field(default_factory=list)
    center_lat: float = 0.0
    center_lon: float = 0.0

    def add(self, point: dict[str, Any]) -> None:
        self.points.append(point)
        count = len(self.points)
        self.center_lat = ((self.center_lat * (count - 1)) + float(point["lat"])) / count
        self.center_lon = ((self.center_lon * (count - 1)) + float(point["lon"])) / count


def cluster_place_candidates(
    media_items: list[dict[str, Any]],
    *,
    radius_m: float = 500.0,
    min_points: int = 5,
    places: list[Place] | None = None,
) -> list[PlaceCluster]:
    """Group nearby GPS-tagged photos into simple local place candidates."""

    points = [_media_point(item) for item in media_items]
    points = [point for point in points if point is not None]
    points.sort(key=lambda point: (point.get("at") or "", point["id"]))

    accumulators: list[_ClusterAccumulator] = []
    for point in points:
        nearest: tuple[float, _ClusterAccumulator] | None = None
        for accumulator in accumulators:
            distance = haversine_meters(
                float(point["lat"]),
                float(point["lon"]),
                accumulator.center_lat,
                accumulator.center_lon,
            )
            if distance <= radius_m and (nearest is None or distance < nearest[0]):
                nearest = (distance, accumulator)
        if nearest is None:
            accumulator = _ClusterAccumulator()
            accumulator.add(point)
            accumulators.append(accumulator)
        else:
            nearest[1].add(point)

    clusters: list[PlaceCluster] = []
    for accumulator in accumulators:
        if len(accumulator.points) < max(min_points, 1):
            continue
        distances = [
            haversine_meters(
                float(point["lat"]),
                float(point["lon"]),
                accumulator.center_lat,
                accumulator.center_lon,
            )
            for point in accumulator.points
        ]
        suggested_radius = max(radius_m, _round_up(max(distances, default=0.0) + 50.0, step=50))
        dates = sorted(
            {
                _record_date(point.get("at"))
                for point in accumulator.points
                if _record_date(point.get("at")) is not None
            }
        )
        nearest_place = match_place(accumulator.center_lat, accumulator.center_lon, places or [])
        clusters.append(
            PlaceCluster(
                cluster_id=f"candidate_place_{len(clusters) + 1:03d}",
                photo_count=len(accumulator.points),
                center_lat=round(accumulator.center_lat, 6),
                center_lon=round(accumulator.center_lon, 6),
                radius_m=float(round(suggested_radius, 1)),
                date_start=dates[0] if dates else None,
                date_end=dates[-1] if dates else None,
                nearest_registered_place=nearest_place,
            )
        )
    sorted_clusters = sorted(clusters, key=lambda cluster: (-cluster.photo_count, cluster.cluster_id))
    return [
        PlaceCluster(
            cluster_id=f"candidate_place_{index:03d}",
            photo_count=cluster.photo_count,
            center_lat=cluster.center_lat,
            center_lon=cluster.center_lon,
            radius_m=cluster.radius_m,
            date_start=cluster.date_start,
            date_end=cluster.date_end,
            nearest_registered_place=cluster.nearest_registered_place,
        )
        for index, cluster in enumerate(sorted_clusters, start=1)
    ]


def format_place_clusters(clusters: list[PlaceCluster], *, limit: int = 30) -> str:
    if not clusters:
        return "Place clusters\n- none"
    lines = ["Place clusters"]
    visible_clusters = clusters[: max(limit, 0)]
    for index, cluster in enumerate(visible_clusters, start=1):
        center_label = privacy_safe_lat_lon(
            cluster.center_lat,
            cluster.center_lon,
            show_exact_location=(
                cluster.nearest_registered_place.show_exact_location
                if cluster.nearest_registered_place
                else False
            ),
            privacy_level=(
                cluster.nearest_registered_place.privacy_level
                if cluster.nearest_registered_place
                else "normal"
            ),
        )
        nearest = (
            cluster.nearest_registered_place.place_id
            if cluster.nearest_registered_place
            else "none"
        )
        lines.extend(
            [
                "",
                f"{index}. {cluster.cluster_id}",
                f"   photos: {cluster.photo_count}",
                f"   date range: {cluster.date_start or 'unknown'}..{cluster.date_end or 'unknown'}",
                f"   center: {center_label}",
                f"   suggested radius: {int(round(cluster.radius_m))}m",
                f"   nearest registered place: {nearest}",
            ]
        )
    if len(clusters) > len(visible_clusters):
        lines.extend(
            [
                "",
                f"... {len(clusters) - len(visible_clusters)} more cluster(s) omitted from terminal output.",
                "Use --output private_config/place_suggestions.yaml to review all candidates locally.",
            ]
        )
    return "\n".join(lines)


def place_clusters_to_yaml(clusters: list[PlaceCluster]) -> str:
    lines = [
        "# Generated local GPS cluster suggestions.",
        "# Edit display_name manually before using.",
        "# Do not commit this file.",
        "# No reverse geocoding was used; candidate labels are intentionally neutral.",
        "",
        "places:",
    ]
    for index, cluster in enumerate(clusters, start=1):
        display_name = f"候補地点{index:03d}"
        lines.extend(
            [
                f"  - id: {cluster.cluster_id}",
                f"    name: \"{cluster.cluster_id}\"",
                f"    display_name: \"{display_name}\"",
                f"    lat: {cluster.center_lat:.6f}",
                f"    lon: {cluster.center_lon:.6f}",
                f"    radius_m: {int(round(cluster.radius_m))}",
                "    category: \"unknown\"",
                "    privacy_level: \"sensitive\"",
                "    show_exact_location: false",
                f"    notes: \"photos={cluster.photo_count}, date_range={cluster.date_start or 'unknown'}..{cluster.date_end or 'unknown'}\"",
            ]
        )
    return "\n".join(lines) + "\n"


def write_place_cluster_suggestions(path: str | Path, clusters: list[PlaceCluster]) -> Path:
    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(place_clusters_to_yaml(clusters), encoding="utf-8")
    return resolved


def _media_point(item: dict[str, Any]) -> dict[str, Any] | None:
    parsed = parse_lat_lon(item.get("gps_lat"), item.get("gps_lon"))
    if parsed is None:
        return None
    lat, lon = parsed
    return {
        "id": item.get("id") or item.get("file_path") or "",
        "lat": lat,
        "lon": lon,
        "at": item.get("captured_at") or item.get("fallback_captured_at") or "",
    }


def _record_date(value: object) -> str | None:
    if not value:
        return None
    text = str(value)
    if len(text) >= 10:
        return text[:10]
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        return None


def _round_up(value: float, *, step: int) -> int:
    return int(ceil(value / step) * step)
