"""Private GPS location-point and place-cluster storage.

This module never calls external geocoding services. Exact coordinates are kept
inside the local SQLite DB and formatted output avoids printing exact lat/lon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import math
from typing import Any

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.db.schema import initialize_schema
from personal_lifelog_rag.places.geo import haversine_meters, parse_lat_lon


LOCATION_PRIVACY_LEVELS = {
    "exact_private",
    "approximate_private",
    "public_hidden",
    "public_place_label",
}
CLUSTER_STATUSES = {"unreviewed", "accepted", "rejected", "merged"}
PLACE_PRIVACY_LEVELS = {"private", "public_label", "public_hidden"}
PLACE_CATEGORIES = {
    "home",
    "school",
    "lab",
    "station",
    "cafe",
    "restaurant",
    "travel",
    "shop",
    "event_venue",
    "other",
}


@dataclass
class LocationPointBuildReport:
    start_date: str | None
    end_date: str | None
    dry_run: bool
    media_scanned: int = 0
    gps_media: int = 0
    existing_points: int = 0
    would_create: int = 0
    created: int = 0
    updated: int = 0
    skipped_no_gps: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class ClusterBuildReport:
    start_date: str | None
    end_date: str | None
    dry_run: bool
    eps_meters: float
    min_samples: int
    points_scanned: int = 0
    clusters: list[dict[str, Any]] = field(default_factory=list)
    saved_clusters: int = 0
    assigned_points: int = 0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class PlaceAssignmentDbReport:
    start_date: str | None
    end_date: str | None
    dry_run: bool
    places_scanned: int = 0
    media_links: int = 0
    event_links: int = 0
    event_location_updates: int = 0
    skipped_no_cluster: int = 0
    assignments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def build_location_points_from_media(
    repository: LifelogRepository,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    dry_run: bool = True,
) -> LocationPointBuildReport:
    """Create one private location point per GPS-tagged media row."""

    media_items = repository.list_media_items(
        start_date=start_date,
        end_date=end_date,
        limit=1_000_000,
    )
    candidates: list[dict[str, Any]] = []
    skipped_no_gps = 0
    for item in media_items:
        parsed = parse_lat_lon(item.get("gps_lat"), item.get("gps_lon"))
        if parsed is None:
            skipped_no_gps += 1
            continue
        lat, lon = parsed
        candidates.append(
            {
                "id": f"lp_media_{item['id']}",
                "media_id": item["id"],
                "event_id": None,
                "captured_at": item.get("captured_at") or item.get("fallback_captured_at"),
                "source": "exif",
                "lat": lat,
                "lon": lon,
                "altitude": None,
                "accuracy_m": None,
                "geohash": _coarse_geohash(lat, lon),
                "privacy_level": "exact_private",
            }
        )

    existing = _existing_location_point_ids(repository, [row["id"] for row in candidates])
    report = LocationPointBuildReport(
        start_date=start_date,
        end_date=end_date,
        dry_run=dry_run,
        media_scanned=len(media_items),
        gps_media=len(candidates),
        existing_points=len(existing),
        would_create=sum(1 for row in candidates if row["id"] not in existing),
        skipped_no_gps=skipped_no_gps,
        samples=[_safe_location_point_sample(row) for row in candidates[:10]],
    )
    if dry_run:
        return report

    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        for row in candidates:
            existed = row["id"] in existing
            connection.execute(
                """
                INSERT INTO location_points (
                    id, media_id, event_id, captured_at, source, lat, lon,
                    altitude, accuracy_m, geohash, privacy_level, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    media_id = excluded.media_id,
                    event_id = excluded.event_id,
                    captured_at = excluded.captured_at,
                    source = excluded.source,
                    lat = excluded.lat,
                    lon = excluded.lon,
                    altitude = excluded.altitude,
                    accuracy_m = excluded.accuracy_m,
                    geohash = excluded.geohash,
                    privacy_level = excluded.privacy_level,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    row["id"],
                    row["media_id"],
                    row["event_id"],
                    row["captured_at"],
                    row["source"],
                    row["lat"],
                    row["lon"],
                    row["altitude"],
                    row["accuracy_m"],
                    row["geohash"],
                    row["privacy_level"],
                ),
            )
            if existed:
                report.updated += 1
            else:
                report.created += 1
        connection.commit()
    return report


def cluster_location_points(
    repository: LifelogRepository,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    eps_meters: float = 100.0,
    min_samples: int = 3,
    dry_run: bool = True,
) -> ClusterBuildReport:
    """Cluster private GPS points with a small local distance-based algorithm."""

    points = _fetch_location_points(repository, start_date=start_date, end_date=end_date)
    raw_clusters = _simple_distance_clusters(points, eps_meters=eps_meters)
    raw_clusters = [cluster for cluster in raw_clusters if len(cluster) >= max(1, min_samples)]
    raw_clusters.sort(key=lambda rows: (-len(rows), rows[0].get("captured_at") or "", rows[0]["id"]))
    method = _cluster_method(start_date, end_date, eps_meters, min_samples)
    clusters = [
        _cluster_record(
            rows,
            cluster_id=f"place_cluster_{_range_token(start_date, end_date)}_{index:03d}",
            method=method,
            eps_meters=eps_meters,
        )
        for index, rows in enumerate(raw_clusters, start=1)
    ]
    report = ClusterBuildReport(
        start_date=start_date,
        end_date=end_date,
        dry_run=dry_run,
        eps_meters=eps_meters,
        min_samples=min_samples,
        points_scanned=len(points),
        clusters=[_safe_cluster_sample(row) for row in clusters],
    )
    if dry_run:
        return report

    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        for cluster, rows in zip(clusters, raw_clusters):
            connection.execute(
                """
                INSERT INTO place_clusters (
                    id, centroid_lat, centroid_lon, radius_m, point_count,
                    photo_count, event_count, first_seen_at, last_seen_at,
                    clustering_method, status, privacy_level, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'unreviewed', 'exact_private', CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    centroid_lat = excluded.centroid_lat,
                    centroid_lon = excluded.centroid_lon,
                    radius_m = excluded.radius_m,
                    point_count = excluded.point_count,
                    photo_count = excluded.photo_count,
                    event_count = excluded.event_count,
                    first_seen_at = excluded.first_seen_at,
                    last_seen_at = excluded.last_seen_at,
                    clustering_method = excluded.clustering_method,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    cluster["id"],
                    cluster["centroid_lat"],
                    cluster["centroid_lon"],
                    cluster["radius_m"],
                    cluster["point_count"],
                    cluster["photo_count"],
                    cluster["event_count"],
                    cluster["first_seen_at"],
                    cluster["last_seen_at"],
                    cluster["clustering_method"],
                ),
            )
            for point in rows:
                connection.execute(
                    "UPDATE location_points SET cluster_id = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (cluster["id"], point["id"]),
                )
                report.assigned_points += 1
            report.saved_clusters += 1
        connection.commit()
    return report


def create_place(
    repository: LifelogRepository,
    *,
    display_name: str,
    public_name: str | None = None,
    category: str | None = None,
    privacy_level: str = "private",
    cluster_id: str | None = None,
    aliases: list[str] | None = None,
    manual_verified: bool = False,
    notes: str | None = None,
    place_id: str | None = None,
) -> dict[str, Any]:
    category = category or "other"
    if category not in PLACE_CATEGORIES:
        raise ValueError(f"category must be one of: {', '.join(sorted(PLACE_CATEGORIES))}")
    if privacy_level not in PLACE_PRIVACY_LEVELS:
        raise ValueError(f"privacy_level must be one of: {', '.join(sorted(PLACE_PRIVACY_LEVELS))}")
    resolved_id = place_id or f"place_{_hash_text(display_name)[:12]}"
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        if cluster_id and not _exists(connection, "place_clusters", "id", cluster_id):
            raise ValueError(f"cluster_id not found: {cluster_id}")
        connection.execute(
            """
            INSERT INTO places (
                id, display_name, public_name, category, cluster_id, aliases_json,
                privacy_level, manual_verified, notes, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                display_name = excluded.display_name,
                public_name = excluded.public_name,
                category = excluded.category,
                cluster_id = excluded.cluster_id,
                aliases_json = excluded.aliases_json,
                privacy_level = excluded.privacy_level,
                manual_verified = excluded.manual_verified,
                notes = excluded.notes,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                resolved_id,
                display_name,
                public_name,
                category,
                cluster_id,
                json.dumps(aliases or [], ensure_ascii=False),
                privacy_level,
                int(bool(manual_verified)),
                notes,
            ),
        )
        connection.commit()
    return get_place(repository, resolved_id) or {}


def get_place(repository: LifelogRepository, place_id: str) -> dict[str, Any] | None:
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        row = connection.execute("SELECT * FROM places WHERE id = ?", (place_id,)).fetchone()
        return dict(row) if row else None


def list_place_clusters(
    repository: LifelogRepository,
    *,
    limit: int = 20,
    status: str | None = None,
    privacy_level: str | None = None,
    category: str | None = None,
    search_text: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    min_points: int | None = None,
    include_exact: bool = False,
) -> list[dict[str, Any]]:
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        clauses: list[str] = []
        params: list[Any] = []
        if status and status != "all":
            clauses.append("place_clusters.status = ?")
            params.append(status)
        if privacy_level and privacy_level != "all" and privacy_level in PLACE_PRIVACY_LEVELS:
            clauses.append("places.privacy_level = ?")
            params.append(privacy_level)
        elif privacy_level and privacy_level != "all":
            clauses.append("place_clusters.privacy_level = ?")
            params.append(privacy_level)
        if category and category != "all":
            clauses.append("places.category = ?")
            params.append(category)
        if start_date:
            clauses.append("substr(place_clusters.last_seen_at, 1, 10) >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("substr(place_clusters.first_seen_at, 1, 10) <= ?")
            params.append(end_date)
        if min_points is not None:
            clauses.append("COALESCE(place_clusters.point_count, 0) >= ?")
            params.append(int(min_points))
        if search_text:
            like_value = f"%{search_text}%"
            clauses.append(
                """
                (
                    COALESCE(places.display_name, '') LIKE ?
                    OR COALESCE(places.public_name, '') LIKE ?
                    OR COALESCE(places.category, '') LIKE ?
                    OR COALESCE(places.aliases_json, '') LIKE ?
                    OR place_clusters.id LIKE ?
                )
                """
            )
            params.extend([like_value, like_value, like_value, like_value, like_value])
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = connection.execute(
            f"""
            SELECT place_clusters.*,
                   places.id AS linked_place_id,
                   places.display_name AS linked_display_name,
                   places.public_name AS linked_public_name,
                   places.category AS linked_category,
                   places.aliases_json AS linked_aliases_json,
                   places.privacy_level AS linked_privacy_level,
                   places.manual_verified AS linked_manual_verified
            FROM place_clusters
            LEFT JOIN places ON places.cluster_id = place_clusters.id
            {where_sql}
            ORDER BY point_count DESC, first_seen_at ASC, place_clusters.id ASC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [_safe_cluster_sample(dict(row)) if not include_exact else dict(row) for row in rows]


def get_place_cluster_detail(
    repository: LifelogRepository,
    cluster_id: str,
    *,
    include_exact: bool = False,
    limit_photos: int = 12,
    limit_events: int = 20,
) -> dict[str, Any] | None:
    """Return review-safe detail for one cluster.

    Exact centroid/point coordinates are omitted unless include_exact=True.
    """

    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        row = connection.execute(
            """
            SELECT place_clusters.*,
                   places.id AS linked_place_id,
                   places.display_name AS linked_display_name,
                   places.public_name AS linked_public_name,
                   places.category AS linked_category,
                   places.aliases_json AS linked_aliases_json,
                   places.privacy_level AS linked_privacy_level,
                   places.manual_verified AS linked_manual_verified,
                   places.notes AS linked_notes
            FROM place_clusters
            LEFT JOIN places ON places.cluster_id = place_clusters.id
            WHERE place_clusters.id = ?
            """,
            (cluster_id,),
        ).fetchone()
        if row is None:
            return None
        cluster = dict(row)
        point_rows = connection.execute(
            """
            SELECT location_points.id, location_points.media_id, location_points.event_id,
                   location_points.captured_at, location_points.source,
                   location_points.privacy_level, location_points.geohash,
                   location_points.lat, location_points.lon,
                   media_items.file_name, media_items.thumbnail_path, media_items.file_path,
                   media_items.captured_at AS media_captured_at,
                   media_items.fallback_captured_at
            FROM location_points
            LEFT JOIN media_items ON media_items.id = location_points.media_id
            WHERE location_points.cluster_id = ?
            ORDER BY COALESCE(location_points.captured_at, media_items.captured_at, media_items.fallback_captured_at) ASC,
                     location_points.id ASC
            """,
            (cluster_id,),
        ).fetchall()
        event_rows = connection.execute(
            """
            SELECT DISTINCT events.id, events.date, events.start_time, events.title,
                   events.summary, events.location_name, events.confidence
            FROM events
            LEFT JOIN event_places ON event_places.event_id = events.id
            LEFT JOIN places ON places.id = event_places.place_id
            WHERE places.cluster_id = ?
               OR events.id IN (
                   SELECT event_evidence.event_id
                   FROM event_evidence
                   JOIN location_points ON location_points.media_id = event_evidence.evidence_id
                   WHERE event_evidence.evidence_type = 'photo'
                     AND location_points.cluster_id = ?
               )
            ORDER BY events.date ASC, events.start_time ASC, events.id ASC
            LIMIT ?
            """,
            (cluster_id, cluster_id, limit_events),
        ).fetchall()
    safe_cluster = dict(cluster) if include_exact else _safe_cluster_sample(cluster)
    safe_cluster["approximate_location_label"] = _approximate_cluster_label(cluster)
    return {
        "cluster": safe_cluster,
        "points": [
            _safe_location_point_detail(dict(point), include_exact=include_exact)
            for point in point_rows[: max(limit_photos, 0)]
        ],
        "representative_thumbnails": _representative_thumbnail_paths([dict(point) for point in point_rows], limit=limit_photos),
        "related_events": [_safe_event_detail(dict(event)) for event in event_rows],
        "related_dates": sorted({str(point["captured_at"])[:10] for point in point_rows if point["captured_at"]}),
    }


def link_place_cluster(repository: LifelogRepository, *, place_id: str, cluster_id: str) -> dict[str, Any]:
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        if not _exists(connection, "places", "id", place_id):
            raise ValueError(f"place_id not found: {place_id}")
        if not _exists(connection, "place_clusters", "id", cluster_id):
            raise ValueError(f"cluster_id not found: {cluster_id}")
        connection.execute(
            "UPDATE places SET cluster_id = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (cluster_id, place_id),
        )
        connection.commit()
    return get_place(repository, place_id) or {}


def update_place(
    repository: LifelogRepository,
    *,
    place_id: str,
    display_name: str | None = None,
    public_name: str | None = None,
    category: str | None = None,
    privacy_level: str | None = None,
    manual_verified: bool | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    place = get_place(repository, place_id)
    if not place:
        raise ValueError(f"place_id not found: {place_id}")
    resolved_category = category if category is not None else place.get("category")
    if resolved_category and resolved_category not in PLACE_CATEGORIES:
        raise ValueError(f"category must be one of: {', '.join(sorted(PLACE_CATEGORIES))}")
    resolved_privacy = privacy_level if privacy_level is not None else place.get("privacy_level")
    if resolved_privacy and resolved_privacy not in PLACE_PRIVACY_LEVELS:
        raise ValueError(f"privacy_level must be one of: {', '.join(sorted(PLACE_PRIVACY_LEVELS))}")
    updates = {
        "display_name": display_name if display_name is not None else place.get("display_name"),
        "public_name": public_name if public_name is not None else place.get("public_name"),
        "category": resolved_category or "other",
        "privacy_level": resolved_privacy or "private",
        "manual_verified": int(bool(manual_verified)) if manual_verified is not None else int(place.get("manual_verified") or 0),
        "notes": notes if notes is not None else place.get("notes"),
    }
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        connection.execute(
            """
            UPDATE places
            SET display_name = ?, public_name = ?, category = ?, privacy_level = ?,
                manual_verified = ?, notes = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                updates["display_name"],
                updates["public_name"],
                updates["category"],
                updates["privacy_level"],
                updates["manual_verified"],
                updates["notes"],
                place_id,
            ),
        )
        connection.commit()
    return get_place(repository, place_id) or {}


def update_place_cluster_status(repository: LifelogRepository, *, cluster_id: str, status: str) -> dict[str, Any]:
    if status not in CLUSTER_STATUSES:
        raise ValueError(f"status must be one of: {', '.join(sorted(CLUSTER_STATUSES))}")
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        if not _exists(connection, "place_clusters", "id", cluster_id):
            raise ValueError(f"cluster_id not found: {cluster_id}")
        connection.execute(
            "UPDATE place_clusters SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, cluster_id),
        )
        connection.commit()
    detail = get_place_cluster_detail(repository, cluster_id)
    return (detail or {}).get("cluster") or {"id": cluster_id, "status": status}


def unlink_place_cluster(repository: LifelogRepository, *, cluster_id: str) -> int:
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        cursor = connection.execute(
            "UPDATE places SET cluster_id = NULL, updated_at = CURRENT_TIMESTAMP WHERE cluster_id = ?",
            (cluster_id,),
        )
        connection.commit()
        return int(cursor.rowcount or 0)


def add_place_alias(repository: LifelogRepository, *, place_id: str, alias: str) -> dict[str, Any]:
    place = get_place(repository, place_id)
    if not place:
        raise ValueError(f"place_id not found: {place_id}")
    aliases = _json_list(place.get("aliases_json"))
    if alias not in aliases:
        aliases.append(alias)
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        connection.execute(
            "UPDATE places SET aliases_json = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (json.dumps(aliases, ensure_ascii=False), place_id),
        )
        connection.commit()
    return get_place(repository, place_id) or {}


def set_place_privacy(repository: LifelogRepository, *, place_id: str, privacy_level: str) -> dict[str, Any]:
    if privacy_level not in PLACE_PRIVACY_LEVELS:
        raise ValueError(f"privacy_level must be one of: {', '.join(sorted(PLACE_PRIVACY_LEVELS))}")
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        if not _exists(connection, "places", "id", place_id):
            raise ValueError(f"place_id not found: {place_id}")
        connection.execute(
            "UPDATE places SET privacy_level = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (privacy_level, place_id),
        )
        connection.commit()
    return get_place(repository, place_id) or {}


def assign_db_places(
    repository: LifelogRepository,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    dry_run: bool = True,
) -> PlaceAssignmentDbReport:
    places = _fetch_db_places(repository)
    report = PlaceAssignmentDbReport(start_date=start_date, end_date=end_date, dry_run=dry_run, places_scanned=len(places))
    points = _fetch_location_points(repository, start_date=start_date, end_date=end_date)
    points_by_cluster: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        cluster_id = point.get("cluster_id")
        if cluster_id:
            points_by_cluster.setdefault(str(cluster_id), []).append(point)

    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        for place in places:
            cluster_id = place.get("cluster_id")
            if not cluster_id:
                report.skipped_no_cluster += 1
                continue
            cluster_points = points_by_cluster.get(str(cluster_id), [])
            if not cluster_points:
                continue
            label = _private_place_label(place)
            media_ids = sorted({str(point["media_id"]) for point in cluster_points if point.get("media_id")})
            for media_id in media_ids:
                report.media_links += 1
                if not dry_run:
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO media_places (
                            media_id, place_id, source, confidence, updated_at
                        )
                        VALUES (?, ?, 'gps_cluster', 0.80, CURRENT_TIMESTAMP)
                        """,
                        (media_id, place["id"]),
                    )
            event_ids = sorted({str(point["event_id"]) for point in cluster_points if point.get("event_id")})
            event_ids.extend(_event_ids_for_media(connection, media_ids))
            for event_id in sorted(set(event_ids)):
                report.event_links += 1
                if not dry_run:
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO event_places (
                            event_id, place_id, source, confidence, updated_at
                        )
                        VALUES (?, ?, 'gps_cluster', 0.75, CURRENT_TIMESTAMP)
                        """,
                        (event_id, place["id"]),
                    )
                    cursor = connection.execute(
                        """
                        UPDATE events
                        SET location_name = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                          AND COALESCE(is_user_edited, 0) = 0
                          AND NOT EXISTS (
                              SELECT 1 FROM event_overrides
                              WHERE event_overrides.event_id = events.id
                                AND event_overrides.location_name_override IS NOT NULL
                                AND TRIM(event_overrides.location_name_override) != ''
                          )
                        """,
                        (label, event_id),
                    )
                    report.event_location_updates += max(cursor.rowcount, 0)
                report.assignments.append(
                    {
                        "event_id": event_id,
                        "place_id": place["id"],
                        "place_label": label,
                        "source": "gps_cluster",
                    }
                )
        if not dry_run:
            connection.commit()
    return report


def location_place_stats(repository: LifelogRepository) -> dict[str, Any]:
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        return {
            "location_points": _count(connection, "SELECT COUNT(*) FROM location_points"),
            "place_clusters": _count(connection, "SELECT COUNT(*) FROM place_clusters"),
            "places": _count(connection, "SELECT COUNT(*) FROM places"),
            "event_places": _count(connection, "SELECT COUNT(*) FROM event_places"),
            "media_places": _count(connection, "SELECT COUNT(*) FROM media_places"),
            "location_points_by_privacy": _count_rows(connection, "location_points", "privacy_level"),
            "places_by_privacy": _count_rows(connection, "places", "privacy_level"),
        }


def public_place_label(place: dict[str, Any]) -> str:
    """Return a public-safe label for reports/UI public mode."""

    privacy_level = str(place.get("privacy_level") or "private")
    category = str(place.get("category") or "other")
    if privacy_level == "public_label":
        return str(place.get("public_name") or place.get("display_name") or category)
    if privacy_level == "public_hidden" or category in {"home", "lab"}:
        return "非公開の場所"
    return str(place.get("public_name") or category or "場所候補")


def format_location_point_build_report(report: LocationPointBuildReport) -> str:
    lines = [
        "Location Points",
        f"- range: {report.start_date or 'all'}..{report.end_date or 'all'}",
        f"- dry_run: {report.dry_run}",
        f"- media scanned: {report.media_scanned}",
        f"- GPS media: {report.gps_media}",
        f"- existing points: {report.existing_points}",
        f"- would create: {report.would_create}",
        f"- created: {report.created}",
        f"- updated: {report.updated}",
        f"- skipped no GPS: {report.skipped_no_gps}",
        "- exact coordinates are stored only in the private SQLite DB.",
    ]
    if report.samples:
        lines.append("Samples:")
        for sample in report.samples[:5]:
            lines.append(f"- {sample['id']} media={sample.get('media_id')} at={sample.get('captured_at')}")
    return "\n".join(lines)


def format_cluster_build_report(report: ClusterBuildReport) -> str:
    lines = [
        "Place Clusters",
        f"- range: {report.start_date or 'all'}..{report.end_date or 'all'}",
        f"- dry_run: {report.dry_run}",
        f"- eps_meters: {report.eps_meters:g}",
        f"- min_samples: {report.min_samples}",
        f"- points scanned: {report.points_scanned}",
        f"- clusters: {len(report.clusters)}",
        f"- saved clusters: {report.saved_clusters}",
        f"- assigned points: {report.assigned_points}",
    ]
    if report.clusters:
        lines.append("Cluster candidates:")
        for cluster in report.clusters[:20]:
            lines.append(
                f"- {cluster['id']}: points={cluster['point_count']} photos={cluster['photo_count']} "
                f"radius={cluster['radius_m']:.1f}m dates={cluster.get('first_seen_at') or 'unknown'}..{cluster.get('last_seen_at') or 'unknown'}"
            )
        if len(report.clusters) > 20:
            lines.append(f"- ... {len(report.clusters) - 20} more")
    lines.append("- centroid coordinates are hidden in CLI output.")
    return "\n".join(lines)


def format_db_place_assignment_report(report: PlaceAssignmentDbReport) -> str:
    lines = [
        "DB Place Assignment",
        f"- range: {report.start_date or 'all'}..{report.end_date or 'all'}",
        f"- dry_run: {report.dry_run}",
        f"- places scanned: {report.places_scanned}",
        f"- media links: {report.media_links}",
        f"- event links: {report.event_links}",
        f"- event location updates: {report.event_location_updates}",
        f"- skipped no cluster: {report.skipped_no_cluster}",
    ]
    if report.assignments:
        lines.append("Assignments:")
        for row in report.assignments[:20]:
            lines.append(f"- {row['event_id']} -> {row['place_label']} ({row['place_id']})")
    return "\n".join(lines)


def format_cluster_list(rows: list[dict[str, Any]]) -> str:
    lines = ["Place cluster list", f"- total shown: {len(rows)}"]
    for row in rows:
        linked = row.get("linked_place_id") or "none"
        lines.append(
            f"- {row['id']}: points={row.get('point_count') or 0}, "
            f"photos={row.get('photo_count') or 0}, radius={row.get('radius_m') or 0:.1f}m, "
            f"status={row.get('status')}, linked_place={linked}"
        )
    return "\n".join(lines)


def format_place_row(place: dict[str, Any]) -> str:
    aliases = ", ".join(_json_list(place.get("aliases_json"))) or "none"
    return "\n".join(
        [
            "Place",
            f"- id: {place.get('id')}",
            f"- display_name: {place.get('display_name')}",
            f"- public_name: {place.get('public_name') or ''}",
            f"- category: {place.get('category') or ''}",
            f"- cluster_id: {place.get('cluster_id') or ''}",
            f"- privacy_level: {place.get('privacy_level')}",
            f"- manual_verified: {int(place.get('manual_verified') or 0)}",
            f"- aliases: {aliases}",
        ]
    )


def format_cluster_detail(detail: dict[str, Any] | None) -> str:
    if not detail:
        return "Place cluster was not found."
    cluster = detail["cluster"]
    lines = [
        "Place cluster detail",
        f"- cluster_id: {cluster.get('id')}",
        f"- points: {cluster.get('point_count') or 0}",
        f"- photos: {cluster.get('photo_count') or 0}",
        f"- events: {cluster.get('event_count') or 0}",
        f"- first_seen_at: {cluster.get('first_seen_at') or ''}",
        f"- last_seen_at: {cluster.get('last_seen_at') or ''}",
        f"- radius_m: {float(cluster.get('radius_m') or 0.0):.1f}",
        f"- status: {cluster.get('status')}",
        f"- privacy_level: {cluster.get('privacy_level')}",
        f"- approximate_location: {cluster.get('approximate_location_label') or ''}",
        f"- linked_place: {cluster.get('linked_place_id') or 'none'}",
        "- exact coordinates: hidden by default",
    ]
    if detail.get("related_dates"):
        lines.append("- related_dates: " + ", ".join(detail["related_dates"][:20]))
    if detail.get("related_events"):
        lines.append("Related events:")
        for event in detail["related_events"][:10]:
            lines.append(f"- {event.get('date') or ''} {event.get('start_time') or ''} {event.get('title') or event.get('id')}")
    if detail.get("points"):
        lines.append("Representative media:")
        for point in detail["points"][:10]:
            lines.append(f"- {point.get('captured_at') or ''} {point.get('media_id') or ''} {point.get('file_name') or ''}")
    return "\n".join(lines)


def _existing_location_point_ids(repository: LifelogRepository, ids: list[str]) -> set[str]:
    if not ids:
        return set()
    placeholders = ", ".join("?" for _ in ids)
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        rows = connection.execute(
            f"SELECT id FROM location_points WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
    return {str(row["id"]) for row in rows}


def _fetch_location_points(
    repository: LifelogRepository,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    clauses = ["lat IS NOT NULL", "lon IS NOT NULL"]
    params: list[Any] = []
    if start_date is not None:
        clauses.append("substr(captured_at, 1, 10) >= ?")
        params.append(start_date)
    if end_date is not None:
        clauses.append("substr(captured_at, 1, 10) <= ?")
        params.append(end_date)
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        rows = connection.execute(
            f"""
            SELECT *
            FROM location_points
            WHERE {' AND '.join(clauses)}
            ORDER BY captured_at ASC, id ASC
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows if parse_lat_lon(row["lat"], row["lon"]) is not None]


def _simple_distance_clusters(points: list[dict[str, Any]], *, eps_meters: float) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    centers: list[tuple[float, float]] = []
    for point in points:
        point_lat = float(point["lat"])
        point_lon = float(point["lon"])
        nearest_index: int | None = None
        nearest_distance: float | None = None
        for index, (center_lat, center_lon) in enumerate(centers):
            distance = haversine_meters(point_lat, point_lon, center_lat, center_lon)
            if distance <= eps_meters and (nearest_distance is None or distance < nearest_distance):
                nearest_index = index
                nearest_distance = distance
        if nearest_index is None:
            clusters.append([point])
            centers.append((point_lat, point_lon))
        else:
            clusters[nearest_index].append(point)
            centers[nearest_index] = _centroid(clusters[nearest_index])
    return clusters


def _cluster_record(
    rows: list[dict[str, Any]],
    *,
    cluster_id: str,
    method: str,
    eps_meters: float,
) -> dict[str, Any]:
    centroid_lat, centroid_lon = _centroid(rows)
    distances = [haversine_meters(float(row["lat"]), float(row["lon"]), centroid_lat, centroid_lon) for row in rows]
    seen = sorted(str(row.get("captured_at") or "") for row in rows if row.get("captured_at"))
    return {
        "id": cluster_id,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "radius_m": max(float(eps_meters), max(distances, default=0.0) + 10.0),
        "point_count": len(rows),
        "photo_count": len({row.get("media_id") for row in rows if row.get("media_id")}),
        "event_count": len({row.get("event_id") for row in rows if row.get("event_id")}),
        "first_seen_at": seen[0] if seen else None,
        "last_seen_at": seen[-1] if seen else None,
        "clustering_method": method,
    }


def _centroid(rows: list[dict[str, Any]]) -> tuple[float, float]:
    return (
        sum(float(row["lat"]) for row in rows) / len(rows),
        sum(float(row["lon"]) for row in rows) / len(rows),
    )


def _safe_location_point_sample(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "media_id": row.get("media_id"),
        "event_id": row.get("event_id"),
        "captured_at": row.get("captured_at"),
        "source": row.get("source"),
        "privacy_level": row.get("privacy_level"),
        "geohash": row.get("geohash"),
    }


def _safe_cluster_sample(row: dict[str, Any]) -> dict[str, Any]:
    safe = dict(row)
    safe.pop("centroid_lat", None)
    safe.pop("centroid_lon", None)
    safe["approximate_location_label"] = _approximate_cluster_label(row)
    return safe


def _safe_location_point_detail(row: dict[str, Any], *, include_exact: bool) -> dict[str, Any]:
    safe = {
        "id": row.get("id"),
        "media_id": row.get("media_id"),
        "event_id": row.get("event_id"),
        "captured_at": row.get("captured_at") or row.get("media_captured_at") or row.get("fallback_captured_at"),
        "source": row.get("source"),
        "privacy_level": row.get("privacy_level"),
        "geohash": row.get("geohash"),
        "file_name": row.get("file_name"),
        "thumbnail_path": row.get("thumbnail_path") or "",
    }
    if include_exact:
        safe["lat"] = row.get("lat")
        safe["lon"] = row.get("lon")
    return safe


def _safe_event_detail(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "date": row.get("date"),
        "start_time": row.get("start_time"),
        "title": row.get("title"),
        "summary": row.get("summary"),
        "location_name": row.get("location_name"),
        "confidence": row.get("confidence"),
    }


def _representative_thumbnail_paths(rows: list[dict[str, Any]], *, limit: int) -> list[str]:
    paths: list[str] = []
    for row in rows:
        value = str(row.get("thumbnail_path") or "").strip()
        if value and value not in paths:
            paths.append(value)
        if len(paths) >= max(limit, 0):
            break
    return paths


def _approximate_cluster_label(row: dict[str, Any]) -> str:
    radius = float(row.get("radius_m") or 0.0)
    points = int(row.get("point_count") or 0)
    if radius:
        return f"半径約{radius:.0f}mの場所候補 ({points} points)"
    return f"場所候補 ({points} points)"


def _fetch_db_places(repository: LifelogRepository) -> list[dict[str, Any]]:
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        rows = connection.execute("SELECT * FROM places ORDER BY id ASC").fetchall()
    return [dict(row) for row in rows]


def _event_ids_for_media(connection, media_ids: list[str]) -> list[str]:
    if not media_ids:
        return []
    placeholders = ", ".join("?" for _ in media_ids)
    rows = connection.execute(
        f"""
        SELECT DISTINCT event_id
        FROM event_evidence
        WHERE evidence_type = 'photo'
          AND evidence_id IN ({placeholders})
        ORDER BY event_id ASC
        """,
        media_ids,
    ).fetchall()
    return [str(row["event_id"]) for row in rows]


def _private_place_label(place: dict[str, Any]) -> str:
    return str(place.get("display_name") or place.get("public_name") or place.get("category") or place.get("id"))


def _coarse_geohash(lat: float, lon: float) -> str:
    return f"{lat:.4f},{lon:.4f}"


def _cluster_method(start_date: str | None, end_date: str | None, eps_meters: float, min_samples: int) -> str:
    return f"local_distance_v1:{_range_token(start_date, end_date)}:eps={eps_meters:g}:min={min_samples}"


def _range_token(start_date: str | None, end_date: str | None) -> str:
    start = (start_date or "all").replace("-", "")
    end = (end_date or "all").replace("-", "")
    return f"{start}_{end}"


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _json_list(raw: object) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw]
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _exists(connection, table_name: str, column_name: str, value: str) -> bool:
    row = connection.execute(
        f"SELECT 1 FROM {table_name} WHERE {column_name} = ? LIMIT 1",
        (value,),
    ).fetchone()
    return row is not None


def _count(connection, sql: str, params: tuple[Any, ...] = ()) -> int:
    row = connection.execute(sql, params).fetchone()
    return int(row[0] or 0)


def _count_rows(connection, table_name: str, column_name: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        f"""
        SELECT COALESCE({column_name}, '(null)') AS value, COUNT(*) AS count
        FROM {table_name}
        GROUP BY {column_name}
        ORDER BY count DESC, value ASC
        """
    ).fetchall()
    return [{"value": row["value"], "count": int(row["count"])} for row in rows]
