"""UI-facing helpers for local place cluster review.

Exact coordinates stay hidden by default. The UI may request private exact
values explicitly, but public-mode payloads only expose labels/categories.
"""

from __future__ import annotations

from typing import Any

from personal_lifelog_rag.core.privacy import redact_text
from personal_lifelog_rag.places.location_store import (
    add_place_alias,
    assign_db_places,
    create_place,
    get_place_cluster_detail,
    link_place_cluster,
    list_place_clusters,
    public_place_label,
    unlink_place_cluster,
    update_place,
    update_place_cluster_status,
)


def place_review_queue_for_ui(
    repository,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    min_points: int | None = None,
    status: str | None = None,
    privacy_level: str | None = None,
    category: str | None = None,
    search_text: str | None = None,
    limit: int = 50,
    mode: str = "public",
) -> dict[str, Any]:
    rows = list_place_clusters(
        repository,
        start_date=date_from,
        end_date=date_to,
        min_points=min_points,
        status=status,
        privacy_level=privacy_level,
        category=category,
        search_text=search_text,
        limit=limit,
        include_exact=False,
    )
    table = [_cluster_row(row, mode=mode) for row in rows]
    return {"rows": rows, "table": table, "cluster_ids": [row[0] for row in table]}


def place_cluster_detail_for_ui(
    repository,
    cluster_id: str | None,
    *,
    private_mode: bool = False,
) -> dict[str, Any]:
    if not cluster_id:
        return _empty_detail()
    detail = get_place_cluster_detail(repository, cluster_id, include_exact=private_mode)
    if not detail:
        return _empty_detail()
    cluster = detail["cluster"]
    linked = _linked_place_label(cluster, mode="private" if private_mode else "public")
    events = [
        [
            event.get("date") or "",
            event.get("start_time") or "",
            redact_text(event.get("title"), max_chars=80),
            redact_text(event.get("location_name"), max_chars=60),
            event.get("confidence") if event.get("confidence") is not None else "",
        ]
        for event in detail.get("related_events") or []
    ]
    photos = [
        [
            point.get("media_id") or "",
            point.get("captured_at") or "",
            redact_text(point.get("file_name"), max_chars=80),
            point.get("thumbnail_path") or "",
        ]
        for point in detail.get("points") or []
    ]
    note_lines = [
        f"cluster_id: {cluster.get('id')}",
        f"points: {cluster.get('point_count') or 0}",
        f"photos: {cluster.get('photo_count') or 0}",
        f"events: {cluster.get('event_count') or 0}",
        f"first_seen_at: {cluster.get('first_seen_at') or ''}",
        f"last_seen_at: {cluster.get('last_seen_at') or ''}",
        f"radius_m: {float(cluster.get('radius_m') or 0.0):.1f}",
        f"status: {cluster.get('status') or ''}",
        f"privacy_level: {cluster.get('privacy_level') or ''}",
        f"linked_place: {linked}",
        f"approximate_location: {cluster.get('approximate_location_label') or ''}",
    ]
    if private_mode and "centroid_lat" in cluster and "centroid_lon" in cluster:
        note_lines.append("exact centroid: private mode enabled")
    else:
        note_lines.append("exact centroid: hidden")
    return {
        "cluster_id": str(cluster.get("id") or ""),
        "summary": "\n".join(note_lines),
        "events": events,
        "photos": photos,
        "gallery": [path for path in (detail.get("representative_thumbnails") or []) if path],
        "linked_place_id": str(cluster.get("linked_place_id") or ""),
        "linked_display_name": str(cluster.get("linked_display_name") or ""),
        "linked_public_name": str(cluster.get("linked_public_name") or ""),
        "linked_category": str(cluster.get("linked_category") or "other"),
        "linked_privacy_level": str(cluster.get("linked_privacy_level") or "private"),
        "linked_aliases": ", ".join(_json_list(cluster.get("linked_aliases_json"))),
    }


def create_place_for_ui(
    repository,
    *,
    display_name: str,
    public_name: str | None = None,
    category: str = "other",
    privacy_level: str = "private",
    cluster_id: str | None = None,
    aliases_text: str | None = None,
    manual_verified: bool = True,
    notes: str | None = None,
) -> str:
    place = create_place(
        repository,
        display_name=display_name,
        public_name=public_name or None,
        category=category or "other",
        privacy_level=privacy_level or "private",
        cluster_id=cluster_id or None,
        aliases=_split_aliases(aliases_text),
        manual_verified=manual_verified,
        notes=notes or None,
    )
    return f"saved place: {place.get('id')} / {place.get('display_name')}"


def update_place_for_ui(
    repository,
    *,
    place_id: str,
    display_name: str | None = None,
    public_name: str | None = None,
    category: str | None = None,
    privacy_level: str | None = None,
    manual_verified: bool | None = None,
    notes: str | None = None,
) -> str:
    place = update_place(
        repository,
        place_id=place_id,
        display_name=display_name or None,
        public_name=public_name or None,
        category=category or None,
        privacy_level=privacy_level or None,
        manual_verified=manual_verified,
        notes=notes or None,
    )
    return f"updated place: {place.get('id')} / {place.get('display_name')}"


def link_cluster_for_ui(repository, *, place_id: str, cluster_id: str) -> str:
    place = link_place_cluster(repository, place_id=place_id, cluster_id=cluster_id)
    update_place_cluster_status(repository, cluster_id=cluster_id, status="accepted")
    return f"linked {cluster_id} -> {place.get('id')}"


def add_place_alias_for_ui(repository, *, place_id: str, alias: str) -> str:
    place = add_place_alias(repository, place_id=place_id, alias=alias)
    return f"aliases: {place.get('aliases_json') or '[]'}"


def set_cluster_status_for_ui(repository, *, cluster_id: str, status: str) -> str:
    row = update_place_cluster_status(repository, cluster_id=cluster_id, status=status)
    return f"cluster {row.get('id')} status={row.get('status')}"


def unlink_cluster_for_ui(repository, *, cluster_id: str) -> str:
    count = unlink_place_cluster(repository, cluster_id=cluster_id)
    return f"unlinked places: {count}"


def reassign_places_for_ui(repository, *, date_from: str | None = None, date_to: str | None = None) -> str:
    report = assign_db_places(repository, start_date=date_from, end_date=date_to, dry_run=False)
    return (
        f"assigned media_links={report.media_links}, event_links={report.event_links}, "
        f"event_location_updates={report.event_location_updates}"
    )


def _cluster_row(row: dict[str, Any], *, mode: str) -> list[Any]:
    return [
        row.get("id") or "",
        int(row.get("point_count") or 0),
        int(row.get("photo_count") or 0),
        int(row.get("event_count") or 0),
        row.get("first_seen_at") or "",
        row.get("last_seen_at") or "",
        round(float(row.get("radius_m") or 0.0), 1),
        _linked_place_label(row, mode=mode),
        row.get("status") or "",
        row.get("privacy_level") or "",
        row.get("approximate_location_label") or "",
    ]


def _linked_place_label(row: dict[str, Any], *, mode: str) -> str:
    if not row.get("linked_place_id"):
        return ""
    place = {
        "display_name": row.get("linked_display_name"),
        "public_name": row.get("linked_public_name"),
        "category": row.get("linked_category"),
        "privacy_level": row.get("linked_privacy_level") or "private",
    }
    if mode == "private":
        return str(row.get("linked_display_name") or row.get("linked_public_name") or row.get("linked_category") or row.get("linked_place_id"))
    return public_place_label(place)


def _split_aliases(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]


def _json_list(raw: object) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw]
    import json

    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _empty_detail() -> dict[str, Any]:
    return {
        "cluster_id": "",
        "summary": "",
        "events": [],
        "photos": [],
        "gallery": [],
        "linked_place_id": "",
        "linked_display_name": "",
        "linked_public_name": "",
        "linked_category": "other",
        "linked_privacy_level": "private",
        "linked_aliases": "",
    }
