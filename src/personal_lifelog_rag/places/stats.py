"""Location-name statistics for generated events."""

from __future__ import annotations

from collections import Counter
from typing import Any

from personal_lifelog_rag.places.schemas import Place
from personal_lifelog_rag.places.location_store import location_place_stats


def place_stats(
    repository,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    places: list[Place] | None = None,
) -> dict[str, Any]:
    """Summarize event place labels without exposing raw GPS coordinates."""

    events = repository.list_events(
        start_date=start_date,
        end_date=end_date,
        limit=1_000_000,
        include_hidden=True,
    )
    event_ids = {str(event["id"]) for event in events}
    evidence = [
        row
        for row in repository.list_event_evidence()
        if str(row.get("event_id")) in event_ids
    ]
    evidence_by_event: dict[str, set[str]] = {}
    for row in evidence:
        evidence_by_event.setdefault(str(row.get("event_id")), set()).add(str(row.get("evidence_type") or "unknown"))

    location_counter: Counter[str] = Counter(
        str(event.get("location_name") or "")
        for event in events
        if str(event.get("location_name") or "").strip()
    )
    unset_count = sum(1 for event in events if not str(event.get("location_name") or "").strip())
    sensitive_names = {
        place.display_name
        for place in places or []
        if place.privacy_level == "sensitive"
    }
    sensitive_event_count = sum(
        count
        for location_name, count in location_counter.items()
        if location_name in sensitive_names
    )
    photo_evidence_event_count = sum(
        1
        for event in events
        if "photo" in evidence_by_event.get(str(event["id"]), set())
    )
    gps_event_count = sum(
        1
        for event in events
        if event.get("gps_lat") is not None and event.get("gps_lon") is not None
    )
    report = {
        "range": {"from": start_date, "to": end_date},
        "total_events": len(events),
        "location_counts": dict(location_counter.most_common()),
        "unset_location_count": unset_count,
        "sensitive_location_event_count": sensitive_event_count,
        "photo_evidence_event_count": photo_evidence_event_count,
        "gps_event_count": gps_event_count,
        "top_locations": [
            {"location_name": name, "event_count": count}
            for name, count in location_counter.most_common(20)
        ],
    }
    try:
        report["location_db"] = location_place_stats(repository)
    except Exception:
        report["location_db"] = {}
    return report


def format_place_stats(report: dict[str, Any]) -> str:
    lines = [
        "Place Stats",
        f"- range: {report['range'].get('from') or 'all'}..{report['range'].get('to') or 'all'}",
        f"- total events: {report['total_events']}",
        f"- location_name未設定イベント: {report['unset_location_count']}",
        f"- sensitive表示名イベント: {report['sensitive_location_event_count']}",
        f"- photo evidence付きイベント: {report['photo_evidence_event_count']}",
        f"- GPS付きイベント: {report['gps_event_count']}",
        f"- location_points: {report.get('location_db', {}).get('location_points', 0)}",
        f"- place_clusters: {report.get('location_db', {}).get('place_clusters', 0)}",
        f"- places: {report.get('location_db', {}).get('places', 0)}",
        f"- event_places: {report.get('location_db', {}).get('event_places', 0)}",
        f"- media_places: {report.get('location_db', {}).get('media_places', 0)}",
        "",
        "Location counts:",
    ]
    if report["location_counts"]:
        for name, count in report["location_counts"].items():
            lines.append(f"- {name}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Top locations:")
    if report["top_locations"]:
        for row in report["top_locations"]:
            lines.append(f"- {row['location_name']}: {row['event_count']}")
    else:
        lines.append("- none")
    return "\n".join(lines)
