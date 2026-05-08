"""Assign registered local place names to generated events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from personal_lifelog_rag.places.matcher import match_place
from personal_lifelog_rag.places.schemas import Place


@dataclass(frozen=True)
class PlaceAssignment:
    event_id: str
    date: str | None
    title: str | None
    current_location_name: str | None
    new_location_name: str
    place_id: str
    distance_m: float
    dry_run: bool
    updated: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "date": self.date,
            "title": self.title,
            "current_location_name": self.current_location_name,
            "new_location_name": self.new_location_name,
            "place_id": self.place_id,
            "distance_m": round(self.distance_m, 1),
            "dry_run": self.dry_run,
            "updated": self.updated,
        }


@dataclass(frozen=True)
class AssignPlacesReport:
    start_date: str | None
    end_date: str | None
    dry_run: bool
    events_scanned: int = 0
    events_with_gps: int = 0
    matched: int = 0
    updated: int = 0
    already_assigned: int = 0
    skipped_overrides: int = 0
    skipped_user_edited: int = 0
    unmatched: int = 0
    assignments: list[PlaceAssignment] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "range": {"from": self.start_date, "to": self.end_date},
            "dry_run": self.dry_run,
            "events_scanned": self.events_scanned,
            "events_with_gps": self.events_with_gps,
            "matched": self.matched,
            "updated": self.updated,
            "already_assigned": self.already_assigned,
            "skipped_overrides": self.skipped_overrides,
            "skipped_user_edited": self.skipped_user_edited,
            "unmatched": self.unmatched,
            "assignments": [assignment.to_dict() for assignment in self.assignments],
        }


def assign_places_to_events(
    repository,
    *,
    places: list[Place],
    start_date: str | None = None,
    end_date: str | None = None,
    dry_run: bool = False,
) -> AssignPlacesReport:
    events = repository.list_events(
        start_date=start_date,
        end_date=end_date,
        limit=1_000_000,
        include_hidden=True,
    )
    assignments: list[PlaceAssignment] = []
    events_with_gps = 0
    matched = 0
    updated = 0
    already_assigned = 0
    skipped_overrides = 0
    skipped_user_edited = 0
    unmatched = 0

    for event in events:
        if event.get("gps_lat") is None or event.get("gps_lon") is None:
            continue
        events_with_gps += 1
        match = match_place(event.get("gps_lat"), event.get("gps_lon"), places)
        if match is None:
            unmatched += 1
            continue
        matched += 1
        event_id = str(event["id"])
        current_location = event.get("location_name")
        is_user_edited = bool(event.get("is_user_edited"))
        did_update = False
        if current_location == match.display_name:
            already_assigned += 1
        elif _has_location_override(repository, event_id):
            skipped_overrides += 1
        elif is_user_edited:
            skipped_user_edited += 1
        elif not dry_run:
            did_update = repository.update_event_location_name(
                event_id,
                location_name=match.display_name,
            )
            if did_update:
                updated += 1
        assignments.append(
            PlaceAssignment(
                event_id=event_id,
                date=event.get("date"),
                title=event.get("title"),
                current_location_name=current_location,
                new_location_name=match.display_name,
                place_id=match.place_id,
                distance_m=match.distance_m,
                dry_run=dry_run,
                updated=did_update,
            )
        )

    return AssignPlacesReport(
        start_date=start_date,
        end_date=end_date,
        dry_run=dry_run,
        events_scanned=len(events),
        events_with_gps=events_with_gps,
        matched=matched,
        updated=updated,
        already_assigned=already_assigned,
        skipped_overrides=skipped_overrides,
        skipped_user_edited=skipped_user_edited,
        unmatched=unmatched,
        assignments=assignments,
    )


def format_assign_places_report(report: AssignPlacesReport) -> str:
    verb = "Dry-run place assignment" if report.dry_run else "Assigned places"
    lines = [
        verb,
        f"- range: {report.start_date or 'all'}..{report.end_date or 'all'}",
        f"- events scanned: {report.events_scanned}",
        f"- events with GPS: {report.events_with_gps}",
        f"- matched: {report.matched}",
        f"- updated: {report.updated}",
        f"- already assigned: {report.already_assigned}",
        f"- skipped overrides: {report.skipped_overrides}",
        f"- skipped user edited: {report.skipped_user_edited}",
        f"- unmatched: {report.unmatched}",
    ]
    if report.assignments:
        lines.append("")
        lines.append("Matches:")
        for assignment in report.assignments[:30]:
            status = "updated" if assignment.updated else ("dry-run" if report.dry_run else "planned/no-change")
            lines.append(
                f"- {assignment.date} {assignment.event_id}: "
                f"{assignment.new_location_name} "
                f"({assignment.distance_m:.1f}m, {status})"
            )
        if len(report.assignments) > 30:
            lines.append(f"- ... {len(report.assignments) - 30} more match(es)")
    return "\n".join(lines)


def _has_location_override(repository, event_id: str) -> bool:
    checker = getattr(repository, "event_has_location_override", None)
    return bool(checker(event_id)) if checker is not None else False
