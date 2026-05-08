"""Build and query a local structured index of LINE call messages."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from personal_lifelog_rag.line.call_parser import CallStatus, parse_line_call_text


CALL_STATUSES: tuple[CallStatus, ...] = ("completed", "missed", "unanswered", "canceled", "unknown")


@dataclass
class BuildCallIndexReport:
    start_date: str | None = None
    end_date: str | None = None
    dry_run: bool = False
    force: bool = False
    messages_scanned: int = 0
    call_events_found: int = 0
    call_events_saved: int = 0
    deleted_existing: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def build_call_index(
    repository,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> BuildCallIndexReport:
    """Extract structured call events from LINE messages into line_call_events."""

    line_messages = repository.list_line_messages(
        start_date=start_date,
        end_date=end_date,
        limit=1_000_000,
    )
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for message in line_messages:
        parsed = parse_line_call_text(message.get("text") or message.get("message_text"))
        if parsed is None:
            continue
        rows.append(
            {
                "message_id": message["id"],
                "chat_id": message.get("chat_id"),
                "sent_at": message.get("sent_at"),
                "sender": message.get("sender") or message.get("sender_name"),
                "call_status": parsed.call_status,
                "duration_sec": parsed.duration_sec,
                "raw_text_short": parsed.raw_text_short,
            }
        )
        warnings.extend(f"{message['id']}: {warning}" for warning in parsed.warnings)

    deleted = 0
    saved = 0
    if not dry_run:
        if force:
            deleted = repository.delete_line_call_events(start_date=start_date, end_date=end_date)
        saved = repository.upsert_line_call_events(rows)

    return BuildCallIndexReport(
        start_date=start_date,
        end_date=end_date,
        dry_run=dry_run,
        force=force,
        messages_scanned=len(line_messages),
        call_events_found=len(rows),
        call_events_saved=saved,
        deleted_existing=deleted,
        status_counts=dict(Counter(str(row["call_status"]) for row in rows)),
        warnings=warnings[:20],
    )


def call_stats(
    repository,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    events = repository.list_line_call_events(
        start_date=start_date,
        end_date=end_date,
        limit=1_000_000,
    )
    status_counts = Counter(str(event.get("call_status") or "unknown") for event in events)
    completed = [event for event in events if event.get("call_status") == "completed"]
    total_duration = sum(int(event.get("duration_sec") or 0) for event in completed)
    daily_completed: dict[str, int] = defaultdict(int)
    monthly_completed: dict[str, int] = defaultdict(int)
    sender_counts: Counter[str] = Counter()
    for event in events:
        sent_at = str(event.get("sent_at") or "")
        if event.get("call_status") == "completed":
            if len(sent_at) >= 10:
                daily_completed[sent_at[:10]] += 1
                monthly_completed[sent_at[:7]] += 1
        sender_counts[str(event.get("sender") or "(unknown)")] += 1

    longest = sorted(
        completed,
        key=lambda event: (int(event.get("duration_sec") or 0), str(event.get("sent_at") or "")),
        reverse=True,
    )[:10]
    return {
        "start_date": start_date,
        "end_date": end_date,
        "total": len(events),
        "status_counts": {status: int(status_counts.get(status, 0)) for status in CALL_STATUSES},
        "total_completed_duration_sec": total_duration,
        "average_completed_duration_sec": round(total_duration / len(completed), 2) if completed else 0.0,
        "longest_calls": [_call_preview(event) for event in longest],
        "daily_completed_counts": dict(sorted(daily_completed.items())),
        "monthly_completed_counts": dict(sorted(monthly_completed.items())),
        "sender_counts": dict(sender_counts.most_common(20)),
    }


def search_calls(
    repository,
    *,
    statuses: list[str] | None = None,
    min_duration_sec: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    rows = repository.list_line_call_events(
        start_date=start_date,
        end_date=end_date,
        statuses=statuses,
        min_duration_sec=min_duration_sec,
        limit=1_000_000,
    )
    if min_duration_sec is not None or statuses == ["completed"]:
        rows = sorted(rows, key=lambda row: (int(row.get("duration_sec") or 0), str(row.get("sent_at") or "")), reverse=True)
    return {
        "filters": {
            "statuses": statuses or [],
            "min_duration_sec": min_duration_sec,
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
        },
        "results": [_call_preview(row) for row in rows[: max(limit, 0)]],
    }


def summarize_call_events(call_events: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a compact per-day call summary for ranking/search output."""

    status_counts = Counter(str(event.get("call_status") or "unknown") for event in call_events)
    completed = [event for event in call_events if event.get("call_status") == "completed"]
    total_duration = sum(int(event.get("duration_sec") or 0) for event in completed)
    return {
        "total": len(call_events),
        "completed": int(status_counts.get("completed", 0)),
        "missed": int(status_counts.get("missed", 0)),
        "unanswered": int(status_counts.get("unanswered", 0)),
        "canceled": int(status_counts.get("canceled", 0)),
        "unknown": int(status_counts.get("unknown", 0)),
        "total_duration_sec": total_duration,
        "max_duration_sec": max((int(event.get("duration_sec") or 0) for event in completed), default=0),
    }


def format_build_call_index_report(report: BuildCallIndexReport) -> str:
    verb = "Dry-run call index" if report.dry_run else "Built call index"
    lines = [
        f"{verb}:",
        f"- range: {_range_label(report.start_date, report.end_date)}",
        f"- messages scanned: {report.messages_scanned}",
        f"- call events found: {report.call_events_found}",
        f"- call events saved: {report.call_events_saved}",
        f"- deleted existing: {report.deleted_existing}",
        "- status counts:",
    ]
    for status in CALL_STATUSES:
        lines.append(f"  - {status}: {report.status_counts.get(status, 0)}")
    if report.warnings:
        lines.append("- warnings:")
        for warning in report.warnings:
            lines.append(f"  - {warning}")
    return "\n".join(lines)


def format_call_stats(report: dict[str, Any]) -> str:
    counts = report["status_counts"]
    lines = [
        "Call Stats",
        f"- range: {_range_label(report.get('start_date'), report.get('end_date'))}",
        f"- total call events: {report['total']}",
        f"- completed: {counts['completed']}",
        f"- missed: {counts['missed']}",
        f"- unanswered: {counts['unanswered']}",
        f"- canceled: {counts['canceled']}",
        f"- unknown: {counts['unknown']}",
        f"- total completed duration: {format_duration(report['total_completed_duration_sec'])}",
        f"- average completed duration: {format_duration(report['average_completed_duration_sec'])}",
        "",
        "longest calls:",
    ]
    if report["longest_calls"]:
        for index, call in enumerate(report["longest_calls"], start=1):
            lines.append(
                f"{index}. {call['date']} {call['time']} {call['sender']} "
                f"{call['call_status']} duration={call['duration']}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "daily completed call counts:"])
    if report["daily_completed_counts"]:
        for date, count in report["daily_completed_counts"].items():
            lines.append(f"- {date}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "monthly completed call counts:"])
    if report["monthly_completed_counts"]:
        for month, count in report["monthly_completed_counts"].items():
            lines.append(f"- {month}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "sender counts:"])
    if report["sender_counts"]:
        for sender, count in report["sender_counts"].items():
            lines.append(f"- {sender}: {count}")
    else:
        lines.append("- none")
    return "\n".join(lines)


def format_search_calls_report(report: dict[str, Any]) -> str:
    filters = report["filters"]
    filter_parts = []
    if filters["statuses"]:
        filter_parts.extend(filters["statuses"])
    if filters["min_duration_sec"] is not None:
        filter_parts.append(f"min_duration_sec={filters['min_duration_sec']}")
    if filters["start_date"] or filters["end_date"]:
        filter_parts.append(_range_label(filters["start_date"], filters["end_date"]))
    lines = [
        "Call Search",
        f"filters: {', '.join(filter_parts) if filter_parts else 'none'}",
        "",
    ]
    if not report["results"]:
        lines.append("No call events found.")
        return "\n".join(lines)
    for index, call in enumerate(report["results"], start=1):
        lines.append(
            f"{index}. {call['date']} {call['time']} {call['sender']} "
            f"{call['call_status']} duration={call['duration']}"
        )
    return "\n".join(lines)


def format_duration(seconds: int | float | None) -> str:
    total = int(seconds or 0)
    hours, remainder = divmod(total, 3600)
    minutes, resolved_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{resolved_seconds:02d}"
    return f"{minutes}:{resolved_seconds:02d}"


def _call_preview(event: dict[str, Any]) -> dict[str, Any]:
    sent_at = str(event.get("sent_at") or "")
    return {
        "message_id": event.get("message_id"),
        "date": sent_at[:10] if len(sent_at) >= 10 else "",
        "time": _time_label(sent_at),
        "sender": event.get("sender") or "",
        "call_status": event.get("call_status") or "unknown",
        "duration_sec": int(event.get("duration_sec") or 0),
        "duration": format_duration(event.get("duration_sec")),
    }


def _time_label(value: str) -> str:
    try:
        return datetime.fromisoformat(value).strftime("%H:%M")
    except ValueError:
        return value[11:16] if len(value) >= 16 else ""


def _range_label(start_date: str | None, end_date: str | None) -> str:
    if start_date and end_date and start_date != end_date:
        return f"{start_date}..{end_date}"
    if start_date:
        return start_date
    if end_date:
        return f"..{end_date}"
    return "all"
