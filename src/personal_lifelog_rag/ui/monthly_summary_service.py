"""UI-facing helpers for monthly/range summaries."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from personal_lifelog_rag.retrieval.monthly_summary import build_monthly_summary_report, format_monthly_summary


def monthly_summary_for_ui(
    repository,
    *,
    month: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    mode: str = "public",
    include_hidden: bool = False,
) -> dict[str, Any]:
    """Build a UI-friendly monthly summary payload without exposing GPS."""

    start_date, end_date = resolve_monthly_summary_range(month=month, date_from=date_from, date_to=date_to)
    report = build_monthly_summary_report(
        repository,
        start_date=start_date,
        end_date=end_date,
        include_hidden=include_hidden and mode == "private",
    )
    media = report["media"]
    metrics = [
        ["events", report["events_count"]],
        ["photos", media["photos"]],
        ["gps_photos", media["gps_photos"]],
        ["line_messages", report["line_messages_count"]],
        ["call_events", report["call_events_count"]],
        ["vlm_success_photos", media["vlm_success_photos"]],
        ["ocr_success_photos", media["ocr_success_photos"]],
    ]
    title_rows = [[title, count] for title, count in report["title_distribution"].items()]
    top_day_rows = [
        [
            row["date"],
            row["events_count"],
            row["photos"],
            row["gps_photos"],
            row["line_messages"],
            row["call_events"],
            row["vlm_success_photos"],
            row["ocr_success_photos"],
        ]
        for row in report["representative_days"]
    ]
    event_rows = []
    for day in report["representative_days"]:
        for event in day.get("events", [])[:5]:
            event_rows.append(
                [
                    day["date"],
                    event.get("start_time") or "",
                    event.get("title") or "",
                    event.get("summary_preview") or "",
                    event.get("confidence") if event.get("confidence") is not None else "",
                    event.get("line_evidence_count") or 0,
                    event.get("photo_evidence_count") or 0,
                    event.get("ocr_evidence_count") or 0,
                    event.get("vlm_evidence_count") or 0,
                ]
            )
    return {
        "report": report,
        "summary_text": format_monthly_summary(report),
        "metrics": metrics,
        "title_distribution_rows": title_rows,
        "representative_day_rows": top_day_rows,
        "representative_event_rows": event_rows,
    }


def resolve_monthly_summary_range(
    *,
    month: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> tuple[str, str]:
    if date_from:
        return date_from, date_to or date_from
    if month:
        normalized = month.strip()
        if len(normalized) == 7 and normalized[4] == "-":
            year_text, month_text = normalized.split("-", 1)
            first = date(int(year_text), int(month_text), 1)
            if first.month == 12:
                next_month = date(first.year + 1, 1, 1)
            else:
                next_month = date(first.year, first.month + 1, 1)
            return first.isoformat(), (next_month - timedelta(days=1)).isoformat()
    raise ValueError("month YYYY-MM or date_from is required")
