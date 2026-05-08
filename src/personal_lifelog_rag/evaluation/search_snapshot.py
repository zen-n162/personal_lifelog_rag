"""Save compact local search snapshots for ranking evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from personal_lifelog_rag.core.privacy import redact_text
from personal_lifelog_rag.retrieval.local_search import LocalSearchOptions, local_text_search


DEFAULT_SEARCH_SNAPSHOT_DIR = Path("eval_outputs")
DEFAULT_SEARCH_SNAPSHOT_QUERIES = ["新宿", "ご飯", "通話"]


@dataclass(frozen=True)
class SearchSnapshotOptions:
    queries: list[str]
    limit: int = 5
    date_from: str | None = None
    date_to: str | None = None


def build_search_snapshot(
    repository,
    options: SearchSnapshotOptions,
    *,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    timestamp = (created_at or datetime.now()).isoformat(timespec="seconds")
    query_reports = []
    for query in options.queries:
        report = local_text_search(
            repository,
            LocalSearchOptions(
                query=query,
                date_from=options.date_from,
                date_to=options.date_to,
                limit=options.limit,
            ),
        )
        query_reports.append(
            {
                "query": query,
                "terms": report.get("terms", []),
                "backend": report.get("backend"),
                "total_dates": report.get("total_dates", 0),
                "results": [_snapshot_result(row) for row in report.get("results", [])],
            }
        )
    return {
        "created_at": timestamp,
        "limit": options.limit,
        "date_from": options.date_from,
        "date_to": options.date_to,
        "queries": query_reports,
    }


def write_search_snapshot(
    snapshot: dict[str, Any],
    *,
    output_dir: str | Path = DEFAULT_SEARCH_SNAPSHOT_DIR,
    now: datetime | None = None,
) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"search_snapshot_{timestamp}.json"
    path.write_text(json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return path


def format_search_snapshot(snapshot: dict[str, Any], *, output_path: Path | None = None) -> str:
    lines = ["Search Snapshot", f"- created_at: {snapshot['created_at']}"]
    if output_path is not None:
        lines.append(f"- saved: {output_path}")
    lines.append(f"- queries: {len(snapshot['queries'])}")
    for query in snapshot["queries"]:
        lines.append("")
        lines.append(f"[{query['query']}] total_dates={query['total_dates']}")
        for index, result in enumerate(query["results"], start=1):
            counts = result["counts"]
            lines.append(
                f"{index}. {result['date']} "
                f"confidence={result['confidence_label']} ({result['confidence_score']:.2f}) "
                f"events={counts['events']} line={counts['line']} photos={counts['photos']}"
            )
            if result["line_samples_redacted"]:
                lines.append(f"   line: {result['line_samples_redacted'][0]}")
            lines.append(f"   evidence: {', '.join(result['evidence_types'])}")
    return "\n".join(lines)


def _snapshot_result(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": row.get("date"),
        "confidence_label": row.get("confidence_label"),
        "confidence_score": row.get("confidence"),
        "score": row.get("ranking_score", row.get("score")),
        "classification": row.get("classification"),
        "reason": row.get("reason"),
        "score_components": row.get("score_components", {}),
        "counts": {
            "events": int(row.get("event_count") or 0),
            "line": int(row.get("line_match_count") or 0),
            "photos": int(row.get("same_day_photo_count", row.get("media_match_count")) or 0),
            "gps_photos": int(row.get("same_day_gps_photo_count") or 0),
        },
        "evidence_types": list(row.get("evidence_types") or []),
        "call_summary": row.get("call_summary") or {},
        "line_samples_redacted": [_line_sample_text(sample) for sample in row.get("line_samples", [])[:5]],
        "events": [
            {
                "id": event.get("id"),
                "time_range": _event_time_range(event),
                "title": redact_text(event.get("title"), max_chars=80),
                "confidence": event.get("confidence"),
                "event_evidence_count": event.get("event_evidence_count"),
                "location_name": redact_text(event.get("location_name"), max_chars=40),
            }
            for event in row.get("events", [])[:5]
        ],
    }


def _line_sample_text(sample: dict[str, Any]) -> str:
    time = str(sample.get("time") or "")
    sender = redact_text(sample.get("sender"), max_chars=24)
    text = redact_text(sample.get("text"), max_chars=60)
    return f"{time} {sender}: {text}".strip()


def _event_time_range(event: dict[str, Any]) -> str:
    start = event.get("start_time")
    end = event.get("end_time")
    if start and end and start != end:
        return f"{str(start)[:5]}-{str(end)[:5]}"
    if start:
        return str(start)[:5]
    return ""
