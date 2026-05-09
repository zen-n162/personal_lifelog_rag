"""Cleanup helpers for non-success local VLM rows."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Any

from personal_lifelog_rag.db.repository import connect
from personal_lifelog_rag.db.schema import initialize_schema


DEFAULT_CLEANUP_STATUSES = ("failed", "engine_unavailable")
VALID_CLEANUP_STATUSES = {
    "pending",
    "success",
    "skipped",
    "failed",
    "no_visual_content",
    "engine_unavailable",
}


def cleanup_vlm_status(
    db_path: str | Path,
    *,
    statuses: list[str] | tuple[str, ...] | None = None,
    engine: str | None = None,
    date: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    dry_run: bool = True,
    yes: bool = False,
) -> dict[str, Any]:
    """Delete selected non-fake media_vlm rows and their VLM event evidence.

    Fake rows are intentionally excluded here so test-data cleanup remains
    centralized in cleanup-fake-analysis.
    """

    requested_statuses = tuple(statuses or DEFAULT_CLEANUP_STATUSES)
    invalid_statuses = sorted(set(requested_statuses) - VALID_CLEANUP_STATUSES)
    effective_from = date or from_date
    effective_to = date or to_date
    report: dict[str, Any] = {
        "dry_run": dry_run,
        "confirmed": bool(yes),
        "range": {"from": effective_from, "to": effective_to},
        "engine": engine,
        "statuses": list(requested_statuses),
        "invalid_statuses": invalid_statuses,
        "fake_rows_ignored": 0,
        "media_vlm_rows": 0,
        "status_counts": {},
        "event_evidence_vlm_rows": 0,
        "target_media_ids_sample": [],
        "deleted": {"media_vlm_rows": 0, "event_evidence_vlm_rows": 0},
    }
    if invalid_statuses:
        report["error"] = f"Invalid VLM status value(s): {', '.join(invalid_statuses)}"
        return report

    with closing(connect(db_path)) as connection:
        initialize_schema(connection)
        rows = _target_vlm_rows(
            connection,
            statuses=requested_statuses,
            engine=engine,
            start=effective_from,
            end=effective_to,
        )
        ignored = _ignored_fake_rows(
            connection,
            statuses=requested_statuses,
            engine=engine,
            start=effective_from,
            end=effective_to,
        )
        media_ids = [str(row["media_id"]) for row in rows]
        report["media_vlm_rows"] = len(rows)
        report["fake_rows_ignored"] = ignored
        report["status_counts"] = _status_counts(rows)
        report["event_evidence_vlm_rows"] = _event_evidence_count(connection, media_ids)
        report["target_media_ids_sample"] = media_ids[:20]
        if dry_run:
            return report
        if not yes:
            report["error"] = "cleanup-vlm-status requires --yes for real deletion"
            return report
        if media_ids:
            report["deleted"]["event_evidence_vlm_rows"] = _delete_event_evidence(connection, media_ids)
            report["deleted"]["media_vlm_rows"] = _delete_vlm_rows(connection, media_ids)
        connection.commit()
        return report


def format_cleanup_vlm_status(report: dict[str, Any]) -> str:
    lines = [
        "VLM status cleanup",
        f"- range: {report['range']['from'] or 'all'}..{report['range']['to'] or 'all'}",
        f"- engine: {report.get('engine') or 'all non-fake engines'}",
        f"- statuses: {', '.join(report.get('statuses') or [])}",
        f"- dry_run: {report['dry_run']}",
        f"- media_vlm rows: {report['media_vlm_rows']}",
        f"- event_evidence vlm rows: {report['event_evidence_vlm_rows']}",
        f"- fake rows ignored: {report.get('fake_rows_ignored', 0)}",
    ]
    status_counts = report.get("status_counts") or {}
    if status_counts:
        lines.append("status counts:")
        for status, count in sorted(status_counts.items()):
            lines.append(f"- {status}: {count}")
    sample = report.get("target_media_ids_sample") or []
    if sample:
        lines.append("target media_id sample:")
        for media_id in sample[:10]:
            lines.append(f"- {media_id}")
    if report.get("error"):
        lines.append(f"- error: {report['error']}")
    if not report.get("dry_run"):
        lines.append("deleted:")
        for key, value in (report.get("deleted") or {}).items():
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _target_vlm_rows(
    connection,
    *,
    statuses: tuple[str, ...],
    engine: str | None,
    start: str | None,
    end: str | None,
) -> list[dict[str, Any]]:
    clauses, params = _base_filters(statuses=statuses, engine=engine, start=start, end=end)
    clauses.extend(
        [
            "LOWER(COALESCE(media_vlm.vlm_engine, '')) NOT LIKE '%fake%'",
            "LOWER(COALESCE(media_vlm.model_name, '')) NOT LIKE '%fake%'",
        ]
    )
    rows = connection.execute(
        f"""
        SELECT media_vlm.media_id, media_vlm.status, media_vlm.vlm_engine, media_vlm.model_name
        FROM media_vlm
        LEFT JOIN media_items ON media_items.id = media_vlm.media_id
        WHERE {' AND '.join(clauses)}
        ORDER BY media_vlm.status ASC, media_vlm.media_id ASC
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def _ignored_fake_rows(
    connection,
    *,
    statuses: tuple[str, ...],
    engine: str | None,
    start: str | None,
    end: str | None,
) -> int:
    clauses, params = _base_filters(statuses=statuses, engine=engine, start=start, end=end)
    clauses.append(
        "(LOWER(COALESCE(media_vlm.vlm_engine, '')) LIKE '%fake%' OR LOWER(COALESCE(media_vlm.model_name, '')) LIKE '%fake%')"
    )
    row = connection.execute(
        f"""
        SELECT COUNT(*) AS count
        FROM media_vlm
        LEFT JOIN media_items ON media_items.id = media_vlm.media_id
        WHERE {' AND '.join(clauses)}
        """,
        params,
    ).fetchone()
    return int(row["count"] if row else 0)


def _base_filters(
    *,
    statuses: tuple[str, ...],
    engine: str | None,
    start: str | None,
    end: str | None,
) -> tuple[list[str], list[Any]]:
    placeholders = ", ".join("?" for _ in statuses)
    clauses = [f"media_vlm.status IN ({placeholders})"]
    params: list[Any] = list(statuses)
    if engine:
        clauses.append("media_vlm.vlm_engine = ?")
        params.append(engine)
    timestamp = "COALESCE(media_items.captured_at, media_items.fallback_captured_at)"
    if start:
        clauses.append(f"substr({timestamp}, 1, 10) >= ?")
        params.append(start)
    if end:
        clauses.append(f"substr({timestamp}, 1, 10) <= ?")
        params.append(end)
    return clauses, params


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "(null)")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _event_evidence_count(connection, media_ids: list[str]) -> int:
    if not media_ids:
        return 0
    placeholders = ", ".join("?" for _ in media_ids)
    row = connection.execute(
        f"SELECT COUNT(*) AS count FROM event_evidence WHERE evidence_type = 'vlm' AND evidence_id IN ({placeholders})",
        media_ids,
    ).fetchone()
    return int(row["count"] if row else 0)


def _delete_event_evidence(connection, media_ids: list[str]) -> int:
    placeholders = ", ".join("?" for _ in media_ids)
    cursor = connection.execute(
        f"DELETE FROM event_evidence WHERE evidence_type = 'vlm' AND evidence_id IN ({placeholders})",
        media_ids,
    )
    return int(cursor.rowcount or 0)


def _delete_vlm_rows(connection, media_ids: list[str]) -> int:
    placeholders = ", ".join("?" for _ in media_ids)
    cursor = connection.execute(f"DELETE FROM media_vlm WHERE media_id IN ({placeholders})", media_ids)
    return int(cursor.rowcount or 0)
