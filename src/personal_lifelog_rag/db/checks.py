"""SQLite integrity checks for the local lifelog database."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Any

from personal_lifelog_rag.db.repository import connect
from personal_lifelog_rag.db.schema import initialize_schema


SAMPLE_LIMIT = 10
CHAT_ID_LIMIT = 20


def run_db_check(db_path: str | Path) -> dict[str, Any]:
    """Return privacy-conscious DB integrity diagnostics."""

    with closing(connect(db_path)) as connection:
        initialize_schema(connection)
        report = {
            "media_items": _media_item_checks(connection),
            "media_ocr": _media_ocr_checks(connection),
            "media_vlm": _media_vlm_checks(connection),
            "line_messages": _line_message_checks(connection),
            "line_call_events": _line_call_event_checks(connection),
            "events": _event_checks(connection),
            "event_evidence": _event_evidence_checks(connection),
        }
    report["strict"] = _strict_summary(report)
    return report


def format_db_check(report: dict[str, Any]) -> str:
    media = report["media_items"]
    media_ocr = report["media_ocr"]
    media_vlm = report["media_vlm"]
    line = report["line_messages"]
    calls = report["line_call_events"]
    events = report["events"]
    evidence = report["event_evidence"]
    strict = report["strict"]

    lines = ["DB integrity check", ""]
    lines.extend(
        [
            "media_items:",
            f"- total: {media['total']}",
            f"- file_hash NULL: {media['file_hash_null']}",
            f"- unique file_hash: {media['unique_file_hash']}",
            f"- duplicate file_hash groups: {media['duplicate_file_hash_groups']}",
            f"- unique file_path: {media['unique_file_path']}",
            f"- duplicate file_path groups: {media['duplicate_file_path_groups']}",
            f"- captured_at NULL: {media['captured_at_null']}",
            f"- fallback_captured_at NULL: {media['fallback_captured_at_null']}",
            f"- GPSあり: {media['gps_present']}",
            f"- missing files: {media['missing_file_count']}",
            f"- missing thumbnails: {media['missing_thumbnail_count']}",
        ]
    )
    lines.extend(_sample_lines("duplicate file_hash sample IDs", media["duplicate_file_hash_sample_ids"]))
    lines.extend(_sample_lines("duplicate file_path sample IDs", media["duplicate_file_path_sample_ids"]))
    lines.extend(_sample_lines("missing file sample IDs", media["missing_file_sample_ids"]))
    lines.extend(_sample_lines("missing thumbnail sample IDs", media["missing_thumbnail_sample_ids"]))

    lines.extend(
        [
            "",
            "media_ocr:",
            f"- total: {media_ocr['total']}",
            "- status counts:",
        ]
    )
    if media_ocr["status_counts"]:
        for row in media_ocr["status_counts"]:
            lines.append(f"  - {row['status']}: {row['count']}")
    else:
        lines.append("  - none")
    lines.extend(
        [
            f"- success: {media_ocr['success_count']}",
            f"- failed: {media_ocr['failed_count']}",
            f"- engine_unavailable: {media_ocr['engine_unavailable_count']}",
            f"- orphan media_id refs: {media_ocr['orphan_media_refs']}",
            f"- invalid status: {media_ocr['invalid_status_count']}",
            f"- success analyzed_at NULL: {media_ocr['success_analyzed_at_null']}",
            f"- ocr_text too long: {media_ocr['ocr_text_too_long']}",
        ]
    )
    lines.extend(_sample_lines("orphan OCR media IDs", media_ocr["orphan_media_sample_ids"]))
    lines.extend(_sample_lines("invalid OCR status media IDs", media_ocr["invalid_status_sample_ids"]))

    lines.extend(
        [
            "",
            "media_vlm:",
            f"- total: {media_vlm['total']}",
            "- status counts:",
        ]
    )
    if media_vlm["status_counts"]:
        for row in media_vlm["status_counts"]:
            lines.append(f"  - {row['status']}: {row['count']}")
    else:
        lines.append("  - none")
    lines.extend(
        [
            f"- success: {media_vlm['success_count']}",
            f"- failed: {media_vlm['failed_count']}",
            f"- engine_unavailable: {media_vlm['engine_unavailable_count']}",
            f"- orphan media_id refs: {media_vlm['orphan_media_refs']}",
            f"- invalid status: {media_vlm['invalid_status_count']}",
            f"- success caption NULL/empty: {media_vlm['success_caption_empty']}",
            f"- caption too long: {media_vlm['caption_too_long']}",
        ]
    )
    lines.extend(_sample_lines("orphan VLM media IDs", media_vlm["orphan_media_sample_ids"]))
    lines.extend(_sample_lines("invalid VLM status media IDs", media_vlm["invalid_status_sample_ids"]))

    lines.extend(
        [
            "",
            "line_messages:",
            f"- total: {line['total']}",
            f"- duplicate id groups: {line['duplicate_id_groups']}",
            f"- sent_at NULL: {line['sent_at_null']}",
            f"- text NULL/empty: {line['text_null_or_empty']}",
            "- chat_id counts:",
        ]
    )
    if line["chat_id_counts"]:
        for row in line["chat_id_counts"]:
            lines.append(f"  - {row['chat_id']}: {row['count']}")
    else:
        lines.append("  - none")

    lines.extend(
        [
            "",
            "line_call_events:",
            f"- total: {calls['total']}",
            "- call_status counts:",
        ]
    )
    if calls["status_counts"]:
        for row in calls["status_counts"]:
            lines.append(f"  - {row['call_status']}: {row['count']}")
    else:
        lines.append("  - none")
    lines.extend(
        [
            f"- orphan message_id refs: {calls['orphan_message_refs']}",
            f"- negative duration_sec: {calls['negative_duration_sec']}",
            f"- completed duration_sec NULL: {calls['completed_duration_null']}",
        ]
    )
    lines.extend(_sample_lines("orphan call message IDs", calls["orphan_message_sample_ids"]))

    lines.extend(
        [
            "",
            "events:",
            f"- total: {events['total']}",
            f"- date NULL: {events['date_null']}",
            f"- start_time NULL: {events['start_time_null']}",
            f"- end_time NULL: {events['end_time_null']}",
            f"- confidence min/max/avg: {_format_float(events['confidence_min'])} / "
            f"{_format_float(events['confidence_max'])} / {_format_float(events['confidence_avg'])}",
            "- event count by date:",
        ]
    )
    if events["date_counts"]:
        for row in events["date_counts"]:
            lines.append(f"  - {row['date']}: {row['count']}")
    else:
        lines.append("  - none")

    lines.extend(
        [
            "",
            "event_evidence:",
            f"- total: {evidence['total']}",
            "- evidence_type counts:",
        ]
    )
    if evidence["evidence_type_counts"]:
        for row in evidence["evidence_type_counts"]:
            lines.append(f"  - {row['evidence_type']}: {row['count']}")
    else:
        lines.append("  - none")
    lines.extend(
        [
            f"- missing photo evidence refs: {evidence['missing_photo_refs']}",
            f"- missing line evidence refs: {evidence['missing_line_refs']}",
            f"- orphan event refs: {evidence['orphan_event_refs']}",
        ]
    )
    lines.extend(_sample_lines("missing photo evidence IDs", evidence["missing_photo_sample_ids"]))
    lines.extend(_sample_lines("missing line evidence IDs", evidence["missing_line_sample_ids"]))
    lines.extend(_sample_lines("orphan event evidence IDs", evidence["orphan_event_sample_ids"]))

    lines.extend(["", "strict:"])
    lines.append(f"- ok: {strict['ok']}")
    if strict["issues"]:
        for issue in strict["issues"]:
            lines.append(f"  - {issue}")
    else:
        lines.append("  - no severe issues")
    return "\n".join(lines)


def _media_item_checks(connection) -> dict[str, Any]:
    rows = connection.execute(
        """
        SELECT id, file_path, file_hash, captured_at, fallback_captured_at,
               gps_lat, gps_lon, thumbnail_path
        FROM media_items
        ORDER BY id ASC
        """
    ).fetchall()
    total = len(rows)
    missing_file_ids: list[str] = []
    missing_thumbnail_ids: list[str] = []

    for row in rows:
        file_path = row["file_path"]
        if file_path and not Path(str(file_path)).expanduser().exists():
            _append_sample(missing_file_ids, row["id"])
        thumbnail_path = row["thumbnail_path"]
        if thumbnail_path and not Path(str(thumbnail_path)).expanduser().exists():
            _append_sample(missing_thumbnail_ids, row["id"])

    duplicate_hash_rows = _duplicate_value_samples(connection, "media_items", "file_hash")
    duplicate_path_rows = _duplicate_value_samples(connection, "media_items", "file_path")

    return {
        "total": total,
        "file_hash_null": _count(connection, "SELECT COUNT(*) FROM media_items WHERE file_hash IS NULL OR TRIM(file_hash) = ''"),
        "unique_file_hash": _count(connection, "SELECT COUNT(DISTINCT file_hash) FROM media_items WHERE file_hash IS NOT NULL AND TRIM(file_hash) != ''"),
        "duplicate_file_hash_groups": len(duplicate_hash_rows),
        "duplicate_file_hash_sample_ids": _flatten_sample_ids(duplicate_hash_rows),
        "unique_file_path": _count(connection, "SELECT COUNT(DISTINCT file_path) FROM media_items WHERE file_path IS NOT NULL AND TRIM(file_path) != ''"),
        "duplicate_file_path_groups": len(duplicate_path_rows),
        "duplicate_file_path_sample_ids": _flatten_sample_ids(duplicate_path_rows),
        "captured_at_null": _count(connection, "SELECT COUNT(*) FROM media_items WHERE captured_at IS NULL OR TRIM(captured_at) = ''"),
        "fallback_captured_at_null": _count(connection, "SELECT COUNT(*) FROM media_items WHERE fallback_captured_at IS NULL OR TRIM(fallback_captured_at) = ''"),
        "gps_present": _count(connection, "SELECT COUNT(*) FROM media_items WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL"),
        "missing_file_count": _missing_path_count(rows, "file_path"),
        "missing_file_sample_ids": missing_file_ids,
        "missing_thumbnail_count": _missing_path_count([row for row in rows if row["thumbnail_path"]], "thumbnail_path"),
        "missing_thumbnail_sample_ids": missing_thumbnail_ids,
    }


def _line_message_checks(connection) -> dict[str, Any]:
    return {
        "total": _count(connection, "SELECT COUNT(*) FROM line_messages"),
        "duplicate_id_groups": len(_duplicate_value_samples(connection, "line_messages", "id", include_null=False)),
        "sent_at_null": _count(connection, "SELECT COUNT(*) FROM line_messages WHERE sent_at IS NULL OR TRIM(sent_at) = ''"),
        "text_null_or_empty": _count(connection, "SELECT COUNT(*) FROM line_messages WHERE text IS NULL OR TRIM(text) = ''"),
        "chat_id_counts": _rows(
            connection,
            """
            SELECT COALESCE(chat_id, '(null)') AS chat_id, COUNT(*) AS count
            FROM line_messages
            GROUP BY chat_id
            ORDER BY count DESC, chat_id ASC
            LIMIT ?
            """,
            [CHAT_ID_LIMIT],
        ),
    }


def _media_ocr_checks(connection) -> dict[str, Any]:
    valid_statuses = ("pending", "success", "skipped", "failed", "no_text", "engine_unavailable")
    placeholders = ", ".join("?" for _ in valid_statuses)
    return {
        "total": _count(connection, "SELECT COUNT(*) FROM media_ocr"),
        "status_counts": _rows(
            connection,
            """
            SELECT COALESCE(status, '(null)') AS status, COUNT(*) AS count
            FROM media_ocr
            GROUP BY status
            ORDER BY count DESC, status ASC
            """,
        ),
        "success_count": _count(connection, "SELECT COUNT(*) FROM media_ocr WHERE status = 'success'"),
        "failed_count": _count(connection, "SELECT COUNT(*) FROM media_ocr WHERE status = 'failed'"),
        "engine_unavailable_count": _count(connection, "SELECT COUNT(*) FROM media_ocr WHERE status = 'engine_unavailable'"),
        "orphan_media_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM media_ocr
            LEFT JOIN media_items ON media_items.id = media_ocr.media_id
            WHERE media_items.id IS NULL
            """,
        ),
        "orphan_media_sample_ids": _sample_query(
            connection,
            """
            SELECT media_ocr.media_id AS id
            FROM media_ocr
            LEFT JOIN media_items ON media_items.id = media_ocr.media_id
            WHERE media_items.id IS NULL
            ORDER BY media_ocr.media_id ASC
            LIMIT ?
            """,
        ),
        "invalid_status_count": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM media_ocr
            WHERE status IS NULL OR status NOT IN ({placeholders})
            """,
            list(valid_statuses),
        ),
        "invalid_status_sample_ids": _sample_query_params(
            connection,
            f"""
            SELECT media_id AS id
            FROM media_ocr
            WHERE status IS NULL OR status NOT IN ({placeholders})
            ORDER BY media_id ASC
            LIMIT ?
            """,
            list(valid_statuses),
        ),
        "success_analyzed_at_null": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM media_ocr
            WHERE status = 'success'
              AND (analyzed_at IS NULL OR TRIM(analyzed_at) = '')
            """,
        ),
        "ocr_text_too_long": _count(
            connection,
            "SELECT COUNT(*) FROM media_ocr WHERE LENGTH(COALESCE(ocr_text, '')) > 20000",
        ),
    }


def _media_vlm_checks(connection) -> dict[str, Any]:
    valid_statuses = ("pending", "success", "skipped", "failed", "no_visual_content", "engine_unavailable")
    placeholders = ", ".join("?" for _ in valid_statuses)
    return {
        "total": _count(connection, "SELECT COUNT(*) FROM media_vlm"),
        "status_counts": _rows(
            connection,
            """
            SELECT COALESCE(status, '(null)') AS status, COUNT(*) AS count
            FROM media_vlm
            GROUP BY status
            ORDER BY count DESC, status ASC
            """,
        ),
        "success_count": _count(connection, "SELECT COUNT(*) FROM media_vlm WHERE status = 'success'"),
        "failed_count": _count(connection, "SELECT COUNT(*) FROM media_vlm WHERE status = 'failed'"),
        "engine_unavailable_count": _count(connection, "SELECT COUNT(*) FROM media_vlm WHERE status = 'engine_unavailable'"),
        "orphan_media_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM media_vlm
            LEFT JOIN media_items ON media_items.id = media_vlm.media_id
            WHERE media_items.id IS NULL
            """,
        ),
        "orphan_media_sample_ids": _sample_query(
            connection,
            """
            SELECT media_vlm.media_id AS id
            FROM media_vlm
            LEFT JOIN media_items ON media_items.id = media_vlm.media_id
            WHERE media_items.id IS NULL
            ORDER BY media_vlm.media_id ASC
            LIMIT ?
            """,
        ),
        "invalid_status_count": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM media_vlm
            WHERE status IS NULL OR status NOT IN ({placeholders})
            """,
            list(valid_statuses),
        ),
        "invalid_status_sample_ids": _sample_query_params(
            connection,
            f"""
            SELECT media_id AS id
            FROM media_vlm
            WHERE status IS NULL OR status NOT IN ({placeholders})
            ORDER BY media_id ASC
            LIMIT ?
            """,
            list(valid_statuses),
        ),
        "success_caption_empty": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM media_vlm
            WHERE status = 'success'
              AND (
                  (caption IS NULL OR TRIM(caption) = '')
                  OR (short_caption IS NULL OR TRIM(short_caption) = '')
              )
            """,
        ),
        "caption_too_long": _count(
            connection,
            "SELECT COUNT(*) FROM media_vlm WHERE LENGTH(COALESCE(caption, '')) > 2000",
        ),
    }


def _line_call_event_checks(connection) -> dict[str, Any]:
    return {
        "total": _count(connection, "SELECT COUNT(*) FROM line_call_events"),
        "status_counts": _rows(
            connection,
            """
            SELECT COALESCE(call_status, '(null)') AS call_status, COUNT(*) AS count
            FROM line_call_events
            GROUP BY call_status
            ORDER BY count DESC, call_status ASC
            """,
        ),
        "orphan_message_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM line_call_events
            LEFT JOIN line_messages ON line_messages.id = line_call_events.message_id
            WHERE line_messages.id IS NULL
            """,
        ),
        "orphan_message_sample_ids": _sample_query(
            connection,
            """
            SELECT line_call_events.message_id AS id
            FROM line_call_events
            LEFT JOIN line_messages ON line_messages.id = line_call_events.message_id
            WHERE line_messages.id IS NULL
            ORDER BY line_call_events.message_id ASC
            LIMIT ?
            """,
        ),
        "negative_duration_sec": _count(
            connection,
            "SELECT COUNT(*) FROM line_call_events WHERE duration_sec < 0",
        ),
        "completed_duration_null": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM line_call_events
            WHERE call_status = 'completed'
              AND duration_sec IS NULL
            """,
        ),
    }


def _event_checks(connection) -> dict[str, Any]:
    stats = connection.execute(
        """
        SELECT
            MIN(confidence) AS confidence_min,
            MAX(confidence) AS confidence_max,
            AVG(confidence) AS confidence_avg
        FROM events
        WHERE confidence IS NOT NULL
        """
    ).fetchone()
    return {
        "total": _count(connection, "SELECT COUNT(*) FROM events"),
        "date_null": _count(connection, "SELECT COUNT(*) FROM events WHERE date IS NULL OR TRIM(date) = ''"),
        "start_time_null": _count(connection, "SELECT COUNT(*) FROM events WHERE start_time IS NULL OR TRIM(start_time) = ''"),
        "end_time_null": _count(connection, "SELECT COUNT(*) FROM events WHERE end_time IS NULL OR TRIM(end_time) = ''"),
        "confidence_min": stats["confidence_min"],
        "confidence_max": stats["confidence_max"],
        "confidence_avg": stats["confidence_avg"],
        "date_counts": _rows(
            connection,
            """
            SELECT COALESCE(date, '(null)') AS date, COUNT(*) AS count
            FROM events
            GROUP BY date
            ORDER BY date ASC
            LIMIT 100
            """,
        ),
    }


def _event_evidence_checks(connection) -> dict[str, Any]:
    return {
        "total": _count(connection, "SELECT COUNT(*) FROM event_evidence"),
        "evidence_type_counts": _rows(
            connection,
            """
            SELECT evidence_type, COUNT(*) AS count
            FROM event_evidence
            GROUP BY evidence_type
            ORDER BY count DESC, evidence_type ASC
            """,
        ),
        "missing_photo_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM event_evidence
            LEFT JOIN media_items ON media_items.id = event_evidence.evidence_id
            WHERE event_evidence.evidence_type = 'photo'
              AND media_items.id IS NULL
            """,
        ),
        "missing_photo_sample_ids": _sample_query(
            connection,
            """
            SELECT event_evidence.evidence_id AS id
            FROM event_evidence
            LEFT JOIN media_items ON media_items.id = event_evidence.evidence_id
            WHERE event_evidence.evidence_type = 'photo'
              AND media_items.id IS NULL
            ORDER BY event_evidence.evidence_id ASC
            LIMIT ?
            """,
        ),
        "missing_line_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM event_evidence
            LEFT JOIN line_messages ON line_messages.id = event_evidence.evidence_id
            WHERE event_evidence.evidence_type = 'line'
              AND line_messages.id IS NULL
            """,
        ),
        "missing_line_sample_ids": _sample_query(
            connection,
            """
            SELECT event_evidence.evidence_id AS id
            FROM event_evidence
            LEFT JOIN line_messages ON line_messages.id = event_evidence.evidence_id
            WHERE event_evidence.evidence_type = 'line'
              AND line_messages.id IS NULL
            ORDER BY event_evidence.evidence_id ASC
            LIMIT ?
            """,
        ),
        "orphan_event_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM event_evidence
            LEFT JOIN events ON events.id = event_evidence.event_id
            WHERE events.id IS NULL
            """,
        ),
        "orphan_event_sample_ids": _sample_query(
            connection,
            """
            SELECT event_evidence.event_id AS id
            FROM event_evidence
            LEFT JOIN events ON events.id = event_evidence.event_id
            WHERE events.id IS NULL
            ORDER BY event_evidence.event_id ASC
            LIMIT ?
            """,
        ),
    }


def _strict_summary(report: dict[str, Any]) -> dict[str, Any]:
    media = report["media_items"]
    media_ocr = report["media_ocr"]
    media_vlm = report["media_vlm"]
    line = report["line_messages"]
    calls = report["line_call_events"]
    events = report["events"]
    evidence = report["event_evidence"]
    issues: list[str] = []

    if line["duplicate_id_groups"]:
        issues.append(f"duplicate line_messages id groups: {line['duplicate_id_groups']}")
    if calls["orphan_message_refs"]:
        issues.append(f"line_call_events orphan message refs: {calls['orphan_message_refs']}")
    if calls["negative_duration_sec"]:
        issues.append(f"line_call_events negative duration_sec: {calls['negative_duration_sec']}")
    if media["duplicate_file_path_groups"]:
        issues.append(f"duplicate file_path groups: {media['duplicate_file_path_groups']}")
    if media["duplicate_file_hash_groups"]:
        issues.append(f"duplicate file_hash groups: {media['duplicate_file_hash_groups']}")
    if media["file_hash_null"] >= _file_hash_null_threshold(media["total"]):
        issues.append(f"file_hash NULL/empty count is high: {media['file_hash_null']}")
    if media_ocr["orphan_media_refs"]:
        issues.append(f"media_ocr orphan media refs: {media_ocr['orphan_media_refs']}")
    if media_ocr["invalid_status_count"]:
        issues.append(f"media_ocr invalid status rows: {media_ocr['invalid_status_count']}")
    if media_ocr["success_analyzed_at_null"]:
        issues.append(f"media_ocr success analyzed_at NULL: {media_ocr['success_analyzed_at_null']}")
    if media_vlm["orphan_media_refs"]:
        issues.append(f"media_vlm orphan media refs: {media_vlm['orphan_media_refs']}")
    if media_vlm["invalid_status_count"]:
        issues.append(f"media_vlm invalid status rows: {media_vlm['invalid_status_count']}")
    if media_vlm["success_caption_empty"]:
        issues.append(f"media_vlm success caption empty: {media_vlm['success_caption_empty']}")
    if events["date_null"]:
        issues.append(f"events date NULL/empty: {events['date_null']}")
    if evidence["orphan_event_refs"]:
        issues.append(f"orphan event_evidence event refs: {evidence['orphan_event_refs']}")
    if evidence["missing_photo_refs"]:
        issues.append(f"photo evidence missing media_items refs: {evidence['missing_photo_refs']}")
    if evidence["missing_line_refs"]:
        issues.append(f"line evidence missing line_messages refs: {evidence['missing_line_refs']}")

    return {
        "ok": not issues,
        "issues": issues,
        "file_hash_null_threshold": _file_hash_null_threshold(media["total"]),
    }


def _file_hash_null_threshold(total: int) -> int:
    if total <= 0:
        return 1
    return max(10, int(total * 0.01))


def _duplicate_value_samples(
    connection,
    table_name: str,
    column_name: str,
    *,
    include_null: bool = False,
) -> list[dict[str, Any]]:
    null_clause = "" if include_null else f"WHERE {column_name} IS NOT NULL AND TRIM({column_name}) != ''"
    rows = connection.execute(
        f"""
        SELECT {column_name} AS value, COUNT(*) AS count
        FROM {table_name}
        {null_clause}
        GROUP BY {column_name}
        HAVING COUNT(*) > 1
        ORDER BY count DESC, value ASC
        LIMIT ?
        """,
        (SAMPLE_LIMIT,),
    ).fetchall()
    samples: list[dict[str, Any]] = []
    for row in rows:
        id_rows = connection.execute(
            f"""
            SELECT id
            FROM {table_name}
            WHERE {column_name} = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (row["value"], SAMPLE_LIMIT),
        ).fetchall()
        samples.append(
            {
                "count": int(row["count"]),
                "ids": [str(id_row["id"]) for id_row in id_rows],
            }
        )
    return samples


def _flatten_sample_ids(groups: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for group in groups:
        for item_id in group["ids"]:
            _append_sample(ids, item_id)
    return ids


def _missing_path_count(rows: list[Any], column_name: str) -> int:
    count = 0
    for row in rows:
        raw_path = row[column_name]
        if raw_path and not Path(str(raw_path)).expanduser().exists():
            count += 1
    return count


def _sample_query(connection, query: str) -> list[str]:
    return [str(row["id"]) for row in connection.execute(query, (SAMPLE_LIMIT,)).fetchall()]


def _sample_query_params(connection, query: str, params: list[Any]) -> list[str]:
    return [str(row["id"]) for row in connection.execute(query, [*params, SAMPLE_LIMIT]).fetchall()]


def _count(connection, query: str) -> int:
    row = connection.execute(query).fetchone()
    return int(row[0] or 0)


def _count_params(connection, query: str, params: list[Any]) -> int:
    row = connection.execute(query, params).fetchone()
    return int(row[0] or 0)


def _rows(connection, query: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(query, params or []).fetchall()]


def _append_sample(samples: list[str], value: Any) -> None:
    if len(samples) < SAMPLE_LIMIT:
        samples.append(str(value))


def _sample_lines(label: str, samples: list[str]) -> list[str]:
    if not samples:
        return [f"- {label}: none"]
    return [f"- {label}: {', '.join(samples)}"]


def _format_float(value: Any) -> str:
    if value is None:
        return "なし"
    return f"{float(value):.3f}"
