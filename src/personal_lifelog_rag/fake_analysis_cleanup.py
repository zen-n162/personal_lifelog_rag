"""Cleanup helpers for test-only fake analysis rows."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Any

from personal_lifelog_rag.db.repository import connect
from personal_lifelog_rag.db.schema import initialize_schema


def cleanup_fake_analysis(
    db_path: str | Path,
    *,
    dry_run: bool = True,
    yes: bool = False,
    include_engine_unavailable: bool = False,
    date: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> dict[str, Any]:
    """Delete fake VLM/embedding rows only when explicitly confirmed."""

    effective_from = date or from_date
    effective_to = date or to_date
    with closing(connect(db_path)) as connection:
        initialize_schema(connection)
        vlm_rows = _target_vlm_rows(connection, include_engine_unavailable=include_engine_unavailable, start=effective_from, end=effective_to)
        embedding_rows = _target_embedding_rows(connection, include_engine_unavailable=include_engine_unavailable, start=effective_from, end=effective_to)
        vlm_ids = [str(row["media_id"]) for row in vlm_rows]
        event_evidence_count = _event_evidence_count(connection, vlm_ids)
        captions_to_clear = _caption_clear_count(connection, vlm_rows)
        report = {
            "dry_run": dry_run,
            "confirmed": bool(yes),
            "range": {"from": effective_from, "to": effective_to},
            "include_engine_unavailable": include_engine_unavailable,
            "media_vlm_rows": len(vlm_rows),
            "media_embeddings_rows": len(embedding_rows),
            "event_evidence_vlm_rows": event_evidence_count,
            "media_item_captions_to_clear": captions_to_clear,
            "deleted": {
                "media_vlm_rows": 0,
                "media_embeddings_rows": 0,
                "event_evidence_vlm_rows": 0,
                "media_item_captions_cleared": 0,
            },
        }
        if dry_run:
            return report
        if not yes:
            report["error"] = "cleanup-fake-analysis requires --yes for real deletion"
            return report
        if vlm_ids:
            report["deleted"]["event_evidence_vlm_rows"] = _delete_event_evidence(connection, vlm_ids)
            report["deleted"]["media_item_captions_cleared"] = _clear_fake_captions(connection, vlm_rows)
            report["deleted"]["media_vlm_rows"] = _delete_vlm_rows(connection, vlm_ids)
        report["deleted"]["media_embeddings_rows"] = _delete_embedding_rows(connection, embedding_rows)
        connection.commit()
        return report


def format_cleanup_fake_analysis(report: dict[str, Any]) -> str:
    lines = [
        "Fake analysis cleanup",
        f"- range: {report['range']['from'] or 'all'}..{report['range']['to'] or 'all'}",
        f"- dry_run: {report['dry_run']}",
        f"- include_engine_unavailable: {report['include_engine_unavailable']}",
        f"- media_vlm rows: {report['media_vlm_rows']}",
        f"- media_embeddings rows: {report['media_embeddings_rows']}",
        f"- event_evidence vlm rows: {report['event_evidence_vlm_rows']}",
        f"- media_items captions to clear: {report['media_item_captions_to_clear']}",
    ]
    if report.get("error"):
        lines.append(f"- error: {report['error']}")
    deleted = report.get("deleted") or {}
    if not report.get("dry_run"):
        lines.append("deleted:")
        for key, value in deleted.items():
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _target_vlm_rows(connection, *, include_engine_unavailable: bool, start: str | None, end: str | None) -> list[dict[str, Any]]:
    clauses = [
        "((LOWER(COALESCE(media_vlm.vlm_engine, '')) LIKE '%fake%') OR (LOWER(COALESCE(media_vlm.model_name, '')) LIKE '%fake%'))"
    ]
    params: list[Any] = []
    if include_engine_unavailable:
        clauses[0] = f"({clauses[0]} OR media_vlm.status = 'engine_unavailable')"
    timestamp = "COALESCE(media_items.captured_at, media_items.fallback_captured_at)"
    if start:
        clauses.append(f"substr({timestamp}, 1, 10) >= ?")
        params.append(start)
    if end:
        clauses.append(f"substr({timestamp}, 1, 10) <= ?")
        params.append(end)
    rows = connection.execute(
        f"""
        SELECT media_vlm.media_id, media_vlm.caption, media_vlm.short_caption
        FROM media_vlm
        LEFT JOIN media_items ON media_items.id = media_vlm.media_id
        WHERE {' AND '.join(clauses)}
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def _target_embedding_rows(connection, *, include_engine_unavailable: bool, start: str | None, end: str | None) -> list[dict[str, Any]]:
    clauses = ["LOWER(COALESCE(media_embeddings.embedding_model, '')) LIKE '%fake%'"]
    params: list[Any] = []
    if include_engine_unavailable:
        clauses[0] = f"({clauses[0]} OR media_embeddings.status = 'engine_unavailable')"
    timestamp = "COALESCE(media_items.captured_at, media_items.fallback_captured_at)"
    if start:
        clauses.append(f"substr({timestamp}, 1, 10) >= ?")
        params.append(start)
    if end:
        clauses.append(f"substr({timestamp}, 1, 10) <= ?")
        params.append(end)
    rows = connection.execute(
        f"""
        SELECT media_embeddings.media_id, media_embeddings.embedding_type, media_embeddings.embedding_model
        FROM media_embeddings
        LEFT JOIN media_items ON media_items.id = media_embeddings.media_id
        WHERE {' AND '.join(clauses)}
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def _event_evidence_count(connection, media_ids: list[str]) -> int:
    if not media_ids:
        return 0
    placeholders = ", ".join("?" for _ in media_ids)
    row = connection.execute(
        f"SELECT COUNT(*) AS count FROM event_evidence WHERE evidence_type = 'vlm' AND evidence_id IN ({placeholders})",
        media_ids,
    ).fetchone()
    return int(row["count"] if row else 0)


def _caption_clear_count(connection, vlm_rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in vlm_rows:
        media_id = str(row.get("media_id") or "")
        caption = row.get("short_caption") or row.get("caption")
        analysis_caption = row.get("caption") or row.get("short_caption")
        current = connection.execute("SELECT caption, analysis_json FROM media_items WHERE id = ?", (media_id,)).fetchone()
        if current and caption and current["caption"] == caption:
            count += 1
        elif current and analysis_caption and current["analysis_json"] and analysis_caption in str(current["analysis_json"]):
            count += 1
    return count


def _delete_event_evidence(connection, media_ids: list[str]) -> int:
    placeholders = ", ".join("?" for _ in media_ids)
    cursor = connection.execute(
        f"DELETE FROM event_evidence WHERE evidence_type = 'vlm' AND evidence_id IN ({placeholders})",
        media_ids,
    )
    return int(cursor.rowcount or 0)


def _clear_fake_captions(connection, vlm_rows: list[dict[str, Any]]) -> int:
    cleared = 0
    for row in vlm_rows:
        media_id = str(row.get("media_id") or "")
        caption = row.get("short_caption") or row.get("caption")
        current = connection.execute("SELECT caption FROM media_items WHERE id = ?", (media_id,)).fetchone()
        if current and caption and current["caption"] == caption:
            cursor = connection.execute("UPDATE media_items SET caption = NULL, analysis_json = NULL WHERE id = ?", (media_id,))
            cleared += int(cursor.rowcount or 0)
    return cleared


def _delete_vlm_rows(connection, media_ids: list[str]) -> int:
    placeholders = ", ".join("?" for _ in media_ids)
    cursor = connection.execute(f"DELETE FROM media_vlm WHERE media_id IN ({placeholders})", media_ids)
    return int(cursor.rowcount or 0)


def _delete_embedding_rows(connection, rows: list[dict[str, Any]]) -> int:
    deleted = 0
    for row in rows:
        cursor = connection.execute(
            """
            DELETE FROM media_embeddings
            WHERE media_id = ? AND embedding_type = ? AND embedding_model = ?
            """,
            (row["media_id"], row["embedding_type"], row["embedding_model"]),
        )
        deleted += int(cursor.rowcount or 0)
    return deleted
