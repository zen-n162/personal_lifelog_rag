"""Build private person links for media and events from manual review data.

This module never infers identity. It only propagates user-verified person
links from reviewed face clusters and manually linked LINE speakers.
"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from typing import Any

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.db.schema import initialize_schema
from personal_lifelog_rag.faces.person_service import public_person_name


MEDIA_PEOPLE_FACE_CONFIDENCE = 0.85
EVENT_PEOPLE_FACE_CONFIDENCE = 0.60
EVENT_PEOPLE_LINE_CONFIDENCE = 0.70
EVENT_PEOPLE_COMBINED_CONFIDENCE = 0.90

MEDIA_PEOPLE_SOURCES = {"face_cluster", "manual"}
EVENT_PEOPLE_SOURCES = {"face", "line_speaker", "line_mention", "manual", "combined"}


def build_media_people(
    repository: LifelogRepository,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    dry_run: bool = True,
    replace: bool = False,
    yes: bool = False,
) -> dict[str, Any]:
    if not dry_run and not yes:
        raise ValueError("build-media-people writes require --yes")
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        candidates = _media_people_candidates(connection, start_date=start_date, end_date=end_date)
        replace_count = _count_media_people_in_range(connection, start_date=start_date, end_date=end_date) if replace else 0
        existing_keys = {
            (row["media_id"], row["person_id"], row["source"], row["face_id"])
            for row in connection.execute("SELECT media_id, person_id, source, face_id FROM media_people").fetchall()
        }
        insert_candidates = [
            row
            for row in candidates
            if replace or (row["media_id"], row["person_id"], "face_cluster", row["face_id"]) not in existing_keys
        ]
        report = {
            "dry_run": dry_run,
            "start_date": start_date,
            "end_date": end_date,
            "candidate_count": len(candidates),
            "existing_replaced": replace_count,
            "would_insert": len(insert_candidates),
            "inserted": 0,
            "source_counts": {"face_cluster": len(insert_candidates)},
        }
        if dry_run:
            return report
        now = _now()
        if replace:
            _delete_media_people_in_range(connection, start_date=start_date, end_date=end_date)
        for row in insert_candidates:
            connection.execute(
                """
                INSERT OR IGNORE INTO media_people (
                    media_id, person_id, source, confidence, face_id,
                    face_cluster_id, verified_by_user, created_at, updated_at
                )
                VALUES (?, ?, 'face_cluster', ?, ?, ?, 1, ?, ?)
                """,
                (
                    row["media_id"],
                    row["person_id"],
                    MEDIA_PEOPLE_FACE_CONFIDENCE,
                    row["face_id"],
                    row["face_cluster_id"],
                    now,
                    now,
                ),
            )
        connection.commit()
        report["inserted"] = len(insert_candidates)
        return report


def build_event_people(
    repository: LifelogRepository,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    dry_run: bool = True,
    replace: bool = False,
    yes: bool = False,
) -> dict[str, Any]:
    if not dry_run and not yes:
        raise ValueError("build-event-people writes require --yes")
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        face_rows = _event_people_face_candidates(connection, start_date=start_date, end_date=end_date)
        line_rows = _event_people_line_candidates(connection, start_date=start_date, end_date=end_date)
        combined_rows = _combined_event_people(face_rows, line_rows)
        candidates = [
            *_event_people_rows(face_rows, source="face", confidence=EVENT_PEOPLE_FACE_CONFIDENCE),
            *_event_people_rows(line_rows, source="line_speaker", confidence=EVENT_PEOPLE_LINE_CONFIDENCE),
            *combined_rows,
        ]
        replace_count = _count_event_people_in_range(connection, start_date=start_date, end_date=end_date) if replace else 0
        existing_keys = {
            (row["event_id"], row["person_id"], row["source"])
            for row in connection.execute("SELECT event_id, person_id, source FROM event_people").fetchall()
        }
        insert_candidates = [
            row
            for row in candidates
            if replace or (row["event_id"], row["person_id"], row["source"]) not in existing_keys
        ]
        source_counts: dict[str, int] = {}
        for row in insert_candidates:
            source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
        report = {
            "dry_run": dry_run,
            "start_date": start_date,
            "end_date": end_date,
            "candidate_count": len(candidates),
            "face_candidates": len(face_rows),
            "line_speaker_candidates": len(line_rows),
            "combined_candidates": len(combined_rows),
            "existing_replaced": replace_count,
            "would_insert": len(insert_candidates),
            "inserted": 0,
            "source_counts": source_counts,
        }
        if dry_run:
            return report
        now = _now()
        if replace:
            _delete_event_people_in_range(connection, start_date=start_date, end_date=end_date)
        for row in insert_candidates:
            connection.execute(
                """
                INSERT OR IGNORE INTO event_people (
                    event_id, person_id, source, confidence, evidence_count,
                    media_count, line_count, verified_by_user, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["event_id"],
                    row["person_id"],
                    row["source"],
                    row["confidence"],
                    row["evidence_count"],
                    row["media_count"],
                    row["line_count"],
                    row["verified_by_user"],
                    now,
                    now,
                ),
            )
        connection.commit()
        report["inserted"] = len(insert_candidates)
        return report


def people_stats(
    repository: LifelogRepository,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    public_mode: bool = False,
    limit: int = 10,
) -> dict[str, Any]:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        date_clause, params = _event_date_clause(start_date, end_date, prefix="events")
        media_date_clause, media_params = _media_date_clause(start_date, end_date)
        top_rows = connection.execute(
            f"""
            SELECT persons.id, persons.display_name, persons.public_name, persons.privacy_level,
                   COUNT(DISTINCT event_people.event_id) AS event_count,
                   COUNT(DISTINCT media_people.media_id) AS media_count
            FROM persons
            LEFT JOIN event_people ON event_people.person_id = persons.id
            LEFT JOIN events ON events.id = event_people.event_id
            LEFT JOIN media_people ON media_people.person_id = persons.id
            WHERE 1=1
              {date_clause}
            GROUP BY persons.id
            ORDER BY event_count DESC, media_count DESC, persons.created_at ASC
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()
        return {
            "persons_total": _count(connection, "SELECT COUNT(*) FROM persons"),
            "linked_face_clusters": _count(connection, "SELECT COUNT(*) FROM person_face_clusters"),
            "linked_line_speakers": _count(connection, "SELECT COUNT(*) FROM line_speaker_links"),
            "media_people_count": _count(
                connection,
                f"""
                SELECT COUNT(*)
                FROM media_people
                JOIN media_items ON media_items.id = media_people.media_id
                WHERE 1=1 {media_date_clause}
                """,
                media_params,
            ),
            "event_people_count": _count(
                connection,
                f"""
                SELECT COUNT(*)
                FROM event_people
                JOIN events ON events.id = event_people.event_id
                WHERE 1=1 {date_clause}
                """,
                params,
            ),
            "media_people_source_counts": _rows(
                connection,
                f"""
                SELECT media_people.source, COUNT(*) AS count
                FROM media_people
                JOIN media_items ON media_items.id = media_people.media_id
                WHERE 1=1 {media_date_clause}
                GROUP BY media_people.source
                ORDER BY count DESC, media_people.source ASC
                """,
                media_params,
            ),
            "event_people_source_counts": _rows(
                connection,
                f"""
                SELECT event_people.source, COUNT(*) AS count
                FROM event_people
                JOIN events ON events.id = event_people.event_id
                WHERE 1=1 {date_clause}
                GROUP BY event_people.source
                ORDER BY count DESC, event_people.source ASC
                """,
                params,
            ),
            "top_persons": [_person_stats_row(dict(row), public_mode=public_mode, index=i + 1) for i, row in enumerate(top_rows)],
            "public_mode": public_mode,
            "start_date": start_date,
            "end_date": end_date,
        }


def list_event_people(
    repository: LifelogRepository,
    *,
    date_value: str | None = None,
    event_id: str | None = None,
    public_mode: bool = False,
    limit: int = 50,
) -> list[dict[str, Any]]:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        clauses = []
        params: list[Any] = []
        if date_value:
            clauses.append("events.date = ?")
            params.append(date_value)
        if event_id:
            clauses.append("event_people.event_id = ?")
            params.append(event_id)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        rows = connection.execute(
            f"""
            SELECT event_people.*, events.date, events.title,
                   persons.display_name, persons.public_name, persons.privacy_level
            FROM event_people
            JOIN events ON events.id = event_people.event_id
            JOIN persons ON persons.id = event_people.person_id
            {where}
            ORDER BY events.date ASC, event_people.confidence DESC, event_people.event_id ASC
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()
        return [_event_person_row(dict(row), public_mode=public_mode, index=i + 1) for i, row in enumerate(rows)]


def list_media_people(
    repository: LifelogRepository,
    *,
    date_value: str | None = None,
    public_mode: bool = False,
    limit: int = 50,
) -> list[dict[str, Any]]:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        clauses = []
        params: list[Any] = []
        if date_value:
            clauses.append("substr(COALESCE(media_items.captured_at, media_items.fallback_captured_at), 1, 10) = ?")
            params.append(date_value)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        rows = connection.execute(
            f"""
            SELECT media_people.*, media_items.captured_at, media_items.fallback_captured_at,
                   media_items.file_name, persons.display_name, persons.public_name, persons.privacy_level
            FROM media_people
            JOIN media_items ON media_items.id = media_people.media_id
            JOIN persons ON persons.id = media_people.person_id
            {where}
            ORDER BY COALESCE(media_items.captured_at, media_items.fallback_captured_at) ASC, media_people.media_id ASC
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()
        return [_media_person_row(dict(row), public_mode=public_mode, index=i + 1) for i, row in enumerate(rows)]


def format_people_build_report(report: dict[str, Any], *, title: str) -> str:
    lines = [
        title,
        f"- dry_run: {report.get('dry_run')}",
        f"- range: {report.get('start_date') or '-'} .. {report.get('end_date') or '-'}",
        f"- candidate_count: {report.get('candidate_count', 0)}",
        f"- would_insert: {report.get('would_insert', 0)}",
        f"- inserted: {report.get('inserted', 0)}",
        f"- existing_replaced: {report.get('existing_replaced', 0)}",
        "- source counts:",
    ]
    source_counts = report.get("source_counts") or {}
    if source_counts:
        for source, count in sorted(source_counts.items()):
            lines.append(f"  - {source}: {count}")
    else:
        lines.append("  - none")
    if "face_candidates" in report:
        lines.extend(
            [
                f"- face_candidates: {report.get('face_candidates', 0)}",
                f"- line_speaker_candidates: {report.get('line_speaker_candidates', 0)}",
                f"- combined_candidates: {report.get('combined_candidates', 0)}",
            ]
        )
    return "\n".join(lines)


def format_people_stats(report: dict[str, Any]) -> str:
    lines = [
        "People stats",
        f"- persons total: {report['persons_total']}",
        f"- linked face clusters: {report['linked_face_clusters']}",
        f"- linked line speakers: {report['linked_line_speakers']}",
        f"- media_people: {report['media_people_count']}",
        f"- event_people: {report['event_people_count']}",
        "- media_people sources:",
    ]
    lines.extend(_source_lines(report["media_people_source_counts"]))
    lines.append("- event_people sources:")
    lines.extend(_source_lines(report["event_people_source_counts"]))
    lines.append("- top persons by event count:")
    top = report.get("top_persons") or []
    if top:
        for row in top:
            lines.append(
                f"  - {row.get('person_label') or '(hidden)'} events={row.get('event_count', 0)} "
                f"media={row.get('media_count', 0)}"
            )
    else:
        lines.append("  - none")
    return "\n".join(lines)


def format_event_people(rows: list[dict[str, Any]]) -> str:
    lines = ["Event people"]
    if not rows:
        lines.append("- none")
        return "\n".join(lines)
    for row in rows:
        lines.append(
            f"- event={row.get('event_id')} date={row.get('date') or ''} person={row.get('person_label') or '(hidden)'} "
            f"source={row.get('source')} confidence={_fmt_float(row.get('confidence'))} "
            f"evidence={row.get('evidence_count', 0)} media={row.get('media_count', 0)} line={row.get('line_count', 0)}"
        )
    return "\n".join(lines)


def format_media_people(rows: list[dict[str, Any]]) -> str:
    lines = ["Media people"]
    if not rows:
        lines.append("- none")
        return "\n".join(lines)
    for row in rows:
        lines.append(
            f"- media={row.get('media_id')} captured_at={row.get('captured_at') or row.get('fallback_captured_at') or ''} "
            f"person={row.get('person_label') or '(hidden)'} source={row.get('source')} "
            f"confidence={_fmt_float(row.get('confidence'))}"
        )
    return "\n".join(lines)


def _media_people_candidates(connection, *, start_date: str | None, end_date: str | None) -> list[dict[str, Any]]:
    date_clause, params = _media_date_clause(start_date, end_date)
    rows = connection.execute(
        f"""
        SELECT DISTINCT face_detections.media_id,
               person_face_clusters.person_id,
               face_detections.id AS face_id,
               face_clusters.id AS face_cluster_id
        FROM person_face_clusters
        JOIN persons ON persons.id = person_face_clusters.person_id
        JOIN face_clusters ON face_clusters.id = person_face_clusters.face_cluster_id
        JOIN face_cluster_members ON face_cluster_members.cluster_id = face_clusters.id
        JOIN face_detections ON face_detections.id = face_cluster_members.face_id
        JOIN media_items ON media_items.id = face_detections.media_id
        WHERE persons.manual_verified = 1
          AND person_face_clusters.verified_by_user = 1
          AND face_clusters.status = 'accepted'
          AND COALESCE(face_clusters.review_status, 'reviewed') != 'bad_cluster'
          AND face_detections.status = 'success'
          AND COALESCE(face_detections.review_status, 'accepted') NOT IN ('rejected', 'bad_detection')
          {date_clause}
        ORDER BY face_detections.media_id ASC, person_face_clusters.person_id ASC, face_detections.id ASC
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def _event_people_face_candidates(connection, *, start_date: str | None, end_date: str | None) -> list[dict[str, Any]]:
    date_clause, params = _event_date_clause(start_date, end_date, prefix="events")
    rows = connection.execute(
        f"""
        SELECT event_evidence.event_id,
               media_people.person_id,
               COUNT(DISTINCT event_evidence.evidence_id) AS media_count,
               COUNT(DISTINCT media_people.face_id) AS evidence_count
        FROM event_evidence
        JOIN events ON events.id = event_evidence.event_id
        JOIN media_people ON media_people.media_id = event_evidence.evidence_id
        JOIN persons ON persons.id = media_people.person_id
        WHERE event_evidence.evidence_type IN ('photo', 'image', 'media', 'media_item')
          AND media_people.source IN ('face_cluster', 'manual')
          AND media_people.verified_by_user = 1
          AND persons.manual_verified = 1
          {date_clause}
        GROUP BY event_evidence.event_id, media_people.person_id
        ORDER BY event_evidence.event_id ASC, media_people.person_id ASC
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def _event_people_line_candidates(connection, *, start_date: str | None, end_date: str | None) -> list[dict[str, Any]]:
    date_clause, params = _event_date_clause(start_date, end_date, prefix="events")
    rows = connection.execute(
        f"""
        SELECT event_evidence.event_id,
               line_speaker_links.person_id,
               COUNT(DISTINCT line_messages.id) AS line_count,
               COUNT(DISTINCT line_messages.id) AS evidence_count
        FROM event_evidence
        JOIN events ON events.id = event_evidence.event_id
        JOIN line_messages ON line_messages.id = event_evidence.evidence_id
        JOIN line_speaker_links
          ON line_speaker_links.chat_id = line_messages.chat_id
         AND line_speaker_links.speaker_name = line_messages.sender
        JOIN persons ON persons.id = line_speaker_links.person_id
        WHERE event_evidence.evidence_type IN ('line', 'line_message')
          AND line_speaker_links.verified_by_user = 1
          AND persons.manual_verified = 1
          {date_clause}
        GROUP BY event_evidence.event_id, line_speaker_links.person_id
        ORDER BY event_evidence.event_id ASC, line_speaker_links.person_id ASC
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def _event_people_rows(rows: list[dict[str, Any]], *, source: str, confidence: float) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        media_count = int(row.get("media_count") or 0)
        line_count = int(row.get("line_count") or 0)
        evidence_count = int(row.get("evidence_count") or media_count or line_count or 0)
        result.append(
            {
                "event_id": row["event_id"],
                "person_id": row["person_id"],
                "source": source,
                "confidence": confidence,
                "evidence_count": evidence_count,
                "media_count": media_count,
                "line_count": line_count,
                "verified_by_user": 0,
            }
        )
    return result


def _combined_event_people(face_rows: list[dict[str, Any]], line_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    face_map = {(row["event_id"], row["person_id"]): row for row in face_rows}
    line_map = {(row["event_id"], row["person_id"]): row for row in line_rows}
    combined = []
    for key in sorted(face_map.keys() & line_map.keys()):
        face = face_map[key]
        line = line_map[key]
        combined.append(
            {
                "event_id": key[0],
                "person_id": key[1],
                "source": "combined",
                "confidence": EVENT_PEOPLE_COMBINED_CONFIDENCE,
                "evidence_count": int(face.get("evidence_count") or 0) + int(line.get("evidence_count") or 0),
                "media_count": int(face.get("media_count") or 0),
                "line_count": int(line.get("line_count") or 0),
                "verified_by_user": 0,
            }
        )
    return combined


def _count_media_people_in_range(connection, *, start_date: str | None, end_date: str | None) -> int:
    clause, params = _media_date_clause(start_date, end_date)
    return _count(
        connection,
        f"""
        SELECT COUNT(*)
        FROM media_people
        JOIN media_items ON media_items.id = media_people.media_id
        WHERE media_people.source = 'face_cluster' {clause}
        """,
        params,
    )


def _delete_media_people_in_range(connection, *, start_date: str | None, end_date: str | None) -> None:
    clause, params = _media_date_clause(start_date, end_date)
    connection.execute(
        f"""
        DELETE FROM media_people
        WHERE source = 'face_cluster'
          AND media_id IN (
              SELECT media_items.id
              FROM media_items
              WHERE 1=1 {clause}
          )
        """,
        params,
    )


def _count_event_people_in_range(connection, *, start_date: str | None, end_date: str | None) -> int:
    clause, params = _event_date_clause(start_date, end_date, prefix="events")
    return _count(
        connection,
        f"""
        SELECT COUNT(*)
        FROM event_people
        JOIN events ON events.id = event_people.event_id
        WHERE event_people.source IN ('face', 'line_speaker', 'combined') {clause}
        """,
        params,
    )


def _delete_event_people_in_range(connection, *, start_date: str | None, end_date: str | None) -> None:
    clause, params = _event_date_clause(start_date, end_date, prefix="events")
    connection.execute(
        f"""
        DELETE FROM event_people
        WHERE source IN ('face', 'line_speaker', 'combined')
          AND event_id IN (
              SELECT events.id
              FROM events
              WHERE 1=1 {clause}
          )
        """,
        params,
    )


def _media_date_clause(start_date: str | None, end_date: str | None) -> tuple[str, list[Any]]:
    clauses = []
    params: list[Any] = []
    captured = "substr(COALESCE(media_items.captured_at, media_items.fallback_captured_at), 1, 10)"
    if start_date:
        clauses.append(f"{captured} >= ?")
        params.append(start_date)
    if end_date:
        clauses.append(f"{captured} <= ?")
        params.append(end_date)
    return (" AND " + " AND ".join(clauses) if clauses else "", params)


def _event_date_clause(start_date: str | None, end_date: str | None, *, prefix: str) -> tuple[str, list[Any]]:
    clauses = []
    params: list[Any] = []
    if start_date:
        clauses.append(f"{prefix}.date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append(f"{prefix}.date <= ?")
        params.append(end_date)
    return (" AND " + " AND ".join(clauses) if clauses else "", params)


def _person_stats_row(row: dict[str, Any], *, public_mode: bool, index: int) -> dict[str, Any]:
    label = public_person_name(row, index=index) if public_mode else row.get("display_name")
    return {
        "person_id": row.get("id"),
        "person_label": label or "",
        "privacy_level": row.get("privacy_level"),
        "event_count": int(row.get("event_count") or 0),
        "media_count": int(row.get("media_count") or 0),
    }


def _event_person_row(row: dict[str, Any], *, public_mode: bool, index: int) -> dict[str, Any]:
    label = public_person_name(row, index=index) if public_mode else row.get("display_name")
    row["person_label"] = label or ""
    if public_mode:
        row.pop("display_name", None)
    return row


def _media_person_row(row: dict[str, Any], *, public_mode: bool, index: int) -> dict[str, Any]:
    label = public_person_name(row, index=index) if public_mode else row.get("display_name")
    row["person_label"] = label or ""
    if public_mode:
        row.pop("display_name", None)
    return row


def _rows(connection, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(sql, params or []).fetchall()]


def _count(connection, sql: str, params: list[Any] | None = None) -> int:
    row = connection.execute(sql, params or []).fetchone()
    return int(row[0] if row else 0)


def _source_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["  - none"]
    return [f"  - {row.get('source')}: {row.get('count')}" for row in rows]


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return ""


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
