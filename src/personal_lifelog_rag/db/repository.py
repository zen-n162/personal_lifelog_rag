"""Repository layer over the local SQLite lifelog database."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable

from personal_lifelog_rag.db.schema import TABLE_NAMES, initialize_schema

DEFAULT_DB_PATH = Path("data/db/lifelog.sqlite")
DB_PATH_ENV_VAR = "PERSONAL_LIFELOG_RAG_DB_PATH"


def resolve_db_path(db_path: str | os.PathLike[str] | None = None) -> Path:
    raw_path = db_path if db_path is not None else os.getenv(DB_PATH_ENV_VAR)
    path = Path(raw_path) if raw_path else DEFAULT_DB_PATH
    return path.expanduser()


def connect(db_path: str | os.PathLike[str]) -> sqlite3.Connection:
    path = Path(db_path).expanduser()
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


class LifelogRepository:
    """Persistence boundary for media records, LINE messages, events, and stats."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()

    def initialize(self) -> None:
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)

    def stats(self) -> dict[str, int]:
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            return {
                table_name: self._count_rows(connection, table_name)
                for table_name in TABLE_NAMES
            }

    def add_media_item(
        self,
        *,
        id: str | None = None,
        file_path: str | None = None,
        file_name: str | None = None,
        file_hash: str | None = None,
        media_type: str = "image",
        captured_at: str | None = None,
        fallback_captured_at: str | None = None,
        gps_lat: float | None = None,
        gps_lon: float | None = None,
        camera_model: str | None = None,
        width: int | None = None,
        height: int | None = None,
        thumbnail_path: str | None = None,
        caption: str | None = None,
        ocr_text: str | None = None,
        analysis_json: str | dict[str, Any] | None = None,
        source_path: str | None = None,
        content_hash: str | None = None,
        **_: Any,
    ) -> str:
        """Insert one media item and return its stable id.

        `source_path` and `content_hash` are accepted as legacy aliases from the
        first scaffold so older tests and callers keep working.
        """

        resolved_file_path = file_path or source_path
        if resolved_file_path is None:
            raise ValueError("file_path is required")
        resolved_file_name = file_name or Path(resolved_file_path).name
        resolved_hash = file_hash or content_hash or _stable_id("media_file", resolved_file_path)
        media_id = id or f"media_{resolved_hash}"

        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO media_items (
                    id,
                    file_path,
                    file_name,
                    file_hash,
                    media_type,
                    captured_at,
                    fallback_captured_at,
                    gps_lat,
                    gps_lon,
                    camera_model,
                    width,
                    height,
                    thumbnail_path,
                    caption,
                    ocr_text,
                    analysis_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    media_id,
                    resolved_file_path,
                    resolved_file_name,
                    resolved_hash,
                    media_type,
                    captured_at,
                    fallback_captured_at,
                    gps_lat,
                    gps_lon,
                    camera_model,
                    width,
                    height,
                    thumbnail_path,
                    caption,
                    ocr_text,
                    analysis_json if isinstance(analysis_json, str) else _json_or_none(analysis_json),
                ),
            )
            connection.commit()
            if cursor.rowcount > 0:
                return media_id
            row = connection.execute(
                "SELECT id FROM media_items WHERE file_hash = ?",
                (resolved_hash,),
            ).fetchone()
            return str(row["id"])

    def add_line_message(
        self,
        *,
        id: str | None = None,
        chat_id: str | None = None,
        source_file: str | None = None,
        sent_at: str,
        sender: str | None = None,
        text: str | None = None,
        message_type: str = "text",
        source_path: str | None = None,
        chat_name: str | None = None,
        sender_name: str | None = None,
        message_text: str | None = None,
        **_: Any,
    ) -> str:
        """Insert one LINE message and return its stable id."""

        resolved_source_file = source_file or Path(source_path or "line_export.txt").name
        resolved_sender = sender if sender is not None else sender_name
        resolved_text = text if text is not None else message_text
        resolved_chat_id = chat_id or _stable_id("line_chat", chat_name or source_path or resolved_source_file)
        message_id = id or _line_message_id(
            source_file=resolved_source_file,
            sent_at=sent_at,
            sender=resolved_sender or "",
            text=resolved_text or "",
        )

        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO line_messages (
                    id,
                    chat_id,
                    source_file,
                    sent_at,
                    sender,
                    text,
                    message_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    resolved_chat_id,
                    resolved_source_file,
                    sent_at,
                    resolved_sender,
                    resolved_text,
                    message_type,
                ),
            )
            connection.commit()
            if cursor.rowcount > 0:
                return message_id
            return message_id

    def add_event(
        self,
        *,
        id: str | None = None,
        date: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        title: str,
        summary: str | None = None,
        location_name: str | None = None,
        gps_lat: float | None = None,
        gps_lon: float | None = None,
        participants: list[str] | None = None,
        confidence: float | None = None,
        source: str | None = "generated",
        generation_method: str | None = None,
        is_user_edited: bool | int = False,
        event_type: str | None = None,
        started_at: str | None = None,
        description: str | None = None,
        **_: Any,
    ) -> str:
        event_date = date or (started_at[:10] if started_at else None)
        event_start = start_time or (started_at[11:19] if started_at and len(started_at) >= 19 else None)
        event_summary = summary or description
        event_id = id or _stable_id("event", "|".join([event_date or "", event_start or "", title, event_type or ""]))

        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            connection.execute(
                """
                INSERT OR REPLACE INTO events (
                    id,
                    date,
                    start_time,
                    end_time,
                    title,
                    summary,
                    location_name,
                    gps_lat,
                    gps_lon,
                    participants_json,
                    confidence,
                    source,
                    generation_method,
                    is_user_edited,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    event_id,
                    event_date,
                    event_start,
                    end_time,
                    title,
                    event_summary,
                    location_name,
                    gps_lat,
                    gps_lon,
                    _json_or_none(participants or []),
                    confidence,
                    source,
                    generation_method,
                    int(bool(is_user_edited)),
                ),
            )
            connection.commit()
        return event_id

    def delete_events(
        self,
        *,
        start_date: str,
        end_date: str | None = None,
    ) -> int:
        """Delete events in a date range and return the number removed."""

        resolved_end = end_date or start_date
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            cursor = connection.execute(
                """
                DELETE FROM events
                WHERE substr(date, 1, 10) >= ? AND substr(date, 1, 10) <= ?
                """,
                (start_date, resolved_end),
            )
            connection.commit()
            return max(cursor.rowcount, 0)

    def delete_generated_events(
        self,
        *,
        start_date: str,
        end_date: str | None = None,
        generation_method: str,
        include_legacy_null: bool = True,
    ) -> int:
        """Delete generated events in a range and return the number removed."""

        resolved_end = end_date or start_date
        if include_legacy_null:
            generation_clause = "(generation_method = ? OR generation_method IS NULL)"
        else:
            generation_clause = "generation_method = ?"

        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            cursor = connection.execute(
                f"""
                DELETE FROM events
                WHERE substr(date, 1, 10) >= ?
                  AND substr(date, 1, 10) <= ?
                  AND {generation_clause}
                  AND COALESCE(is_user_edited, 0) = 0
                  AND (source = 'generated' OR source IS NULL)
                """,
                (start_date, resolved_end, generation_method),
            )
            connection.commit()
            return max(cursor.rowcount, 0)

    def count_events(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if start_date is not None:
            clauses.append("substr(date, 1, 10) >= ?")
            params.append(start_date)
        if end_date is not None:
            clauses.append("substr(date, 1, 10) <= ?")
            params.append(end_date)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            row = connection.execute(
                f"SELECT COUNT(*) AS count FROM events {where_sql}",
                params,
            ).fetchone()
            return int(row["count"])

    def list_record_dates(self) -> list[str]:
        """Return sorted local dates that have photos or LINE messages."""

        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            rows = connection.execute(
                """
                SELECT DISTINCT record_date
                FROM (
                    SELECT substr(COALESCE(captured_at, fallback_captured_at), 1, 10) AS record_date
                    FROM media_items
                    WHERE COALESCE(captured_at, fallback_captured_at) IS NOT NULL
                      AND length(COALESCE(captured_at, fallback_captured_at)) >= 10
                    UNION
                    SELECT substr(sent_at, 1, 10) AS record_date
                    FROM line_messages
                    WHERE sent_at IS NOT NULL
                      AND length(sent_at) >= 10
                )
                WHERE record_date IS NOT NULL AND record_date != ''
                ORDER BY record_date ASC
                """
            ).fetchall()
        return [str(row["record_date"]) for row in rows]

    def add_event_evidence(
        self,
        *,
        event_id: str,
        evidence_type: str,
        evidence_id: str | None = None,
        weight: float = 1.0,
        media_item_id: str | None = None,
        line_message_id: str | None = None,
        **_: Any,
    ) -> None:
        resolved_evidence_id = evidence_id or media_item_id or line_message_id
        if resolved_evidence_id is None:
            raise ValueError("evidence_id is required")
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            connection.execute(
                """
                INSERT OR REPLACE INTO event_evidence (
                    event_id,
                    evidence_type,
                    evidence_id,
                    weight
                )
                VALUES (?, ?, ?, ?)
                """,
                (event_id, evidence_type, resolved_evidence_id, weight),
            )
            connection.commit()

    def replace_event_evidence(self, event_id: str, evidence_rows: Iterable[dict[str, Any]]) -> int:
        """Replace evidence links for one event and return the saved count."""

        saved = 0
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            connection.execute("DELETE FROM event_evidence WHERE event_id = ?", (event_id,))
            for row in evidence_rows:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO event_evidence (
                        event_id,
                        evidence_type,
                        evidence_id,
                        weight
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        row["evidence_type"],
                        row["evidence_id"],
                        row.get("weight", 1.0),
                    ),
                )
                saved += 1
            connection.commit()
        return saved

    def list_event_evidence(self, event_id: str | None = None) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if event_id is not None:
            clauses.append("event_id = ?")
            params.append(event_id)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self._fetch_all(
            f"""
            SELECT *
            FROM event_evidence
            {where_sql}
            ORDER BY event_id ASC, evidence_type ASC, evidence_id ASC
            """,
            params,
        )

    def add_line_messages(self, messages: Iterable[Any]) -> int:
        inserted = 0
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            for message in messages:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO line_messages (
                        id,
                        chat_id,
                        source_file,
                        sent_at,
                        sender,
                        text,
                        message_type
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message.message_id,
                        message.chat_id,
                        message.source_file,
                        message.sent_at,
                        message.sender_name,
                        message.message_text,
                        message.message_type,
                    ),
                )
                inserted += max(cursor.rowcount, 0)
            connection.commit()
        return inserted

    def list_line_messages(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        keyword: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if start_date is not None:
            clauses.append("substr(sent_at, 1, 10) >= ?")
            params.append(start_date)
        if end_date is not None:
            clauses.append("substr(sent_at, 1, 10) <= ?")
            params.append(end_date)
        if keyword:
            clauses.append("text LIKE ?")
            params.append(f"%{keyword}%")

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self._fetch_all(
            f"""
            SELECT *
            FROM line_messages
            {where_sql}
            ORDER BY sent_at ASC, id ASC
            LIMIT ?
            """,
            params,
        )
        return [_with_legacy_line_keys(row) for row in rows]

    def upsert_line_call_events(self, events: Iterable[dict[str, Any]]) -> int:
        """Insert or replace structured LINE call events by message id."""

        saved = 0
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            for event in events:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO line_call_events (
                        message_id,
                        chat_id,
                        sent_at,
                        sender,
                        call_status,
                        duration_sec,
                        raw_text_short
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event["message_id"],
                        event.get("chat_id"),
                        event.get("sent_at"),
                        event.get("sender"),
                        event.get("call_status"),
                        event.get("duration_sec"),
                        event.get("raw_text_short"),
                    ),
                )
                saved += 1
            connection.commit()
        return saved

    def delete_line_call_events(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if start_date is not None:
            clauses.append("substr(sent_at, 1, 10) >= ?")
            params.append(start_date)
        if end_date is not None:
            clauses.append("substr(sent_at, 1, 10) <= ?")
            params.append(end_date)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            cursor = connection.execute(f"DELETE FROM line_call_events {where_sql}", params)
            connection.commit()
            return max(cursor.rowcount, 0)

    def list_line_call_events(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        statuses: list[str] | None = None,
        min_duration_sec: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if start_date is not None:
            clauses.append("substr(sent_at, 1, 10) >= ?")
            params.append(start_date)
        if end_date is not None:
            clauses.append("substr(sent_at, 1, 10) <= ?")
            params.append(end_date)
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"call_status IN ({placeholders})")
            params.extend(statuses)
        if min_duration_sec is not None:
            clauses.append("COALESCE(duration_sec, 0) >= ?")
            params.append(min_duration_sec)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        return self._fetch_all(
            f"""
            SELECT *
            FROM line_call_events
            {where_sql}
            ORDER BY sent_at ASC, message_id ASC
            LIMIT ?
            """,
            params,
        )

    def list_media_items(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        timestamp = "COALESCE(captured_at, fallback_captured_at)"
        if start_date is not None:
            clauses.append(f"substr({timestamp}, 1, 10) >= ?")
            params.append(start_date)
        if end_date is not None:
            clauses.append(f"substr({timestamp}, 1, 10) <= ?")
            params.append(end_date)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self._fetch_all(
            f"""
            SELECT *
            FROM media_items
            {where_sql}
            ORDER BY {timestamp} ASC, id ASC
            LIMIT ?
            """,
            params,
        )
        return [_with_legacy_media_keys(row) for row in rows]

    def list_media_items_for_analysis(
        self,
        *,
        limit: int = 100,
        include_analyzed: bool = False,
    ) -> list[dict[str, Any]]:
        """Return local image rows that are candidates for OCR/VLM analysis."""

        clauses = ["media_type = ?"]
        params: list[Any] = ["image"]
        if not include_analyzed:
            clauses.append("(analysis_json IS NULL OR TRIM(analysis_json) = '')")

        params.append(limit)
        rows = self._fetch_all(
            f"""
            SELECT *
            FROM media_items
            WHERE {' AND '.join(clauses)}
            ORDER BY COALESCE(captured_at, fallback_captured_at) ASC, id ASC
            LIMIT ?
            """,
            params,
        )
        return [_with_legacy_media_keys(row) for row in rows]

    def update_media_analysis(
        self,
        media_id: str,
        *,
        caption: str | None = None,
        ocr_text: str | None = None,
        analysis: dict[str, Any] | str | None = None,
    ) -> None:
        """Persist local OCR/VLM-derived fields for one media item."""

        analysis_json = analysis if isinstance(analysis, str) else _json_or_none(analysis)
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            connection.execute(
                """
                UPDATE media_items
                SET
                    caption = COALESCE(?, caption),
                    ocr_text = COALESCE(?, ocr_text),
                    analysis_json = ?
                WHERE id = ?
                """,
                (caption, ocr_text, analysis_json, media_id),
            )
            connection.commit()

    def upsert_media_ocr(
        self,
        *,
        media_id: str,
        ocr_text: str | None = None,
        ocr_text_redacted: str | None = None,
        ocr_engine: str | None = None,
        ocr_languages: str | list[str] | None = None,
        confidence: float | None = None,
        blocks_json: str | list[dict[str, Any]] | None = None,
        status: str = "pending",
        error_message: str | None = None,
        analysis_version: str | None = None,
    ) -> None:
        """Persist local OCR result and mirror searchable text to media_items."""

        languages_text = "+".join(ocr_languages) if isinstance(ocr_languages, list) else ocr_languages
        blocks_text = blocks_json if isinstance(blocks_json, str) else _json_or_none(blocks_json)
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            connection.execute(
                """
                INSERT OR REPLACE INTO media_ocr (
                    media_id,
                    ocr_text,
                    ocr_text_redacted,
                    ocr_engine,
                    ocr_languages,
                    confidence,
                    blocks_json,
                    status,
                    error_message,
                    analyzed_at,
                    analysis_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                """,
                (
                    media_id,
                    ocr_text,
                    ocr_text_redacted,
                    ocr_engine,
                    languages_text,
                    confidence,
                    blocks_text,
                    status,
                    error_message,
                    analysis_version,
                ),
            )
            if status == "success":
                connection.execute(
                    "UPDATE media_items SET ocr_text = ? WHERE id = ?",
                    (ocr_text, media_id),
                )
            connection.commit()

    def get_media_ocr(self, media_id: str) -> dict[str, Any] | None:
        rows = self._fetch_all(
            """
            SELECT
                media_ocr.*,
                media_items.file_name,
                media_items.file_path,
                media_items.captured_at,
                media_items.fallback_captured_at,
                media_items.thumbnail_path,
                media_items.gps_lat,
                media_items.gps_lon
            FROM media_ocr
            LEFT JOIN media_items ON media_items.id = media_ocr.media_id
            WHERE media_ocr.media_id = ?
            LIMIT 1
            """,
            [media_id],
        )
        return rows[0] if rows else None

    def list_media_ocr(
        self,
        *,
        media_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        statuses: list[str] | None = None,
        keyword: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        timestamp = "COALESCE(media_items.captured_at, media_items.fallback_captured_at)"
        if media_id is not None:
            clauses.append("media_ocr.media_id = ?")
            params.append(media_id)
        if start_date is not None:
            clauses.append(f"substr({timestamp}, 1, 10) >= ?")
            params.append(start_date)
        if end_date is not None:
            clauses.append(f"substr({timestamp}, 1, 10) <= ?")
            params.append(end_date)
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"media_ocr.status IN ({placeholders})")
            params.extend(statuses)
        if keyword:
            clauses.append("(COALESCE(media_ocr.ocr_text, '') LIKE ? OR COALESCE(media_ocr.ocr_text_redacted, '') LIKE ?)")
            params.extend([f"%{keyword}%", f"%{keyword}%"])
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        return self._fetch_all(
            f"""
            SELECT
                media_ocr.*,
                media_items.file_name,
                media_items.file_path,
                media_items.captured_at,
                media_items.fallback_captured_at,
                media_items.thumbnail_path,
                media_items.gps_lat,
                media_items.gps_lon
            FROM media_ocr
            LEFT JOIN media_items ON media_items.id = media_ocr.media_id
            {where_sql}
            ORDER BY {timestamp} ASC, media_ocr.media_id ASC
            LIMIT ?
            """,
            params,
        )

    def upsert_media_vlm(
        self,
        *,
        media_id: str,
        caption: str | None = None,
        short_caption: str | None = None,
        scene_tags: list[str] | str | None = None,
        object_tags: list[str] | str | None = None,
        activity_tags: list[str] | str | None = None,
        location_cues: list[str] | str | None = None,
        food_cues: list[str] | str | None = None,
        people_count: int | None = None,
        contains_text_hint: bool | int | None = None,
        safety_flags: list[str] | str | None = None,
        vlm_engine: str | None = None,
        model_name: str | None = None,
        prompt_version: str | None = None,
        confidence: float | None = None,
        status: str = "pending",
        error_message: str | None = None,
        analysis_version: str | None = None,
    ) -> None:
        """Persist local VLM result and mirror caption to media_items."""

        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            connection.execute(
                """
                INSERT OR REPLACE INTO media_vlm (
                    media_id,
                    caption,
                    short_caption,
                    scene_tags_json,
                    object_tags_json,
                    activity_tags_json,
                    location_cues_json,
                    food_cues_json,
                    people_count,
                    contains_text_hint,
                    safety_flags_json,
                    vlm_engine,
                    model_name,
                    prompt_version,
                    confidence,
                    status,
                    error_message,
                    analyzed_at,
                    analysis_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                """,
                (
                    media_id,
                    caption,
                    short_caption,
                    _json_or_none(scene_tags),
                    _json_or_none(object_tags),
                    _json_or_none(activity_tags),
                    _json_or_none(location_cues),
                    _json_or_none(food_cues),
                    people_count,
                    None if contains_text_hint is None else int(bool(contains_text_hint)),
                    _json_or_none(safety_flags),
                    vlm_engine,
                    model_name,
                    prompt_version,
                    confidence,
                    status,
                    error_message,
                    analysis_version,
                ),
            )
            if status == "success":
                connection.execute(
                    """
                    UPDATE media_items
                    SET caption = COALESCE(?, caption),
                        analysis_json = COALESCE(?, analysis_json)
                    WHERE id = ?
                    """,
                    (
                        short_caption or caption,
                        _json_or_none(
                            {
                                "caption": caption,
                                "short_caption": short_caption,
                                "scene_tags": scene_tags if isinstance(scene_tags, list) else [],
                                "object_tags": object_tags if isinstance(object_tags, list) else [],
                                "activity_tags": activity_tags if isinstance(activity_tags, list) else [],
                                "location_cues": location_cues if isinstance(location_cues, list) else [],
                                "food_cues": food_cues if isinstance(food_cues, list) else [],
                                "safety_flags": safety_flags if isinstance(safety_flags, list) else [],
                            }
                        ),
                        media_id,
                    ),
                )
            connection.commit()

    def get_media_vlm(self, media_id: str) -> dict[str, Any] | None:
        rows = self._fetch_all(
            """
            SELECT
                media_vlm.*,
                media_items.file_name,
                media_items.file_path,
                media_items.captured_at,
                media_items.fallback_captured_at,
                media_items.thumbnail_path,
                media_items.gps_lat,
                media_items.gps_lon,
                media_ocr.ocr_text,
                media_ocr.ocr_text_redacted,
                media_ocr.status AS ocr_status
            FROM media_vlm
            LEFT JOIN media_items ON media_items.id = media_vlm.media_id
            LEFT JOIN media_ocr ON media_ocr.media_id = media_vlm.media_id
            WHERE media_vlm.media_id = ?
            LIMIT 1
            """,
            [media_id],
        )
        return rows[0] if rows else None

    def list_media_vlm(
        self,
        *,
        media_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        statuses: list[str] | None = None,
        keyword: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        timestamp = "COALESCE(media_items.captured_at, media_items.fallback_captured_at)"
        if media_id is not None:
            clauses.append("media_vlm.media_id = ?")
            params.append(media_id)
        if start_date is not None:
            clauses.append(f"substr({timestamp}, 1, 10) >= ?")
            params.append(start_date)
        if end_date is not None:
            clauses.append(f"substr({timestamp}, 1, 10) <= ?")
            params.append(end_date)
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"media_vlm.status IN ({placeholders})")
            params.extend(statuses)
        if keyword:
            clauses.append(
                """
                (
                    COALESCE(media_vlm.caption, '') LIKE ?
                    OR COALESCE(media_vlm.short_caption, '') LIKE ?
                    OR COALESCE(media_vlm.scene_tags_json, '') LIKE ?
                    OR COALESCE(media_vlm.object_tags_json, '') LIKE ?
                    OR COALESCE(media_vlm.activity_tags_json, '') LIKE ?
                    OR COALESCE(media_vlm.location_cues_json, '') LIKE ?
                    OR COALESCE(media_vlm.food_cues_json, '') LIKE ?
                    OR COALESCE(media_ocr.ocr_text, '') LIKE ?
                    OR COALESCE(media_items.file_name, '') LIKE ?
                )
                """
            )
            params.extend([f"%{keyword}%"] * 9)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        return self._fetch_all(
            f"""
            SELECT
                media_vlm.*,
                media_items.file_name,
                media_items.file_path,
                media_items.captured_at,
                media_items.fallback_captured_at,
                media_items.thumbnail_path,
                media_items.gps_lat,
                media_items.gps_lon,
                media_ocr.ocr_text,
                media_ocr.ocr_text_redacted,
                media_ocr.status AS ocr_status
            FROM media_vlm
            LEFT JOIN media_items ON media_items.id = media_vlm.media_id
            LEFT JOIN media_ocr ON media_ocr.media_id = media_vlm.media_id
            {where_sql}
            ORDER BY {timestamp} ASC, media_vlm.media_id ASC
            LIMIT ?
            """,
            params,
        )

    def list_events(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        event_id: str | None = None,
        keyword: str | None = None,
        limit: int = 100,
        include_hidden: bool = False,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if event_id is not None:
            clauses.append("events.id = ?")
            params.append(event_id)
        if start_date is not None:
            clauses.append("substr(events.date, 1, 10) >= ?")
            params.append(start_date)
        if end_date is not None:
            clauses.append("substr(events.date, 1, 10) <= ?")
            params.append(end_date)
        if keyword:
            clauses.append(
                """
                (
                    COALESCE(events.title, '') LIKE ?
                    OR COALESCE(events.summary, '') LIKE ?
                    OR COALESCE(events.location_name, '') LIKE ?
                    OR COALESCE(event_overrides.title_override, '') LIKE ?
                    OR COALESCE(event_overrides.summary_override, '') LIKE ?
                    OR COALESCE(event_overrides.location_name_override, '') LIKE ?
                    OR COALESCE(event_overrides.tags_json, '') LIKE ?
                )
                """
            )
            params.extend([f"%{keyword}%"] * 7)
        if not include_hidden:
            clauses.append("COALESCE(event_overrides.is_hidden, 0) = 0")

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self._fetch_all(
            f"""
            SELECT
                events.*,
                event_overrides.title_override,
                event_overrides.summary_override,
                event_overrides.location_name_override,
                event_overrides.tags_json,
                COALESCE(event_overrides.is_verified, 0) AS is_verified,
                COALESCE(event_overrides.is_hidden, 0) AS is_hidden,
                COALESCE(event_overrides.is_pinned, 0) AS is_pinned,
                event_overrides.updated_at AS override_updated_at,
                (
                    SELECT COUNT(*)
                    FROM event_evidence
                    WHERE event_evidence.event_id = events.id
                ) AS event_evidence_count
                ,
                (
                    SELECT COUNT(*)
                    FROM event_evidence
                    WHERE event_evidence.event_id = events.id
                      AND event_evidence.evidence_type = 'line'
                ) AS line_evidence_count,
                (
                    SELECT COUNT(*)
                    FROM event_evidence
                    WHERE event_evidence.event_id = events.id
                      AND event_evidence.evidence_type = 'photo'
                ) AS photo_evidence_count
            FROM events
            LEFT JOIN event_overrides ON event_overrides.event_id = events.id
            {where_sql}
            ORDER BY
                COALESCE(event_overrides.is_pinned, 0) DESC,
                events.date ASC,
                events.start_time ASC,
                events.id ASC
            LIMIT ?
            """,
            params,
        )
        return [_with_legacy_event_keys(row) for row in rows]

    def get_event(self, event_id: str, *, include_hidden: bool = True) -> dict[str, Any] | None:
        rows = self.list_events(event_id=event_id, include_hidden=include_hidden, limit=1)
        return rows[0] if rows else None

    def get_event_override(self, event_id: str) -> dict[str, Any] | None:
        rows = self._fetch_all(
            """
            SELECT *
            FROM event_overrides
            WHERE event_id = ?
            LIMIT 1
            """,
            [event_id],
        )
        return rows[0] if rows else None

    def upsert_event_override(
        self,
        event_id: str,
        *,
        title_override: str | None = None,
        summary_override: str | None = None,
        location_name_override: str | None = None,
        tags: list[str] | None = None,
        is_verified: bool | None = None,
        is_hidden: bool | None = None,
        is_pinned: bool | None = None,
    ) -> dict[str, Any]:
        """Create or update a user override without modifying generated events."""

        existing = self.get_event_override(event_id) or {}
        resolved_title = _blank_to_none(title_override) if title_override is not None else existing.get("title_override")
        resolved_summary = _blank_to_none(summary_override) if summary_override is not None else existing.get("summary_override")
        resolved_location = (
            _blank_to_none(location_name_override)
            if location_name_override is not None
            else existing.get("location_name_override")
        )
        resolved_tags = _json_or_none(tags) if tags is not None else existing.get("tags_json")
        resolved_verified = _resolve_bool(is_verified, existing.get("is_verified"))
        resolved_hidden = _resolve_bool(is_hidden, existing.get("is_hidden"))
        resolved_pinned = _resolve_bool(is_pinned, existing.get("is_pinned"))
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            connection.execute(
                """
                INSERT OR REPLACE INTO event_overrides (
                    event_id,
                    title_override,
                    summary_override,
                    location_name_override,
                    tags_json,
                    is_verified,
                    is_hidden,
                    is_pinned,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    event_id,
                    resolved_title,
                    resolved_summary,
                    resolved_location,
                    resolved_tags,
                    resolved_verified,
                    resolved_hidden,
                    resolved_pinned,
                ),
            )
            connection.commit()
        return self.get_event_override(event_id) or {}

    def delete_event_override(self, event_id: str) -> int:
        """Remove one manual override row while leaving the generated event intact."""

        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            cursor = connection.execute(
                "DELETE FROM event_overrides WHERE event_id = ?",
                (event_id,),
            )
            connection.commit()
            return max(cursor.rowcount, 0)

    def update_event_location_name(self, event_id: str, *, location_name: str) -> bool:
        """Update one generated event's safe display location.

        User-edited events are intentionally left untouched so future manual UI
        overrides will not be overwritten by automated place assignment.
        """

        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            cursor = connection.execute(
                """
                UPDATE events
                SET location_name = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                  AND COALESCE(is_user_edited, 0) = 0
                """,
                (location_name, event_id),
            )
            connection.commit()
            return cursor.rowcount > 0

    def event_has_location_override(self, event_id: str) -> bool:
        """Return whether an optional future override table protects location_name."""

        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            row = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                ("event_overrides",),
            ).fetchone()
            if row is None:
                return False
            columns = {column[1] for column in connection.execute("PRAGMA table_info(event_overrides)")}
            if not {"event_id", "location_name_override"}.issubset(columns):
                return False
            row = connection.execute(
                """
                SELECT 1
                FROM event_overrides
                WHERE event_id = ?
                  AND location_name_override IS NOT NULL
                  AND TRIM(location_name_override) != ''
                LIMIT 1
                """,
                (event_id,),
            ).fetchone()
            return row is not None

    def search_text_records(
        self,
        *,
        terms: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 10_000,
        include_hidden: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """Search local text-bearing records with SQLite LIKE clauses."""

        clean_terms = [term for term in terms if term.strip()]
        if not clean_terms:
            return {"line_messages": [], "events": [], "media_items": [], "media_ocr": [], "media_vlm": []}

        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            line_rows = connection.execute(
                *_search_query(
                    table_name="line_messages",
                    columns=["text"],
                    terms=clean_terms,
                    timestamp_expr="sent_at",
                    start_date=start_date,
                    end_date=end_date,
                    order_sql="sent_at ASC, id ASC",
                    limit=limit,
                )
            ).fetchall()
            event_rows = connection.execute(
                *_search_events_query(
                    terms=clean_terms,
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit,
                    include_hidden=include_hidden,
                )
            ).fetchall()
            media_rows = connection.execute(
                *_search_query(
                    table_name="media_items",
                    columns=["file_name", "caption", "ocr_text"],
                    terms=clean_terms,
                    timestamp_expr="COALESCE(captured_at, fallback_captured_at)",
                    start_date=start_date,
                    end_date=end_date,
                    order_sql="COALESCE(captured_at, fallback_captured_at) ASC, id ASC",
                    limit=limit,
                )
            ).fetchall()
            ocr_rows = connection.execute(
                *_search_media_ocr_query(
                    terms=clean_terms,
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit,
                )
            ).fetchall()
            vlm_rows = connection.execute(
                *_search_media_vlm_query(
                    terms=clean_terms,
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit,
                )
            ).fetchall()

        return {
            "line_messages": [_with_legacy_line_keys(dict(row)) for row in line_rows],
            "events": [_with_legacy_event_keys(dict(row)) for row in event_rows],
            "media_items": [_with_legacy_media_keys(dict(row)) for row in media_rows],
            "media_ocr": [dict(row) for row in ocr_rows],
            "media_vlm": [dict(row) for row in vlm_rows],
        }

    def embedding_sources(self) -> list[dict[str, Any]]:
        """Return local texts that should have embeddings."""

        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            line_rows = connection.execute(
                """
                SELECT id, text
                FROM line_messages
                WHERE text IS NOT NULL AND TRIM(text) != ''
                ORDER BY sent_at ASC, id ASC
                """
            ).fetchall()
            media_rows = connection.execute(
                """
                SELECT
                    id,
                    file_name,
                    camera_model,
                    caption,
                    ocr_text
                FROM media_items
                ORDER BY COALESCE(captured_at, fallback_captured_at) ASC, id ASC
                """
            ).fetchall()

        sources = [
            {
                "source_type": "line_message",
                "source_id": row["id"],
                "text": row["text"],
            }
            for row in line_rows
        ]

        for row in media_rows:
            text = _media_embedding_text(dict(row))
            if text:
                sources.append(
                    {
                        "source_type": "media_item",
                        "source_id": row["id"],
                        "text": text,
                    }
                )
        return sources

    def upsert_embeddings(self, rows: Iterable[dict[str, Any]]) -> int:
        upserted = 0
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            for row in rows:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO text_embeddings (
                        source_type,
                        source_id,
                        text,
                        embedding_json,
                        embedding_model,
                        embedding_dim,
                        content_hash
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["source_type"],
                        row["source_id"],
                        row["text"],
                        row["embedding_json"],
                        row["embedding_model"],
                        row["embedding_dim"],
                        row["content_hash"],
                    ),
                )
                upserted += 1
            connection.commit()
        return upserted

    def list_embeddings(self, *, model_name: str | None = None) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if model_name:
            clauses.append("embedding_model = ?")
            params.append(model_name)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self._fetch_all(
            f"""
            SELECT *
            FROM text_embeddings
            {where_sql}
            ORDER BY created_at ASC
            """,
            params,
        )

    def get_embedding_record(self, source_type: str, source_id: str) -> dict[str, Any] | None:
        if source_type == "line_message":
            rows = self._fetch_all(
                "SELECT * FROM line_messages WHERE id = ? LIMIT 1",
                [source_id],
            )
            return _with_legacy_line_keys(rows[0]) if rows else None
        if source_type == "media_item":
            rows = self._fetch_all(
                "SELECT * FROM media_items WHERE id = ? LIMIT 1",
                [source_id],
            )
            return _with_legacy_media_keys(rows[0]) if rows else None
        return None

    def _fetch_all(self, query: str, params: list[Any]) -> list[dict[str, Any]]:
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            rows = connection.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    @staticmethod
    def _count_rows(connection: sqlite3.Connection, table_name: str) -> int:
        cursor = connection.execute(f"SELECT COUNT(*) AS count FROM {table_name}")
        row = cursor.fetchone()
        return int(row["count"])


def _with_legacy_line_keys(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row.setdefault("sender_name", row.get("sender"))
    row.setdefault("message_text", row.get("text"))
    row.setdefault("metadata_json", "{}")
    return row


def _with_legacy_media_keys(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row.setdefault("source_path", row.get("file_path"))
    row.setdefault("content_hash", row.get("file_hash"))
    return row


def _with_legacy_event_keys(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["original_title"] = row.get("title")
    row["original_summary"] = row.get("summary")
    row["original_location_name"] = row.get("location_name")
    if row.get("title_override"):
        row["title"] = row["title_override"]
    if row.get("summary_override"):
        row["summary"] = row["summary_override"]
    if row.get("location_name_override"):
        row["location_name"] = row["location_name_override"]
    row["is_verified"] = int(row.get("is_verified") or 0)
    row["is_hidden"] = int(row.get("is_hidden") or 0)
    row["is_pinned"] = int(row.get("is_pinned") or 0)
    row["is_user_edited"] = int(row.get("is_user_edited") or 0)
    if row.get("date"):
        row.setdefault("started_at", f"{row['date']}T{row.get('start_time') or '00:00:00'}")
    row.setdefault("description", row.get("summary"))
    return row


def _media_embedding_text(row: dict[str, Any]) -> str:
    parts = [
        row.get("caption") or "",
        row.get("ocr_text") or "",
        row.get("file_name") or "",
        row.get("camera_model") or "",
    ]
    return " ".join(part.strip() for part in parts if part and part.strip())


def _line_message_id(*, source_file: str, sent_at: str, sender: str, text: str) -> str:
    return _stable_id("line_msg", "\u001f".join([source_file, sent_at, sender, text]))


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}"


def _json_or_none(value: Any | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _resolve_bool(value: bool | None, existing: Any) -> int:
    if value is None:
        return int(bool(existing)) if existing is not None else 0
    return int(bool(value))


def _search_query(
    *,
    table_name: str,
    columns: list[str],
    terms: list[str],
    timestamp_expr: str,
    start_date: str | None,
    end_date: str | None,
    order_sql: str,
    limit: int,
    select_sql: str = "*",
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    term_clauses: list[str] = []
    for column in columns:
        for term in terms:
            term_clauses.append(f"COALESCE({column}, '') LIKE ?")
            params.append(f"%{term}%")
    clauses.append("(" + " OR ".join(term_clauses) + ")")
    if start_date is not None:
        clauses.append(f"substr({timestamp_expr}, 1, 10) >= ?")
        params.append(start_date)
    if end_date is not None:
        clauses.append(f"substr({timestamp_expr}, 1, 10) <= ?")
        params.append(end_date)
    params.append(limit)
    query = f"""
        SELECT {select_sql}
        FROM {table_name}
        WHERE {' AND '.join(clauses)}
        ORDER BY {order_sql}
        LIMIT ?
    """
    return query, params


def _search_events_query(
    *,
    terms: list[str],
    start_date: str | None,
    end_date: str | None,
    limit: int,
    include_hidden: bool,
) -> tuple[str, list[Any]]:
    """Build an event search query that includes manual overrides and tags."""

    searchable_columns = [
        "events.title",
        "events.summary",
        "events.location_name",
        "event_overrides.title_override",
        "event_overrides.summary_override",
        "event_overrides.location_name_override",
        "event_overrides.tags_json",
    ]
    clauses: list[str] = []
    params: list[Any] = []
    term_clauses: list[str] = []
    for column in searchable_columns:
        for term in terms:
            term_clauses.append(f"COALESCE({column}, '') LIKE ?")
            params.append(f"%{term}%")
    clauses.append("(" + " OR ".join(term_clauses) + ")")
    if start_date is not None:
        clauses.append("substr(events.date, 1, 10) >= ?")
        params.append(start_date)
    if end_date is not None:
        clauses.append("substr(events.date, 1, 10) <= ?")
        params.append(end_date)
    if not include_hidden:
        clauses.append("COALESCE(event_overrides.is_hidden, 0) = 0")
    params.append(limit)
    query = f"""
        SELECT
            events.*,
            event_overrides.title_override,
            event_overrides.summary_override,
            event_overrides.location_name_override,
            event_overrides.tags_json,
            COALESCE(event_overrides.is_verified, 0) AS is_verified,
            COALESCE(event_overrides.is_hidden, 0) AS is_hidden,
            COALESCE(event_overrides.is_pinned, 0) AS is_pinned,
            event_overrides.updated_at AS override_updated_at,
            (
                SELECT COUNT(*)
                FROM event_evidence
                WHERE event_evidence.event_id = events.id
            ) AS event_evidence_count,
            (
                SELECT COUNT(*)
                FROM event_evidence
                WHERE event_evidence.event_id = events.id
                  AND event_evidence.evidence_type = 'line'
            ) AS line_evidence_count,
            (
                SELECT COUNT(*)
                FROM event_evidence
                WHERE event_evidence.event_id = events.id
                  AND event_evidence.evidence_type = 'photo'
            ) AS photo_evidence_count
        FROM events
        LEFT JOIN event_overrides ON event_overrides.event_id = events.id
        WHERE {' AND '.join(clauses)}
        ORDER BY
            COALESCE(event_overrides.is_pinned, 0) DESC,
            events.date ASC,
            events.start_time ASC,
            events.id ASC
        LIMIT ?
    """
    return query, params


def _search_media_ocr_query(
    *,
    terms: list[str],
    start_date: str | None,
    end_date: str | None,
    limit: int,
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    term_clauses: list[str] = []
    for column in ("media_ocr.ocr_text", "media_ocr.ocr_text_redacted"):
        for term in terms:
            term_clauses.append(f"COALESCE({column}, '') LIKE ?")
            params.append(f"%{term}%")
    clauses.append("(" + " OR ".join(term_clauses) + ")")
    timestamp = "COALESCE(media_items.captured_at, media_items.fallback_captured_at)"
    if start_date is not None:
        clauses.append(f"substr({timestamp}, 1, 10) >= ?")
        params.append(start_date)
    if end_date is not None:
        clauses.append(f"substr({timestamp}, 1, 10) <= ?")
        params.append(end_date)
    params.append(limit)
    query = f"""
        SELECT
            media_ocr.*,
            media_items.id AS media_item_id,
            media_items.file_name,
            media_items.file_path,
            media_items.captured_at,
            media_items.fallback_captured_at,
            media_items.thumbnail_path,
            media_items.gps_lat,
            media_items.gps_lon
        FROM media_ocr
        LEFT JOIN media_items ON media_items.id = media_ocr.media_id
        WHERE {' AND '.join(clauses)}
        ORDER BY {timestamp} ASC, media_ocr.media_id ASC
        LIMIT ?
    """
    return query, params


def _search_media_vlm_query(
    *,
    terms: list[str],
    start_date: str | None,
    end_date: str | None,
    limit: int,
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    term_clauses: list[str] = []
    for column in (
        "media_vlm.caption",
        "media_vlm.short_caption",
        "media_vlm.scene_tags_json",
        "media_vlm.object_tags_json",
        "media_vlm.activity_tags_json",
        "media_vlm.location_cues_json",
        "media_vlm.food_cues_json",
        "media_ocr.ocr_text",
        "media_items.file_name",
    ):
        for term in terms:
            term_clauses.append(f"COALESCE({column}, '') LIKE ?")
            params.append(f"%{term}%")
    clauses.append("(" + " OR ".join(term_clauses) + ")")
    timestamp = "COALESCE(media_items.captured_at, media_items.fallback_captured_at)"
    if start_date is not None:
        clauses.append(f"substr({timestamp}, 1, 10) >= ?")
        params.append(start_date)
    if end_date is not None:
        clauses.append(f"substr({timestamp}, 1, 10) <= ?")
        params.append(end_date)
    params.append(limit)
    query = f"""
        SELECT
            media_vlm.*,
            media_items.id AS media_item_id,
            media_items.file_name,
            media_items.file_path,
            media_items.captured_at,
            media_items.fallback_captured_at,
            media_items.thumbnail_path,
            media_items.gps_lat,
            media_items.gps_lon,
            media_ocr.ocr_text,
            media_ocr.ocr_text_redacted,
            media_ocr.status AS ocr_status
        FROM media_vlm
        LEFT JOIN media_items ON media_items.id = media_vlm.media_id
        LEFT JOIN media_ocr ON media_ocr.media_id = media_vlm.media_id
        WHERE {' AND '.join(clauses)}
        ORDER BY {timestamp} ASC, media_vlm.media_id ASC
        LIMIT ?
    """
    return query, params
