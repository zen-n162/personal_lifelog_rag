"""Manual person labels for private face cluster review.

This module never infers names from faces. It only stores labels explicitly
entered by the user and keeps them out of normal QA/search/report flows.
"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import json
from typing import Any
import uuid

from personal_lifelog_rag.core.privacy import redact_text
from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.db.schema import initialize_schema


VALID_PERSON_PRIVACY_LEVELS = {"private", "public_alias", "public_hidden"}
VALID_PERSON_ALIAS_SOURCES = {"manual", "line_speaker", "nickname"}


def create_person(
    repository: LifelogRepository,
    *,
    name: str,
    public_name: str | None = None,
    privacy_level: str = "private",
    notes: str | None = None,
) -> dict[str, Any]:
    _validate_privacy_level(privacy_level)
    display_name = name.strip()
    if not display_name:
        raise ValueError("person display name is required")
    now = _now()
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        existing = connection.execute(
            """
            SELECT id
            FROM persons
            WHERE display_name = ?
              AND deleted_at IS NULL
              AND COALESCE(hidden, 0) = 0
            ORDER BY created_at ASC, id ASC
            LIMIT 1
            """,
            (display_name,),
        ).fetchone()
        if existing is not None:
            return get_person(repository, str(existing["id"])) or {}
        person_id = f"person_{uuid.uuid4().hex[:16]}"
        connection.execute(
            """
            INSERT INTO persons (
                id, display_name, public_name, aliases_json, privacy_level,
                notes, manual_verified, created_at, updated_at
            )
            VALUES (?, ?, ?, '[]', ?, ?, 1, ?, ?)
            """,
            (person_id, display_name, _clean_optional(public_name), privacy_level, notes, now, now),
        )
        connection.commit()
    return get_person(repository, person_id) or {}


def list_persons(repository: LifelogRepository, *, limit: int = 100, public_mode: bool = False, include_deleted: bool = False) -> list[dict[str, Any]]:
    where = "" if include_deleted else "WHERE persons.deleted_at IS NULL AND COALESCE(persons.hidden, 0) = 0"
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        rows = connection.execute(
            f"""
            SELECT persons.*,
                   COUNT(DISTINCT person_face_clusters.face_cluster_id) AS linked_clusters_count,
                   COUNT(DISTINCT person_aliases.id) AS alias_count
            FROM persons
            LEFT JOIN person_face_clusters ON person_face_clusters.person_id = persons.id
            LEFT JOIN person_aliases ON person_aliases.person_id = persons.id
            {where}
            GROUP BY persons.id
            ORDER BY persons.created_at ASC, persons.id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_person_row(dict(row), public_mode=public_mode, index=i + 1) for i, row in enumerate(rows)]


def get_person(repository: LifelogRepository, person_id: str, *, public_mode: bool = False) -> dict[str, Any] | None:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        row = connection.execute(
            """
            SELECT persons.*,
                   COUNT(DISTINCT person_face_clusters.face_cluster_id) AS linked_clusters_count,
                   COUNT(DISTINCT person_aliases.id) AS alias_count
            FROM persons
            LEFT JOIN person_face_clusters ON person_face_clusters.person_id = persons.id
            LEFT JOIN person_aliases ON person_aliases.person_id = persons.id
            WHERE persons.id = ?
            GROUP BY persons.id
            """,
            (person_id,),
        ).fetchone()
        if row is None:
            return None
        result = _person_row(dict(row), public_mode=public_mode, index=1)
        result["aliases"] = [
            dict(alias_row)
            for alias_row in connection.execute(
                "SELECT id, alias, source, created_at FROM person_aliases WHERE person_id = ? ORDER BY created_at ASC, id ASC",
                (person_id,),
            ).fetchall()
        ]
        result["face_clusters"] = [
            dict(cluster_row)
            for cluster_row in connection.execute(
                """
                SELECT face_clusters.id, face_clusters.cluster_label, face_clusters.face_count,
                       face_clusters.status, face_clusters.review_status,
                       person_face_clusters.verified_at, person_face_clusters.source
                FROM person_face_clusters
                JOIN face_clusters ON face_clusters.id = person_face_clusters.face_cluster_id
                WHERE person_face_clusters.person_id = ?
                ORDER BY face_clusters.first_seen_at ASC, face_clusters.id ASC
                """,
                (person_id,),
            ).fetchall()
        ]
        return result


def update_person(
    repository: LifelogRepository,
    *,
    person_id: str,
    name: str | None = None,
    public_name: str | None = None,
    privacy_level: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    if privacy_level is not None:
        _validate_privacy_level(privacy_level)
    assignments: list[str] = []
    params: list[Any] = []
    if name is not None:
        display_name = name.strip()
        if not display_name:
            raise ValueError("person display name cannot be empty")
        assignments.append("display_name = ?")
        params.append(display_name)
    if public_name is not None:
        assignments.append("public_name = ?")
        params.append(_clean_optional(public_name))
    if privacy_level is not None:
        assignments.append("privacy_level = ?")
        params.append(privacy_level)
    if notes is not None:
        assignments.append("notes = ?")
        params.append(notes)
    if not assignments:
        existing = get_person(repository, person_id)
        if existing is None:
            raise ValueError(f"person not found: {person_id}")
        return existing
    assignments.append("updated_at = ?")
    params.append(_now())
    params.append(person_id)
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        cursor = connection.execute(f"UPDATE persons SET {', '.join(assignments)} WHERE id = ?", params)
        if cursor.rowcount == 0:
            raise ValueError(f"person not found: {person_id}")
        connection.commit()
    return get_person(repository, person_id) or {}


def add_person_alias(
    repository: LifelogRepository,
    *,
    person_id: str,
    alias: str,
    source: str = "manual",
) -> dict[str, Any]:
    if source not in VALID_PERSON_ALIAS_SOURCES:
        raise ValueError(f"invalid person alias source: {source}")
    alias_text = alias.strip()
    if not alias_text:
        raise ValueError("alias is required")
    alias_id = f"person_alias_{uuid.uuid4().hex[:16]}"
    now = _now()
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        if connection.execute("SELECT 1 FROM persons WHERE id = ?", (person_id,)).fetchone() is None:
            raise ValueError(f"person not found: {person_id}")
        connection.execute(
            """
            INSERT INTO person_aliases (id, person_id, alias, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (alias_id, person_id, alias_text, source, now, now),
        )
        aliases = [
            row["alias"]
            for row in connection.execute(
                "SELECT alias FROM person_aliases WHERE person_id = ? ORDER BY created_at ASC, id ASC",
                (person_id,),
            ).fetchall()
        ]
        connection.execute(
            "UPDATE persons SET aliases_json = ?, updated_at = ? WHERE id = ?",
            (json.dumps(aliases, ensure_ascii=False), now, person_id),
        )
        connection.commit()
    return get_person(repository, person_id) or {}


def link_person_face_cluster(
    repository: LifelogRepository,
    *,
    person_id: str,
    cluster_id: str,
    yes: bool = False,
) -> dict[str, Any]:
    if not yes:
        raise ValueError("linking a face cluster to a person requires --yes")
    now = _now()
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        if connection.execute("SELECT 1 FROM persons WHERE id = ?", (person_id,)).fetchone() is None:
            raise ValueError(f"person not found: {person_id}")
        if connection.execute("SELECT 1 FROM face_clusters WHERE id = ?", (cluster_id,)).fetchone() is None:
            raise ValueError(f"face cluster not found: {cluster_id}")
        connection.execute(
            """
            INSERT OR REPLACE INTO person_face_clusters (
                person_id, face_cluster_id, verified_by_user, verified_at,
                source, confidence, created_at, updated_at
            )
            VALUES (?, ?, 1, ?, 'manual', 1.0, ?, ?)
            """,
            (person_id, cluster_id, now, now, now),
        )
        connection.execute(
            """
            UPDATE face_clusters
            SET status = CASE WHEN status = 'unreviewed' THEN 'accepted' ELSE status END,
                review_status = 'reviewed',
                updated_at = ?
            WHERE id = ?
            """,
            (now, cluster_id),
        )
        connection.commit()
    return {"person": get_person(repository, person_id), "cluster_id": cluster_id}


def unlink_person_face_cluster(
    repository: LifelogRepository,
    *,
    person_id: str,
    cluster_id: str,
    yes: bool = False,
) -> dict[str, Any]:
    if not yes:
        raise ValueError("unlinking a face cluster from a person requires --yes")
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        cursor = connection.execute(
            "DELETE FROM person_face_clusters WHERE person_id = ? AND face_cluster_id = ?",
            (person_id, cluster_id),
        )
        connection.commit()
    return {"person_id": person_id, "cluster_id": cluster_id, "deleted": int(cursor.rowcount)}


def anonymize_preview(repository: LifelogRepository) -> list[dict[str, Any]]:
    rows = list_persons(repository, limit=10_000, public_mode=False)
    return [
        {
            "person_id": row["id"],
            "private_display_name": row["display_name"],
            "public_name": public_person_name(row, index=i + 1),
            "privacy_level": row["privacy_level"],
            "linked_clusters_count": row.get("linked_clusters_count", 0),
        }
        for i, row in enumerate(rows)
    ]


def persons_for_media(repository: LifelogRepository, media_id: str, *, public_mode: bool = False) -> list[dict[str, Any]]:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        rows = connection.execute(
            """
            SELECT DISTINCT persons.*, face_clusters.id AS face_cluster_id,
                            face_clusters.status AS face_cluster_status,
                            face_clusters.review_status AS face_cluster_review_status
            FROM persons
            JOIN person_face_clusters ON person_face_clusters.person_id = persons.id
            JOIN face_clusters ON face_clusters.id = person_face_clusters.face_cluster_id
            JOIN face_cluster_members ON face_cluster_members.cluster_id = face_clusters.id
            JOIN face_detections ON face_detections.id = face_cluster_members.face_id
            WHERE face_detections.media_id = ?
              AND person_face_clusters.verified_by_user = 1
              AND face_clusters.status = 'accepted'
            ORDER BY persons.display_name ASC
            """,
            (media_id,),
        ).fetchall()
        return [_person_row(dict(row), public_mode=public_mode, index=i + 1) for i, row in enumerate(rows)]


def persons_for_event(repository: LifelogRepository, event_id: str, *, public_mode: bool = False) -> list[dict[str, Any]]:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        rows = connection.execute(
            """
            SELECT DISTINCT persons.*, face_clusters.id AS face_cluster_id
            FROM events
            JOIN event_evidence ON event_evidence.event_id = events.id
            JOIN face_detections ON face_detections.media_id = event_evidence.evidence_id
            JOIN face_cluster_members ON face_cluster_members.face_id = face_detections.id
            JOIN face_clusters ON face_clusters.id = face_cluster_members.cluster_id
            JOIN person_face_clusters ON person_face_clusters.face_cluster_id = face_clusters.id
            JOIN persons ON persons.id = person_face_clusters.person_id
            WHERE events.id = ?
              AND event_evidence.evidence_type IN ('photo', 'image', 'media')
              AND person_face_clusters.verified_by_user = 1
              AND face_clusters.status = 'accepted'
            ORDER BY persons.display_name ASC
            """,
            (event_id,),
        ).fetchall()
        return [_person_row(dict(row), public_mode=public_mode, index=i + 1) for i, row in enumerate(rows)]


def public_person_name(row: dict[str, Any], *, index: int = 1) -> str:
    privacy_level = row.get("privacy_level") or "private"
    if privacy_level == "public_hidden":
        return ""
    if row.get("public_name"):
        return str(row["public_name"])
    if privacy_level == "public_alias":
        return f"人物{index}"
    return ""


def format_persons(rows: list[dict[str, Any]]) -> str:
    lines = ["Persons"]
    if not rows:
        lines.append("- none")
        return "\n".join(lines)
    for row in rows:
        lines.append(
            f"- {row['id']} name={row.get('display_name') or ''} "
            f"public={row.get('public_name') or ''} privacy={row.get('privacy_level') or ''} "
            f"clusters={row.get('linked_clusters_count', 0)} aliases={row.get('alias_count', 0)}"
        )
    return "\n".join(lines)


def format_person_detail(row: dict[str, Any] | None) -> str:
    if row is None:
        return "person not found"
    lines = [
        f"person_id: {row.get('id')}",
        f"display_name: {row.get('display_name')}",
        f"public_name: {row.get('public_name') or ''}",
        f"privacy_level: {row.get('privacy_level')}",
        f"manual_verified: {row.get('manual_verified')}",
        f"linked_clusters_count: {row.get('linked_clusters_count', 0)}",
        "aliases:",
    ]
    aliases = row.get("aliases") or []
    if aliases:
        lines.extend(f"- {alias.get('alias')} ({alias.get('source')})" for alias in aliases)
    else:
        lines.append("- none")
    lines.append("face_clusters:")
    clusters = row.get("face_clusters") or []
    if clusters:
        lines.extend(f"- {cluster.get('id')} {cluster.get('cluster_label')}" for cluster in clusters)
    else:
        lines.append("- none")
    return "\n".join(lines)


def format_anonymize_preview(rows: list[dict[str, Any]]) -> str:
    lines = ["Person anonymize preview"]
    if not rows:
        lines.append("- none")
        return "\n".join(lines)
    for row in rows:
        lines.append(
            f"- {row['person_id']}: private={redact_text(row.get('private_display_name'), max_chars=40)} "
            f"public={row.get('public_name') or '(hidden)'} privacy={row.get('privacy_level')}"
        )
    return "\n".join(lines)


def _person_row(row: dict[str, Any], *, public_mode: bool, index: int) -> dict[str, Any]:
    row = dict(row)
    row["aliases"] = _parse_aliases_json(row.get("aliases_json"))
    if public_mode:
        row["display_name"] = public_person_name(row, index=index)
        if not row["display_name"]:
            row["display_name"] = "非公開"
    return row


def _parse_aliases_json(value: Any) -> list[str]:
    if not value:
        return []
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in payload] if isinstance(payload, list) else []


def _validate_privacy_level(value: str) -> None:
    if value not in VALID_PERSON_PRIVACY_LEVELS:
        raise ValueError(f"invalid person privacy_level: {value}")


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
