"""Local privacy controls for people, faces, places, and public exports.

All operations are local-only. Destructive operations default to dry-run style
previews and record an audit row in ``privacy_actions`` when executed.
"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import uuid

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.db.schema import initialize_schema
from personal_lifelog_rag.faces.person_service import public_person_name
from personal_lifelog_rag.places.location_store import public_place_label
from personal_lifelog_rag.reporting.portfolio_html import check_public_portfolio_path, check_public_portfolio_text


VALID_PERSON_EXPORT_MODES = {"private", "public_redacted"}


def log_privacy_action(
    repository: LifelogRepository,
    *,
    action_type: str,
    target_type: str,
    target_id: str | None,
    mode: str,
    details: dict[str, Any] | None = None,
) -> str:
    action_id = f"privacy_action_{uuid.uuid4().hex[:16]}"
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        connection.execute(
            """
            INSERT INTO privacy_actions (id, action_type, target_type, target_id, mode, details_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                action_id,
                action_type,
                target_type,
                target_id,
                mode,
                json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
                _now(),
            ),
        )
        connection.commit()
    return action_id


def privacy_audit(
    repository: LifelogRepository,
    *,
    public: bool = False,
    public_paths: list[str | Path] | None = None,
    log_action: bool = True,
) -> dict[str, Any]:
    paths = [Path(path) for path in (public_paths or [])]
    if public and not paths:
        paths = [Path("reports/portfolio_public.html")]
    issues: list[dict[str, Any]] = []
    checked_files: list[str] = []
    blocked_patterns: set[str] = set()

    for path in paths:
        if not path.exists():
            issues.append({"file": str(path), "pattern": "missing_file", "line": 0, "snippet": "file not found"})
            continue
        checked_files.append(str(path))
        if path.name.endswith(".html"):
            report = check_public_portfolio_path(path)
        else:
            report = check_public_portfolio_text(path.read_text(encoding="utf-8", errors="replace"), file_name=str(path))
        blocked_patterns.update(report.get("blocked_patterns") or [])
        issues.extend(report.get("issues") or [])

    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        counts = {
            "persons_total": _count(connection, "SELECT COUNT(*) FROM persons"),
            "persons_hidden": _count(connection, "SELECT COUNT(*) FROM persons WHERE COALESCE(hidden, 0) = 1"),
            "persons_deleted": _count(connection, "SELECT COUNT(*) FROM persons WHERE deleted_at IS NOT NULL"),
            "persons_public_hidden": _count(connection, "SELECT COUNT(*) FROM persons WHERE privacy_level = 'public_hidden'"),
            "places_hidden": _count(connection, "SELECT COUNT(*) FROM places WHERE COALESCE(hidden, 0) = 1"),
            "places_public_hidden": _count(connection, "SELECT COUNT(*) FROM places WHERE privacy_level = 'public_hidden'"),
            "face_crops": _count(
                connection,
                """
                SELECT COUNT(*)
                FROM face_detections
                WHERE crop_path IS NOT NULL OR thumbnail_path IS NOT NULL
                """,
            ),
            "face_embeddings": _count(connection, "SELECT COUNT(*) FROM face_embeddings"),
            "privacy_actions": _count(connection, "SELECT COUNT(*) FROM privacy_actions"),
        }

    result = {
        "public": public,
        "passed": not issues,
        "issue_count": len(issues),
        "issues": issues,
        "checked_files": checked_files,
        "blocked_patterns": sorted(blocked_patterns),
        "counts": counts,
    }
    if log_action:
        log_privacy_action(
            repository,
            action_type="privacy_audit",
            target_type="portfolio" if public else "database",
            target_id=None,
            mode="executed",
            details={"public": public, "passed": result["passed"], "issue_count": result["issue_count"]},
        )
    return result


def person_export(
    repository: LifelogRepository,
    *,
    person_id: str,
    output: str | Path | None = None,
    mode: str = "public_redacted",
    dry_run: bool = True,
    yes: bool = False,
) -> dict[str, Any]:
    if mode not in VALID_PERSON_EXPORT_MODES:
        raise ValueError(f"invalid person export mode: {mode}")
    payload = _person_export_payload(repository, person_id=person_id, mode=mode)
    result = {
        "person_id": person_id,
        "mode": mode,
        "dry_run": dry_run,
        "output": str(output) if output else None,
        "payload": payload if dry_run else None,
        "would_write": output is not None,
    }
    if dry_run:
        return result
    if not yes:
        raise ValueError("person-export writes a file and requires --yes unless --dry-run is used")
    if output:
        path = Path(output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        result["written"] = str(path)
    action_id = log_privacy_action(
        repository,
        action_type="export_person",
        target_type="person",
        target_id=person_id,
        mode="executed",
        details={"export_mode": mode, "output": str(output) if output else None},
    )
    result["privacy_action_id"] = action_id
    return result


def person_detach(
    repository: LifelogRepository,
    *,
    person_id: str,
    dry_run: bool = True,
    yes: bool = False,
) -> dict[str, Any]:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        _require_person(connection, person_id)
        counts = {
            "person_face_clusters": _count_params(connection, "SELECT COUNT(*) FROM person_face_clusters WHERE person_id = ?", [person_id]),
            "person_aliases": _count_params(connection, "SELECT COUNT(*) FROM person_aliases WHERE person_id = ?", [person_id]),
            "line_speaker_links": _count_params(connection, "SELECT COUNT(*) FROM line_speaker_links WHERE person_id = ?", [person_id]),
            "media_people": _count_params(connection, "SELECT COUNT(*) FROM media_people WHERE person_id = ?", [person_id]),
            "event_people": _count_params(connection, "SELECT COUNT(*) FROM event_people WHERE person_id = ?", [person_id]),
            "person_line_mentions": _count_params(connection, "SELECT COUNT(*) FROM person_line_mentions WHERE person_id = ?", [person_id]),
        }
        if dry_run:
            return {"person_id": person_id, "dry_run": True, "would_detach": counts}
        if not yes:
            raise ValueError("person-detach requires --yes unless --dry-run is used")
        for table in ("person_face_clusters", "person_aliases", "line_speaker_links", "media_people", "event_people", "person_line_mentions"):
            connection.execute(f"DELETE FROM {table} WHERE person_id = ?", (person_id,))
        connection.commit()
    action_id = log_privacy_action(
        repository,
        action_type="detach_person",
        target_type="person",
        target_id=person_id,
        mode="executed",
        details={"detached": counts},
    )
    return {"person_id": person_id, "dry_run": False, "detached": counts, "privacy_action_id": action_id}


def person_delete(
    repository: LifelogRepository,
    *,
    person_id: str,
    soft: bool = True,
    hard_delete: bool = False,
    dry_run: bool = True,
    yes: bool = False,
) -> dict[str, Any]:
    if hard_delete:
        raise ValueError("hard person delete is intentionally not implemented in PR72; detach first and keep a backup")
    if not soft:
        soft = True
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        person = _require_person(connection, person_id)
        linked_counts = {
            "media_people": _count_params(connection, "SELECT COUNT(*) FROM media_people WHERE person_id = ?", [person_id]),
            "event_people": _count_params(connection, "SELECT COUNT(*) FROM event_people WHERE person_id = ?", [person_id]),
            "line_speaker_links": _count_params(connection, "SELECT COUNT(*) FROM line_speaker_links WHERE person_id = ?", [person_id]),
            "person_face_clusters": _count_params(connection, "SELECT COUNT(*) FROM person_face_clusters WHERE person_id = ?", [person_id]),
        }
        if dry_run:
            return {"person_id": person_id, "dry_run": True, "would_soft_delete": True, "person": _redact_person_row(person), "links": linked_counts}
        if not yes:
            raise ValueError("person-delete requires --yes unless --dry-run is used")
        now = _now()
        connection.execute(
            """
            UPDATE persons
            SET deleted_at = COALESCE(deleted_at, ?),
                hidden = 1,
                searchable = 0,
                event_usable = 0,
                manual_verified = 0,
                updated_at = ?
            WHERE id = ?
            """,
            (now, now, person_id),
        )
        connection.execute("UPDATE media_people SET hidden = 1, updated_at = ? WHERE person_id = ?", (now, person_id))
        connection.execute("UPDATE event_people SET hidden = 1, updated_at = ? WHERE person_id = ?", (now, person_id))
        connection.commit()
    action_id = log_privacy_action(
        repository,
        action_type="delete_person",
        target_type="person",
        target_id=person_id,
        mode="executed",
        details={"soft": True, "links_hidden": linked_counts},
    )
    return {"person_id": person_id, "dry_run": False, "soft_deleted": True, "links_hidden": linked_counts, "privacy_action_id": action_id}


def face_delete_data(
    repository: LifelogRepository,
    *,
    face_id: str,
    delete_crop: bool = False,
    delete_embedding: bool = False,
    dry_run: bool = True,
    yes: bool = False,
) -> dict[str, Any]:
    if not delete_crop and not delete_embedding:
        raise ValueError("select --delete-crop and/or --delete-embedding")
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        row = connection.execute("SELECT * FROM face_detections WHERE id = ?", (face_id,)).fetchone()
        if row is None:
            raise ValueError(f"face detection not found: {face_id}")
        face = dict(row)
        has_embedding = connection.execute("SELECT 1 FROM face_embeddings WHERE face_id = ?", (face_id,)).fetchone() is not None
        paths = [path for path in (face.get("crop_path"), face.get("thumbnail_path")) if path]
        existing_paths = [str(Path(path)) for path in paths if Path(path).exists()]
        result = {
            "face_id": face_id,
            "dry_run": dry_run,
            "delete_crop": delete_crop,
            "delete_embedding": delete_embedding,
            "crop_paths": paths,
            "existing_crop_paths": existing_paths,
            "has_embedding": has_embedding,
        }
        if dry_run:
            return result
        if not yes:
            raise ValueError("face-delete-data requires --yes unless --dry-run is used")
        deleted_files: list[str] = []
        if delete_crop:
            for path_text in paths:
                path = Path(path_text)
                if path.exists() and path.is_file():
                    path.unlink()
                    deleted_files.append(str(path))
            connection.execute(
                "UPDATE face_detections SET crop_path = NULL, thumbnail_path = NULL, hidden = 1, updated_at = ? WHERE id = ?",
                (_now(), face_id),
            )
        deleted_embeddings = 0
        if delete_embedding:
            cursor = connection.execute("DELETE FROM face_embeddings WHERE face_id = ?", (face_id,))
            deleted_embeddings = int(cursor.rowcount)
        connection.commit()
    action_id = log_privacy_action(
        repository,
        action_type="delete_face_embedding" if delete_embedding and not delete_crop else "delete_face_crop",
        target_type="face",
        target_id=face_id,
        mode="executed",
        details={"deleted_files": deleted_files, "deleted_embeddings": deleted_embeddings},
    )
    result.update({"dry_run": False, "deleted_files": deleted_files, "deleted_embeddings": deleted_embeddings, "privacy_action_id": action_id})
    return result


def hide_place(
    repository: LifelogRepository,
    *,
    place_id: str,
    dry_run: bool = True,
    yes: bool = False,
) -> dict[str, Any]:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        row = connection.execute("SELECT * FROM places WHERE id = ?", (place_id,)).fetchone()
        if row is None:
            raise ValueError(f"place not found: {place_id}")
        place = dict(row)
        if dry_run:
            return {"place_id": place_id, "dry_run": True, "would_hide": _redact_place_row(place)}
        if not yes:
            raise ValueError("places hide requires --yes unless --dry-run is used")
        now = _now()
        connection.execute(
            """
            UPDATE places
            SET hidden = 1,
                searchable = 0,
                privacy_level = 'public_hidden',
                updated_at = ?
            WHERE id = ?
            """,
            (now, place_id),
        )
        connection.commit()
    action_id = log_privacy_action(
        repository,
        action_type="hide_place",
        target_type="place",
        target_id=place_id,
        mode="executed",
        details={"previous_privacy_level": place.get("privacy_level")},
    )
    return {"place_id": place_id, "dry_run": False, "hidden": True, "privacy_action_id": action_id}


def format_privacy_audit(report: dict[str, Any]) -> str:
    lines = [
        "Privacy audit",
        f"- public mode: {report.get('public')}",
        f"- passed: {report.get('passed')}",
        f"- issue_count: {report.get('issue_count')}",
    ]
    if report.get("checked_files"):
        lines.append("- checked files:")
        lines.extend(f"  - {path}" for path in report["checked_files"])
    counts = report.get("counts") or {}
    if counts:
        lines.append("- local privacy counts:")
        for key in sorted(counts):
            lines.append(f"  - {key}: {counts[key]}")
    if report.get("issues"):
        lines.append("- issues:")
        for issue in report["issues"]:
            lines.append(f"  - {issue.get('file')}:{issue.get('line')} {issue.get('pattern')} {issue.get('snippet')}")
    return "\n".join(lines)


def format_privacy_operation(report: dict[str, Any], *, title: str) -> str:
    lines = [title]
    for key, value in report.items():
        if key == "payload":
            lines.append(f"- payload keys: {', '.join(sorted(value.keys())) if isinstance(value, dict) else 'none'}")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _person_export_payload(repository: LifelogRepository, *, person_id: str, mode: str) -> dict[str, Any]:
    public = mode == "public_redacted"
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        person = _require_person(connection, person_id)
        label = public_person_name(person, index=1) or "人物A" if public else person.get("display_name")
        aliases = [
            row["alias"]
            for row in connection.execute(
                "SELECT alias FROM person_aliases WHERE person_id = ? ORDER BY created_at ASC, id ASC",
                (person_id,),
            ).fetchall()
        ]
        line_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT line_speaker_links.chat_id,
                       line_speaker_links.speaker_name,
                       COUNT(line_messages.id) AS message_count,
                       MIN(line_messages.sent_at) AS first_seen_at,
                       MAX(line_messages.sent_at) AS last_seen_at
                FROM line_speaker_links
                LEFT JOIN line_messages
                  ON line_messages.chat_id = line_speaker_links.chat_id
                 AND line_messages.sender = line_speaker_links.speaker_name
                WHERE line_speaker_links.person_id = ?
                GROUP BY line_speaker_links.chat_id, line_speaker_links.speaker_name
                ORDER BY first_seen_at ASC
                """,
                (person_id,),
            ).fetchall()
        ]
        media_count = _count_params(connection, "SELECT COUNT(*) FROM media_people WHERE person_id = ? AND COALESCE(hidden, 0) = 0", [person_id])
        event_count = _count_params(connection, "SELECT COUNT(*) FROM event_people WHERE person_id = ? AND COALESCE(hidden, 0) = 0", [person_id])
        clusters_count = _count_params(connection, "SELECT COUNT(*) FROM person_face_clusters WHERE person_id = ?", [person_id])

    payload: dict[str, Any] = {
        "mode": mode,
        "person": {
            "id": person_id if not public else "PERSON_1",
            "name": label,
            "privacy_level": person.get("privacy_level"),
            "hidden": int(person.get("hidden") or 0),
            "deleted": bool(person.get("deleted_at")),
        },
        "counts": {
            "line_speaker_links": len(line_rows),
            "face_clusters": clusters_count,
            "media_people": media_count,
            "event_people": event_count,
        },
        "line_activity": [
            {
                "speaker_name": row["speaker_name"] if not public else "SENDER_1",
                "message_count": int(row.get("message_count") or 0),
                "first_seen_at": row.get("first_seen_at"),
                "last_seen_at": row.get("last_seen_at"),
            }
            for row in line_rows
        ],
    }
    if not public:
        payload["person"]["display_name"] = person.get("display_name")
        payload["person"]["public_name"] = person.get("public_name")
        payload["aliases"] = aliases
    return payload


def _require_person(connection, person_id: str) -> dict[str, Any]:
    row = connection.execute("SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()
    if row is None:
        raise ValueError(f"person not found: {person_id}")
    return dict(row)


def _redact_person_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "public_name": row.get("public_name"),
        "privacy_level": row.get("privacy_level"),
        "hidden": int(row.get("hidden") or 0),
        "deleted_at": bool(row.get("deleted_at")),
    }


def _redact_place_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "public_label": public_place_label(row),
        "category": row.get("category"),
        "privacy_level": row.get("privacy_level"),
        "hidden": int(row.get("hidden") or 0),
    }


def _count(connection, sql: str) -> int:
    row = connection.execute(sql).fetchone()
    return int(row[0] or 0)


def _count_params(connection, sql: str, params: list[Any]) -> int:
    row = connection.execute(sql, params).fetchone()
    return int(row[0] or 0)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
