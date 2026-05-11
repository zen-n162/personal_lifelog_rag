"""Manual links between LINE speakers and person labels.

The app never infers identity from LINE names or face clusters. Links in this
module are created only by explicit user action.
"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from typing import Any
import uuid

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.db.schema import initialize_schema
from personal_lifelog_rag.faces.person_service import add_person_alias, get_person


def list_line_speakers(repository: LifelogRepository, *, limit: int = 100) -> list[dict[str, Any]]:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        rows = connection.execute(
            """
            SELECT line_messages.chat_id,
                   line_messages.sender AS speaker_name,
                   COUNT(*) AS message_count,
                   MIN(line_messages.sent_at) AS first_seen_at,
                   MAX(line_messages.sent_at) AS last_seen_at,
                   GROUP_CONCAT(DISTINCT persons.id) AS linked_person_ids,
                   GROUP_CONCAT(DISTINCT persons.display_name) AS linked_person_names,
                   GROUP_CONCAT(DISTINCT persons.public_name) AS linked_person_public_names
            FROM line_messages
            LEFT JOIN line_speaker_links
              ON line_speaker_links.chat_id = line_messages.chat_id
             AND line_speaker_links.speaker_name = line_messages.sender
            LEFT JOIN persons ON persons.id = line_speaker_links.person_id
            WHERE line_messages.sender IS NOT NULL
              AND trim(line_messages.sender) != ''
            GROUP BY line_messages.chat_id, line_messages.sender
            ORDER BY message_count DESC, line_messages.chat_id ASC, line_messages.sender ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def link_line_speaker_to_person(
    repository: LifelogRepository,
    *,
    chat_id: str,
    speaker_name: str,
    person_id: str,
    yes: bool = False,
    add_alias: bool = False,
) -> dict[str, Any]:
    if not yes:
        raise ValueError("linking a LINE speaker to a person requires --yes")
    clean_chat_id = chat_id.strip()
    clean_speaker = speaker_name.strip()
    if not clean_chat_id:
        raise ValueError("chat_id is required")
    if not clean_speaker:
        raise ValueError("speaker_name is required")
    link_id = f"line_speaker_link_{uuid.uuid4().hex[:16]}"
    now = _now()
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        if connection.execute("SELECT 1 FROM persons WHERE id = ?", (person_id,)).fetchone() is None:
            raise ValueError(f"person not found: {person_id}")
        connection.execute(
            """
            INSERT OR REPLACE INTO line_speaker_links (
                id, chat_id, speaker_name, person_id, source, confidence,
                verified_by_user, verified_at, created_at, updated_at
            )
            VALUES (
                COALESCE(
                    (SELECT id FROM line_speaker_links WHERE chat_id = ? AND speaker_name = ? AND person_id = ?),
                    ?
                ),
                ?, ?, ?, 'manual', 1.0, 1, ?, ?, ?
            )
            """,
            (
                clean_chat_id,
                clean_speaker,
                person_id,
                link_id,
                clean_chat_id,
                clean_speaker,
                person_id,
                now,
                now,
                now,
            ),
        )
        connection.commit()
    if add_alias:
        _add_alias_if_missing(repository, person_id=person_id, alias=clean_speaker)
    return {
        "chat_id": clean_chat_id,
        "speaker_name": clean_speaker,
        "person": get_person(repository, person_id),
    }


def unlink_line_speaker_from_person(
    repository: LifelogRepository,
    *,
    chat_id: str,
    speaker_name: str,
    person_id: str,
    yes: bool = False,
) -> dict[str, Any]:
    if not yes:
        raise ValueError("unlinking a LINE speaker from a person requires --yes")
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        cursor = connection.execute(
            """
            DELETE FROM line_speaker_links
            WHERE chat_id = ? AND speaker_name = ? AND person_id = ?
            """,
            (chat_id, speaker_name, person_id),
        )
        connection.commit()
    return {"chat_id": chat_id, "speaker_name": speaker_name, "person_id": person_id, "deleted": int(cursor.rowcount)}


def list_line_speaker_links(repository: LifelogRepository, *, limit: int = 200) -> list[dict[str, Any]]:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        rows = connection.execute(
            """
            SELECT line_speaker_links.*,
                   persons.display_name,
                   persons.public_name,
                   persons.privacy_level
            FROM line_speaker_links
            JOIN persons ON persons.id = line_speaker_links.person_id
            ORDER BY line_speaker_links.created_at ASC, line_speaker_links.id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def suggest_line_speaker_persons(repository: LifelogRepository, *, speaker_name: str, limit: int = 10) -> list[dict[str, Any]]:
    term = speaker_name.strip()
    if not term:
        return []
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        rows = connection.execute(
            """
            SELECT persons.id, persons.display_name, persons.public_name, persons.privacy_level,
                   CASE
                     WHEN persons.display_name = ? THEN 'display_name exact match'
                     WHEN persons.public_name = ? THEN 'public_name exact match'
                     WHEN person_aliases.alias = ? THEN 'person alias exact match'
                     ELSE 'name contains speaker text'
                   END AS reason
            FROM persons
            LEFT JOIN person_aliases ON person_aliases.person_id = persons.id
            WHERE persons.display_name = ?
               OR persons.public_name = ?
               OR person_aliases.alias = ?
               OR persons.display_name LIKE ?
            GROUP BY persons.id
            ORDER BY persons.created_at ASC
            LIMIT ?
            """,
            (term, term, term, term, term, term, f"%{term}%", limit),
        ).fetchall()
        return [dict(row) for row in rows]


def find_person_by_name_or_alias(repository: LifelogRepository, name: str) -> dict[str, Any] | None:
    term = name.strip()
    if not term:
        return None
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        row = connection.execute(
            """
            SELECT persons.*
            FROM persons
            LEFT JOIN person_aliases ON person_aliases.person_id = persons.id
            WHERE persons.display_name = ?
               OR persons.public_name = ?
               OR person_aliases.alias = ?
            ORDER BY persons.created_at ASC
            LIMIT 1
            """,
            (term, term, term),
        ).fetchone()
        return dict(row) if row else None


def search_person_line_days(repository: LifelogRepository, *, person_name: str, limit: int = 10) -> dict[str, Any]:
    person = find_person_by_name_or_alias(repository, person_name)
    if person is None:
        return {
            "person_name": person_name,
            "person": None,
            "results": [],
            "answer": f"{person_name} に手動リンクされたLINE話者は見つかりませんでした。",
        }
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        rows = connection.execute(
            """
            SELECT substr(line_messages.sent_at, 1, 10) AS date,
                   COUNT(*) AS message_count,
                   MIN(line_messages.sent_at) AS first_seen_at,
                   MAX(line_messages.sent_at) AS last_seen_at,
                   GROUP_CONCAT(DISTINCT line_messages.chat_id) AS chat_ids,
                   GROUP_CONCAT(DISTINCT line_messages.sender) AS speakers
            FROM line_speaker_links
            JOIN line_messages
              ON line_messages.chat_id = line_speaker_links.chat_id
             AND line_messages.sender = line_speaker_links.speaker_name
            WHERE line_speaker_links.person_id = ?
            GROUP BY substr(line_messages.sent_at, 1, 10)
            ORDER BY message_count DESC, date DESC
            LIMIT ?
            """,
            (person["id"], limit),
        ).fetchall()
        results = [dict(row) for row in rows]
    answer = format_person_line_search_answer(person_name, person, results)
    return {"person_name": person_name, "person": person, "results": results, "answer": answer}


def format_person_line_search_answer(person_name: str, person: dict[str, Any], results: list[dict[str, Any]]) -> str:
    if not results:
        return f"{person_name} に手動リンクされたLINE話者のメッセージ日は見つかりませんでした。"
    top_dates = ", ".join(str(row["date"]) for row in results[:5])
    lines = [
        f"{person_name} と手動リンク済みLINE話者の記録が見つかりました。",
        "これはユーザーが手動で設定したLINE話者リンクに基づく集計です。関係性は推定していません。",
        f"主な日付: {top_dates}",
        "",
        "候補:",
    ]
    for row in results[:5]:
        lines.append(f"- {row['date']}: LINE messages={row['message_count']} first={row.get('first_seen_at') or ''}")
    return "\n".join(lines)


def format_line_speakers(rows: list[dict[str, Any]]) -> str:
    lines = ["LINE speakers"]
    if not rows:
        lines.append("- none")
        return "\n".join(lines)
    for row in rows:
        lines.append(
            f"- chat_id={row.get('chat_id') or ''} speaker={row.get('speaker_name') or ''} "
            f"messages={row.get('message_count') or 0} first={row.get('first_seen_at') or ''} "
            f"last={row.get('last_seen_at') or ''} linked={row.get('linked_person_names') or ''} "
            f"public={row.get('linked_person_public_names') or ''}"
        )
    return "\n".join(lines)


def format_line_speaker_links(rows: list[dict[str, Any]]) -> str:
    lines = ["LINE speaker links"]
    if not rows:
        lines.append("- none")
        return "\n".join(lines)
    for row in rows:
        lines.append(
            f"- {row.get('id')}: chat_id={row.get('chat_id') or ''} speaker={row.get('speaker_name') or ''} "
            f"person={row.get('person_id') or ''} name={row.get('display_name') or ''}"
        )
    return "\n".join(lines)


def format_line_speaker_suggestions(rows: list[dict[str, Any]]) -> str:
    lines = ["LINE speaker person suggestions"]
    if not rows:
        lines.append("- none")
        return "\n".join(lines)
    for row in rows:
        lines.append(
            f"- person={row.get('id')} name={row.get('display_name') or ''} "
            f"public={row.get('public_name') or ''} reason={row.get('reason') or ''}"
        )
    return "\n".join(lines)


def _add_alias_if_missing(repository: LifelogRepository, *, person_id: str, alias: str) -> None:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        exists = connection.execute(
            "SELECT 1 FROM person_aliases WHERE person_id = ? AND alias = ? LIMIT 1",
            (person_id, alias),
        ).fetchone()
    if exists is None:
        add_person_alias(repository, person_id=person_id, alias=alias, source="line_speaker")


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
