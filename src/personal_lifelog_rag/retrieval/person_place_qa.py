"""Manual person/place QA helpers.

This module intentionally avoids identity inference. It only uses persons,
LINE speaker links, media_people/event_people, and places that have been
explicitly stored in the local database by the user.
"""

from __future__ import annotations

from collections import Counter
from contextlib import closing
from dataclasses import dataclass
import json
import re
from typing import Any

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.db.schema import initialize_schema
from personal_lifelog_rag.faces.person_service import public_person_name
from personal_lifelog_rag.places.location_store import public_place_label


PERSON_QA_INTENTS = {
    "person_line_search",
    "person_photo_search",
    "person_event_search",
    "person_place_search",
    "person_activity_search",
}
PLACE_QA_INTENTS = {
    "place_visit_search",
    "place_photo_search",
    "place_activity_search",
    "monthly_place_summary",
}
FORBIDDEN_RELATIONSHIP_TERMS = (
    "恋人",
    "彼氏",
    "彼女",
    "家族",
    "友人",
    "親密",
    "交際",
    "確実に一緒",
)
FOOD_ACTIVITY_TERMS = ("ご飯", "食事", "食べ", "カフェ", "ランチ", "ディナー", "料理", "meal", "food", "cafe")
PHOTO_TERMS = ("写真", "画像", "写って", "撮った", "撮影")
LINE_TERMS = ("LINE", "ライン", "話した", "話して", "やりとり")
CALL_TERMS = ("通話", "電話", "長電話", "話した")
VISIT_TERMS = ("行った", "行く", "いた", "いる", "着いた", "到着", "寄った")
PLACE_SUMMARY_TERMS = ("行った場所", "よく行った場所", "場所は", "場所を")


@dataclass(frozen=True)
class EntityResolution:
    status: str
    query_name: str | None
    candidates: list[dict[str, Any]]

    @property
    def resolved(self) -> dict[str, Any] | None:
        return self.candidates[0] if self.status == "resolved" and self.candidates else None


def route_person_place_query(
    repository: LifelogRepository,
    query: str,
    entities: dict[str, Any],
    *,
    limit: int = 5,
    include_hidden: bool = False,
    public_mode: bool = False,
) -> dict[str, Any] | None:
    """Return a routed QA response for manual person/place queries if applicable."""

    text = query.strip()
    if not text:
        return None

    if _looks_like_monthly_place_summary(text, entities):
        report = monthly_place_summary(repository, entities=entities, limit=limit, public_mode=public_mode)
        return _route_payload(
            intent="monthly_place_summary",
            routing="person-place-qa",
            answer=_format_monthly_place_summary(report),
            results=report["results"],
            person=None,
            place=None,
            privacy_mode=_privacy_mode(public_mode),
            overclaim_flags=[],
            source_counts=report["source_counts"],
        )

    person_name = _extract_person_name(text, entities)
    place_name = _extract_place_name(text, entities, person_name=person_name)

    person_resolution = resolve_person(repository, person_name, public_mode=public_mode) if person_name else None
    place_resolution = resolve_place(repository, place_name, public_mode=public_mode) if place_name else None

    if person_resolution and person_resolution.status == "ambiguous":
        answer = _ambiguous_answer("人物", person_resolution.candidates)
        return _route_payload(
            intent="person_event_search",
            routing="entity-resolution",
            answer=answer,
            results=[],
            person=None,
            place=place_resolution.resolved if place_resolution else None,
            privacy_mode=_privacy_mode(public_mode),
            overclaim_flags=_overclaim_flags(answer),
        )
    if place_resolution and place_resolution.status == "ambiguous":
        answer = _ambiguous_answer("場所", place_resolution.candidates)
        return _route_payload(
            intent="place_visit_search",
            routing="entity-resolution",
            answer=answer,
            results=[],
            person=person_resolution.resolved if person_resolution else None,
            place=None,
            privacy_mode=_privacy_mode(public_mode),
            overclaim_flags=_overclaim_flags(answer),
        )

    person = person_resolution.resolved if person_resolution else None
    place = place_resolution.resolved if place_resolution else None

    if person_name and person is None and _looks_like_person_query(text):
        if any(term in text for term in LINE_TERMS):
            answer = "手動リンクされたLINE話者は見つかりませんでした。"
        else:
            answer = "該当する手動確認済みpersonまたはLINE話者リンクは見つかりませんでした。"
        return _route_payload(
            intent=_person_intent(text, has_place=bool(place_name)),
            routing="person-place-qa",
            answer=answer,
            results=[],
            person=None,
            place=place,
            privacy_mode=_privacy_mode(public_mode),
            overclaim_flags=_overclaim_flags(answer),
        )
    if place_name and place is None and person is None:
        # Preserve existing local place/text search when no reviewed place label
        # is available yet. Missing place labels are not treated as hard errors.
        return None

    if person:
        intent = _person_intent(text, has_place=place is not None)
        if intent == "person_line_search":
            results = search_person_line_days(repository, person_id=str(person["id"]), limit=limit, public_mode=public_mode)
        elif intent == "person_photo_search":
            results = search_person_photos(repository, person_id=str(person["id"]), limit=limit, public_mode=public_mode)
        else:
            results = search_person_events(
                repository,
                person_id=str(person["id"]),
                place_id=str(place["id"]) if place else None,
                activity=_activity_from_query(text),
                limit=limit,
                include_hidden=include_hidden,
                public_mode=public_mode,
            )
        answer = _format_person_answer(text, person, place, results, intent=intent, public_mode=public_mode)
        return _route_payload(
            intent=intent,
            routing="person-place-qa",
            answer=answer,
            results=results,
            person=person,
            place=place,
            privacy_mode=_privacy_mode(public_mode),
            overclaim_flags=_overclaim_flags(answer),
            source_counts=_source_counts(results),
        )

    if place:
        intent = _place_intent(text)
        if intent == "place_photo_search":
            results = search_place_photos(repository, place_id=str(place["id"]), limit=limit, public_mode=public_mode)
        else:
            results = search_place_events(
                repository,
                place_id=str(place["id"]),
                activity=_activity_from_query(text),
                limit=limit,
                include_hidden=include_hidden,
                public_mode=public_mode,
            )
        answer = _format_place_answer(text, place, results, intent=intent, public_mode=public_mode)
        return _route_payload(
            intent=intent,
            routing="person-place-qa",
            answer=answer,
            results=results,
            person=None,
            place=place,
            privacy_mode=_privacy_mode(public_mode),
            overclaim_flags=_overclaim_flags(answer),
            source_counts=_source_counts(results),
        )

    return None


def resolve_person(repository: LifelogRepository, query_name: str | None, *, public_mode: bool = False) -> EntityResolution:
    term = (query_name or "").strip()
    if not term:
        return EntityResolution("none", None, [])
    matches: dict[str, dict[str, Any]] = {}
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        person_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT *
                FROM persons
                WHERE manual_verified = 1
                  AND COALESCE(hidden, 0) = 0
                  AND COALESCE(searchable, 1) = 1
                  AND deleted_at IS NULL
                """
            ).fetchall()
        ]
        alias_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT person_aliases.person_id, person_aliases.alias, persons.*
                FROM person_aliases
                JOIN persons ON persons.id = person_aliases.person_id
                WHERE persons.manual_verified = 1
                  AND COALESCE(persons.hidden, 0) = 0
                  AND COALESCE(persons.searchable, 1) = 1
                  AND persons.deleted_at IS NULL
                """
            ).fetchall()
        ]
        speaker_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT line_speaker_links.person_id, line_speaker_links.speaker_name, persons.*
                FROM line_speaker_links
                JOIN persons ON persons.id = line_speaker_links.person_id
                WHERE line_speaker_links.verified_by_user = 1
                  AND persons.manual_verified = 1
                  AND COALESCE(persons.hidden, 0) = 0
                  AND COALESCE(persons.searchable, 1) = 1
                  AND persons.deleted_at IS NULL
                """
            ).fetchall()
        ]
    for row in person_rows:
        aliases = set(_json_list(row.get("aliases_json")))
        if term in {str(row.get("display_name") or ""), str(row.get("public_name") or ""), *aliases}:
            _add_person_match(matches, row, "person_name", public_mode=public_mode)
    for row in alias_rows:
        if term == str(row.get("alias") or ""):
            _add_person_match(matches, row, "alias", public_mode=public_mode)
    for row in speaker_rows:
        if term == str(row.get("speaker_name") or ""):
            _add_person_match(matches, row, "line_speaker", public_mode=public_mode)
    candidates = sorted(matches.values(), key=lambda row: str(row.get("created_at") or ""))
    if not candidates:
        return EntityResolution("none", term, [])
    return EntityResolution("resolved" if len(candidates) == 1 else "ambiguous", term, candidates)


def resolve_persons_from_query(repository: LifelogRepository, query: str, *, public_mode: bool = False) -> EntityResolution:
    """Resolve a manually verified person mentioned in a natural-language query."""

    entities: dict[str, Any] = {"raw_terms": re.split(r"[\s,、/]+|(?:で|に|を|は|が|と)", query)}
    person_name = _extract_person_name(query, entities)
    if person_name:
        return resolve_person(repository, person_name, public_mode=public_mode)
    for raw in entities["raw_terms"]:
        term = _clean_entity(str(raw))
        if not term or term in {"写真", "画像", "LINE", "ライン", "ご飯", "食事", "カフェ", "いつ"}:
            continue
        result = resolve_person(repository, term, public_mode=public_mode)
        if result.status != "none":
            return result
    return EntityResolution("none", None, [])


def get_person_line_speakers(repository: LifelogRepository, person_id: str) -> list[dict[str, Any]]:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        return [
            dict(row)
            for row in connection.execute(
                """
                SELECT chat_id, speaker_name, source, confidence, verified_by_user, verified_at
                FROM line_speaker_links
                WHERE person_id = ?
                  AND verified_by_user = 1
                ORDER BY chat_id ASC, speaker_name ASC
                """,
                (person_id,),
            ).fetchall()
        ]


def get_person_face_clusters(repository: LifelogRepository, person_id: str) -> list[dict[str, Any]]:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        return [
            dict(row)
            for row in connection.execute(
                """
                SELECT face_clusters.id, face_clusters.cluster_label, face_clusters.face_count,
                       face_clusters.status, face_clusters.review_status,
                       person_face_clusters.verified_by_user, person_face_clusters.verified_at
                FROM person_face_clusters
                JOIN face_clusters ON face_clusters.id = person_face_clusters.face_cluster_id
                WHERE person_face_clusters.person_id = ?
                  AND person_face_clusters.verified_by_user = 1
                  AND face_clusters.status = 'accepted'
                ORDER BY face_clusters.first_seen_at ASC, face_clusters.id ASC
                """,
                (person_id,),
            ).fetchall()
        ]


def get_person_media(repository: LifelogRepository, person_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
    return search_person_photos(repository, person_id=person_id, limit=limit)


def get_person_events(repository: LifelogRepository, person_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
    return search_person_events(repository, person_id=person_id, limit=limit)


def get_person_evidence(repository: LifelogRepository, person_id: str) -> dict[str, Any]:
    """Return compact private evidence counts for a manually linked person."""

    return {
        "line_speakers": get_person_line_speakers(repository, person_id),
        "face_clusters": get_person_face_clusters(repository, person_id),
        "media_count": len(get_person_media(repository, person_id, limit=10_000)),
        "event_count": len(get_person_events(repository, person_id, limit=10_000)),
    }


def resolve_place(repository: LifelogRepository, query_name: str | None, *, public_mode: bool = False) -> EntityResolution:
    term = (query_name or "").strip()
    if not term:
        return EntityResolution("none", None, [])
    matches: dict[str, dict[str, Any]] = {}
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT *
                FROM places
                WHERE COALESCE(hidden, 0) = 0
                  AND COALESCE(searchable, 1) = 1
                """
            ).fetchall()
        ]
    for row in rows:
        aliases = _json_list(row.get("aliases_json"))
        values = {
            str(row.get("display_name") or ""),
            str(row.get("public_name") or ""),
            str(row.get("category") or ""),
            *aliases,
        }
        if term in values:
            _add_place_match(matches, row, "place_label", public_mode=public_mode)
    candidates = sorted(matches.values(), key=lambda row: (-int(row.get("manual_verified") or 0), str(row.get("created_at") or "")))
    if not candidates:
        return EntityResolution("none", term, [])
    return EntityResolution("resolved" if len(candidates) == 1 else "ambiguous", term, candidates)


def search_person_line_days(
    repository: LifelogRepository,
    *,
    person_id: str,
    limit: int = 10,
    public_mode: bool = False,
) -> list[dict[str, Any]]:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        rows = connection.execute(
            """
            SELECT substr(line_messages.sent_at, 1, 10) AS date,
                   COUNT(*) AS message_count,
                   MIN(line_messages.sent_at) AS first_seen_at,
                   MAX(line_messages.sent_at) AS last_seen_at,
                   COUNT(DISTINCT line_messages.chat_id) AS chat_count,
                   COUNT(DISTINCT line_speaker_links.speaker_name) AS speaker_count
            FROM line_speaker_links
            JOIN persons ON persons.id = line_speaker_links.person_id
            JOIN line_messages
              ON line_messages.chat_id = line_speaker_links.chat_id
             AND line_messages.sender = line_speaker_links.speaker_name
            WHERE line_speaker_links.person_id = ?
              AND line_speaker_links.verified_by_user = 1
              AND persons.manual_verified = 1
              AND COALESCE(persons.hidden, 0) = 0
              AND COALESCE(persons.searchable, 1) = 1
              AND persons.deleted_at IS NULL
            GROUP BY substr(line_messages.sent_at, 1, 10)
            ORDER BY message_count DESC, date DESC
            LIMIT ?
            """,
            (person_id, limit),
        ).fetchall()
        call_rows = connection.execute(
            """
            SELECT substr(line_call_events.sent_at, 1, 10) AS date,
                   COUNT(*) AS call_count,
                   SUM(CASE WHEN line_call_events.call_status = 'completed' THEN 1 ELSE 0 END) AS completed_call_count,
                   SUM(COALESCE(line_call_events.duration_sec, 0)) AS duration_sec
            FROM line_speaker_links
            JOIN persons ON persons.id = line_speaker_links.person_id
            JOIN line_call_events
              ON line_call_events.chat_id = line_speaker_links.chat_id
             AND line_call_events.sender = line_speaker_links.speaker_name
            WHERE line_speaker_links.person_id = ?
              AND line_speaker_links.verified_by_user = 1
              AND persons.manual_verified = 1
              AND COALESCE(persons.hidden, 0) = 0
              AND COALESCE(persons.searchable, 1) = 1
              AND persons.deleted_at IS NULL
            GROUP BY substr(line_call_events.sent_at, 1, 10)
            """,
            (person_id,),
        ).fetchall()
    calls_by_date = {str(row["date"]): dict(row) for row in call_rows}
    results = []
    for row in rows:
        result = dict(row)
        call = calls_by_date.get(str(row["date"])) or {}
        call_count = int(call.get("call_count") or 0)
        completed_call_count = int(call.get("completed_call_count") or 0)
        result.update(
            {
                "call_count": call_count,
                "completed_call_count": completed_call_count,
                "call_duration_sec": int(call.get("duration_sec") or 0),
                "evidence_types": ["line_speaker", *(["call"] if call_count else [])],
                "evidence_strength": "medium" if call_count else "weak",
                "confidence_label": "中" if call_count else "低",
                "source_counts": {"line_speaker": int(row["message_count"] or 0), "call": call_count},
                "privacy_mode": _privacy_mode(public_mode),
            }
        )
        results.append(result)
    known_dates = {str(row.get("date")) for row in results}
    for date_value, call in sorted(calls_by_date.items(), key=lambda item: item[0], reverse=True):
        if date_value in known_dates:
            continue
        call_count = int(call.get("call_count") or 0)
        results.append(
            {
                "date": date_value,
                "message_count": 0,
                "first_seen_at": "",
                "last_seen_at": "",
                "chat_count": 0,
                "speaker_count": 0,
                "call_count": call_count,
                "completed_call_count": int(call.get("completed_call_count") or 0),
                "call_duration_sec": int(call.get("duration_sec") or 0),
                "evidence_types": ["call", "line_speaker"],
                "evidence_strength": "weak",
                "confidence_label": "低",
                "source_counts": {"line_speaker": 0, "call": call_count},
                "privacy_mode": _privacy_mode(public_mode),
            }
        )
    results.sort(key=lambda row: (-(int(row.get("message_count") or 0) + int(row.get("call_count") or 0)), str(row.get("date") or "")))
    return results[:limit]


def search_person_photos(
    repository: LifelogRepository,
    *,
    person_id: str,
    limit: int = 10,
    public_mode: bool = False,
) -> list[dict[str, Any]]:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        rows = connection.execute(
            """
            SELECT media_people.media_id,
                   media_people.source,
                   media_people.confidence,
                   media_people.face_cluster_id,
                   COALESCE(media_items.captured_at, media_items.fallback_captured_at) AS captured_at,
                   substr(COALESCE(media_items.captured_at, media_items.fallback_captured_at), 1, 10) AS date,
                   media_items.file_name
            FROM media_people
            JOIN persons ON persons.id = media_people.person_id
            JOIN media_items ON media_items.id = media_people.media_id
            WHERE media_people.person_id = ?
              AND media_people.verified_by_user = 1
              AND COALESCE(media_people.hidden, 0) = 0
              AND persons.manual_verified = 1
              AND COALESCE(persons.hidden, 0) = 0
              AND COALESCE(persons.searchable, 1) = 1
              AND persons.deleted_at IS NULL
            ORDER BY captured_at ASC, media_people.media_id ASC
            LIMIT ?
            """,
            (person_id, limit),
        ).fetchall()
    results = []
    for row in rows:
        result = dict(row)
        result.update(
            {
                "evidence_types": ["media_people", "face_cluster"],
                "evidence_strength": "medium",
                "confidence_label": "中",
                "source_counts": {"media_people": 1},
                "privacy_mode": _privacy_mode(public_mode),
            }
        )
        if public_mode:
            result.pop("file_name", None)
            result.pop("face_cluster_id", None)
        results.append(result)
    return results


def search_person_events(
    repository: LifelogRepository,
    *,
    person_id: str,
    place_id: str | None = None,
    activity: str | None = None,
    limit: int = 10,
    include_hidden: bool = False,
    public_mode: bool = False,
) -> list[dict[str, Any]]:
    activity_clause, activity_params = _activity_clause(activity)
    place_clause = (
        """
        AND EXISTS (
            SELECT 1
            FROM event_places
            JOIN places AS place_filter ON place_filter.id = event_places.place_id
            WHERE event_places.event_id = events.id
              AND event_places.place_id = ?
              AND COALESCE(place_filter.hidden, 0) = 0
              AND COALESCE(place_filter.searchable, 1) = 1
        )
        """
        if place_id
        else ""
    )
    hidden_clause = "" if include_hidden else "AND COALESCE(event_overrides.is_hidden, 0) = 0"
    params: list[Any] = [person_id, *activity_params]
    if place_id:
        params.append(place_id)
    params.append(limit)
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        rows = connection.execute(
            f"""
            SELECT events.id AS event_id,
                   events.date,
                   events.start_time,
                   events.title,
                   events.summary,
                   event_people.source,
                   event_people.confidence,
                   event_people.evidence_count,
                   event_people.media_count,
                   event_people.line_count,
                   (
                     SELECT GROUP_CONCAT(COALESCE(places.public_name, places.category, places.display_name), ', ')
                     FROM event_places
                     JOIN places ON places.id = event_places.place_id
                     WHERE event_places.event_id = events.id
                   ) AS place_label
            FROM event_people
            JOIN persons ON persons.id = event_people.person_id
            JOIN events ON events.id = event_people.event_id
            LEFT JOIN event_overrides ON event_overrides.event_id = events.id
            WHERE event_people.person_id = ?
              AND persons.manual_verified = 1
              AND COALESCE(persons.hidden, 0) = 0
              AND COALESCE(persons.searchable, 1) = 1
              AND COALESCE(persons.event_usable, 1) = 1
              AND persons.deleted_at IS NULL
              AND COALESCE(event_people.hidden, 0) = 0
              {activity_clause}
              {place_clause}
              {hidden_clause}
            ORDER BY events.date ASC, events.start_time ASC, event_people.confidence DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [_event_result(dict(row), public_mode=public_mode, with_person=True, with_place=bool(place_id), activity=activity) for row in rows]


def search_place_events(
    repository: LifelogRepository,
    *,
    place_id: str,
    activity: str | None = None,
    limit: int = 10,
    include_hidden: bool = False,
    public_mode: bool = False,
) -> list[dict[str, Any]]:
    activity_clause, activity_params = _activity_clause(activity)
    hidden_clause = "" if include_hidden else "AND COALESCE(event_overrides.is_hidden, 0) = 0"
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        rows = connection.execute(
            f"""
            SELECT events.id AS event_id,
                   events.date,
                   events.start_time,
                   events.title,
                   events.summary,
                   event_places.source,
                   event_places.confidence,
                   places.id AS place_id,
                   places.display_name,
                   places.public_name,
                   places.category,
                   places.privacy_level
            FROM event_places
            JOIN places ON places.id = event_places.place_id
            JOIN events ON events.id = event_places.event_id
            LEFT JOIN event_overrides ON event_overrides.event_id = events.id
            WHERE event_places.place_id = ?
              AND COALESCE(places.hidden, 0) = 0
              AND COALESCE(places.searchable, 1) = 1
              {activity_clause}
              {hidden_clause}
            ORDER BY events.date ASC, events.start_time ASC, event_places.confidence DESC
            LIMIT ?
            """,
            [place_id, *activity_params, limit],
        ).fetchall()
    return [_place_event_result(dict(row), public_mode=public_mode, activity=activity) for row in rows]


def search_place_photos(
    repository: LifelogRepository,
    *,
    place_id: str,
    limit: int = 10,
    public_mode: bool = False,
) -> list[dict[str, Any]]:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        rows = connection.execute(
            """
            SELECT media_places.media_id,
                   media_places.source,
                   media_places.confidence,
                   COALESCE(media_items.captured_at, media_items.fallback_captured_at) AS captured_at,
                   substr(COALESCE(media_items.captured_at, media_items.fallback_captured_at), 1, 10) AS date,
                   media_items.file_name
            FROM media_places
            JOIN places ON places.id = media_places.place_id
            JOIN media_items ON media_items.id = media_places.media_id
            WHERE media_places.place_id = ?
              AND COALESCE(places.hidden, 0) = 0
              AND COALESCE(places.searchable, 1) = 1
            ORDER BY captured_at ASC, media_places.media_id ASC
            LIMIT ?
            """,
            (place_id, limit),
        ).fetchall()
    results = []
    for row in rows:
        result = dict(row)
        result.update(
            {
                "evidence_types": ["media_place", "place"],
                "evidence_strength": "medium",
                "confidence_label": "中",
                "source_counts": {"media_place": 1},
                "privacy_mode": _privacy_mode(public_mode),
            }
        )
        if public_mode:
            result.pop("file_name", None)
        results.append(result)
    return results


def monthly_place_summary(
    repository: LifelogRepository,
    *,
    entities: dict[str, Any],
    limit: int = 10,
    public_mode: bool = False,
) -> dict[str, Any]:
    start_date = str(entities.get("date_from") or "")
    end_date = str(entities.get("date_to") or start_date)
    if not start_date:
        return {"results": [], "source_counts": {}, "date_from": None, "date_to": None}
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        event_rows = connection.execute(
            """
            SELECT places.*, COUNT(DISTINCT event_places.event_id) AS event_count
            FROM event_places
            JOIN places ON places.id = event_places.place_id
            JOIN events ON events.id = event_places.event_id
            LEFT JOIN event_overrides ON event_overrides.event_id = events.id
            WHERE events.date >= ?
              AND events.date <= ?
              AND COALESCE(event_overrides.is_hidden, 0) = 0
              AND COALESCE(places.hidden, 0) = 0
              AND COALESCE(places.searchable, 1) = 1
            GROUP BY places.id
            ORDER BY event_count DESC, places.category ASC, places.id ASC
            LIMIT ?
            """,
            (start_date, end_date, limit),
        ).fetchall()
        media_rows = connection.execute(
            """
            SELECT places.id, COUNT(DISTINCT media_places.media_id) AS media_count
            FROM media_places
            JOIN places ON places.id = media_places.place_id
            JOIN media_items ON media_items.id = media_places.media_id
            WHERE substr(COALESCE(media_items.captured_at, media_items.fallback_captured_at), 1, 10) >= ?
              AND substr(COALESCE(media_items.captured_at, media_items.fallback_captured_at), 1, 10) <= ?
              AND COALESCE(places.hidden, 0) = 0
              AND COALESCE(places.searchable, 1) = 1
            GROUP BY places.id
            """,
            (start_date, end_date),
        ).fetchall()
    media_counts = {row["id"]: int(row["media_count"] or 0) for row in media_rows}
    results = []
    for index, row in enumerate(event_rows, start=1):
        place = dict(row)
        label = _place_label(place, public_mode=public_mode, index=index)
        results.append(
            {
                "place_id": place["id"],
                "place_label": label,
                "category": place.get("category") or "other",
                "event_count": int(place.get("event_count") or 0),
                "media_count": media_counts.get(place["id"], 0),
                "evidence_types": ["event_places", "media_places"] if media_counts.get(place["id"], 0) else ["event_places"],
                "evidence_strength": "medium",
                "confidence_label": "中",
                "privacy_mode": _privacy_mode(public_mode),
            }
        )
    return {
        "date_from": start_date,
        "date_to": end_date,
        "results": results,
        "source_counts": {
            "event_places": sum(int(row.get("event_count") or 0) for row in results),
            "media_places": sum(int(row.get("media_count") or 0) for row in results),
        },
    }


def _route_payload(
    *,
    intent: str,
    routing: str,
    answer: str,
    results: list[dict[str, Any]],
    person: dict[str, Any] | None,
    place: dict[str, Any] | None,
    privacy_mode: str,
    overclaim_flags: list[str],
    source_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    top_dates = _top_dates(results)
    evidence_types = sorted({str(evidence) for row in results for evidence in (row.get("evidence_types") or [])})
    return {
        "intent": intent,
        "routing": routing,
        "answer": answer,
        "results": results,
        "metadata": {
            "resolved_person_id": person.get("id") if person else None,
            "resolved_place_id": place.get("id") if place else None,
            "evidence_types": evidence_types,
            "top_dates": top_dates,
            "source_counts": source_counts or _source_counts(results),
            "privacy_mode": privacy_mode,
            "overclaim_flags": overclaim_flags,
        },
    }


def _format_person_answer(
    query: str,
    person: dict[str, Any],
    place: dict[str, Any] | None,
    results: list[dict[str, Any]],
    *,
    intent: str,
    public_mode: bool,
) -> str:
    label = _person_label(person, public_mode=public_mode)
    place_label = _place_label(place, public_mode=public_mode, index=1) if place else ""
    if not results:
        base = f"{label} に関連する手動確認済みの候補は見つかりませんでした。"
        return base + "\n未確認の顔クラスタや未リンクのLINE話者は使っていません。"
    top_dates = ", ".join(_top_dates(results)[:5])
    if intent == "person_line_search":
        intro = f"{label} と手動リンク済みLINE話者の記録が見つかりました。"
        caution = "LINE上でやりとりがあった日付です。会っていたことや関係性は推定していません。"
    elif intent == "person_photo_search":
        intro = f"{label} が写真に写っている可能性がある候補が見つかりました。"
        caution = "手動確認済みpersonとmedia_peopleに基づく候補です。顔だけで一緒にいたとは断定しません。"
    elif place:
        intro = f"{label} と {place_label} に関連するイベント候補が見つかりました。"
        caution = "人物・場所・イベントの手動リンク由来の候補です。一緒だったことは可能性として扱います。"
    else:
        intro = f"{label} に関連するイベント候補が見つかりました。"
        caution = "手動確認済みpersonリンクに基づく候補です。関係性は推定していません。"
    lines = [
        f"質問: {query}",
        "",
        intro,
        caution,
        f"主な日付: {top_dates}",
        "",
        "候補:",
    ]
    for row in results[:5]:
        source = ", ".join(row.get("evidence_types") or [row.get("source") or "manual"])
        count_text = ""
        if intent == "person_line_search":
            count_text = f" line={int(row.get('message_count') or 0)} call={int(row.get('call_count') or 0)}"
        lines.append(
            f"- {row.get('date') or ''}: source={source}{count_text} "
            f"evidence_strength={row.get('evidence_strength') or 'weak'} confidence={row.get('confidence_label') or '低'}"
        )
    lines.append("")
    lines.append("禁止事項: 人物関係・感情・身元は自動推定していません。")
    return "\n".join(lines)


def _format_place_answer(
    query: str,
    place: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    intent: str,
    public_mode: bool,
) -> str:
    label = _place_label(place, public_mode=public_mode, index=1)
    if not results:
        return f"{label} に関連する場所候補は見つかりませんでした。\n正確なGPS座標は表示していません。"
    top_dates = ", ".join(_top_dates(results)[:5])
    if intent == "place_photo_search":
        intro = f"{label} の写真候補が見つかりました。"
    elif intent == "place_activity_search":
        intro = f"{label} での活動候補が見つかりました。"
    else:
        intro = f"{label} に行った、または関連している可能性がある日が見つかりました。"
    lines = [
        f"質問: {query}",
        "",
        intro,
        "場所ラベル・イベント・写真のローカルDB根拠に基づく候補です。正確なGPS座標は表示していません。",
        f"主な日付: {top_dates}",
        "",
        "候補:",
    ]
    for row in results[:5]:
        source = ", ".join(row.get("evidence_types") or [row.get("source") or "place"])
        lines.append(
            f"- {row.get('date') or ''}: source={source} "
            f"evidence_strength={row.get('evidence_strength') or 'weak'} confidence={row.get('confidence_label') or '低'}"
        )
    return "\n".join(lines)


def _format_monthly_place_summary(report: dict[str, Any]) -> str:
    start_date = report.get("date_from") or ""
    end_date = report.get("date_to") or start_date
    rows = report.get("results") or []
    if not rows:
        return f"{start_date}..{end_date} の場所候補は見つかりませんでした。正確なGPS座標は表示していません。"
    lines = [
        f"{start_date}..{end_date} の場所候補",
        "",
        "手動場所ラベル・event_places・media_placesを集計しました。正確なGPS座標は表示していません。",
        "",
        "候補:",
    ]
    for row in rows[:10]:
        lines.append(
            f"- {row.get('place_label')}: category={row.get('category')} "
            f"events={row.get('event_count', 0)} media={row.get('media_count', 0)}"
        )
    return "\n".join(lines)


def _activity_clause(activity: str | None) -> tuple[str, list[Any]]:
    if activity != "food":
        return "", []
    clauses = []
    params: list[Any] = []
    for term in FOOD_ACTIVITY_TERMS:
        like = f"%{term}%"
        clauses.append("(COALESCE(events.title, '') LIKE ? OR COALESCE(events.summary, '') LIKE ?)")
        params.extend([like, like])
    return "AND (" + " OR ".join(clauses) + ")", params


def _event_result(
    row: dict[str, Any],
    *,
    public_mode: bool,
    with_person: bool,
    with_place: bool,
    activity: str | None,
) -> dict[str, Any]:
    evidence = ["event_people", str(row.get("source") or "manual")]
    if with_place:
        evidence.append("event_places")
    if activity:
        evidence.append(activity)
    media_count = int(row.get("media_count") or 0)
    line_count = int(row.get("line_count") or 0)
    strength = "strong" if with_person and with_place and activity else "medium"
    if media_count == 0 and line_count == 0 and not with_place:
        strength = "weak"
    result = {
        "event_id": row.get("event_id"),
        "date": row.get("date"),
        "start_time": row.get("start_time"),
        "title": row.get("title"),
        "source": row.get("source"),
        "confidence": row.get("confidence"),
        "evidence_count": int(row.get("evidence_count") or 0),
        "media_count": media_count,
        "line_count": line_count,
        "evidence_types": evidence,
        "evidence_strength": strength,
        "confidence_label": "高" if strength == "strong" else "中" if strength == "medium" else "低",
        "source_counts": {"event_people": 1, "media_people": media_count, "line_speaker": line_count},
        "privacy_mode": _privacy_mode(public_mode),
    }
    if not public_mode:
        result["summary"] = row.get("summary")
        result["place_label"] = row.get("place_label")
    return result


def _place_event_result(row: dict[str, Any], *, public_mode: bool, activity: str | None) -> dict[str, Any]:
    evidence = ["event_places", "place"]
    if activity:
        evidence.append(activity)
    strength = "medium" if activity else "weak"
    result = {
        "event_id": row.get("event_id"),
        "date": row.get("date"),
        "start_time": row.get("start_time"),
        "title": row.get("title"),
        "source": row.get("source"),
        "confidence": row.get("confidence"),
        "place_label": _place_label(row, public_mode=public_mode, index=1),
        "evidence_types": evidence,
        "evidence_strength": strength,
        "confidence_label": "中" if strength == "medium" else "低",
        "source_counts": {"event_places": 1},
        "privacy_mode": _privacy_mode(public_mode),
    }
    if not public_mode:
        result["summary"] = row.get("summary")
    return result


def _person_intent(query: str, *, has_place: bool) -> str:
    if any(term in query for term in (*LINE_TERMS, *CALL_TERMS)):
        return "person_line_search"
    if any(term in query for term in PHOTO_TERMS):
        if "写" in query and not _activity_from_query(query):
            return "person_photo_search"
        return "person_photo_search" if not _activity_from_query(query) and not has_place else "person_activity_search"
    if has_place:
        return "person_place_search"
    if _activity_from_query(query):
        return "person_activity_search"
    return "person_event_search"


def _place_intent(query: str) -> str:
    if any(term in query for term in PHOTO_TERMS):
        return "place_photo_search"
    if _activity_from_query(query):
        return "place_activity_search"
    return "place_visit_search"


def _activity_from_query(query: str) -> str | None:
    if any(term in query for term in FOOD_ACTIVITY_TERMS):
        return "food"
    return None


def _looks_like_person_query(query: str) -> bool:
    return any(term in query for term in (*LINE_TERMS, *PHOTO_TERMS, "一緒", "ご飯", "食事", "カフェ"))


def _looks_like_place_query(query: str) -> bool:
    return any(term in query for term in (*VISIT_TERMS, *PHOTO_TERMS, "場所", "ご飯", "食事", "カフェ"))


def _looks_like_monthly_place_summary(query: str, entities: dict[str, Any]) -> bool:
    return bool(entities.get("date_from") and entities.get("date_to")) and any(term in query for term in PLACE_SUMMARY_TERMS)


def _extract_person_name(query: str, entities: dict[str, Any]) -> str | None:
    if entities.get("person"):
        return str(entities["person"])
    patterns = [
        r"([一-龥ぁ-んァ-ヶA-Za-z0-9_ー]{1,24})(?:さん)?が写",
        r"([一-龥ぁ-んァ-ヶA-Za-z0-9_ー]{1,24})(?:さん)?との?LINE",
        r"([一-龥ぁ-んァ-ヶA-Za-z0-9_ー]{1,24})(?:さん)?と(?:ご飯|食事|カフェ|一緒|新宿|[一-龥ぁ-んァ-ヶA-Za-z0-9_ー]{1,24}に)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            value = _clean_entity(match.group(1))
            if value and value not in {"誰", "何", "いつ", "どこ"}:
                return value
    return None


def _extract_place_name(query: str, entities: dict[str, Any], *, person_name: str | None) -> str | None:
    if entities.get("place"):
        value = str(entities["place"])
        if person_name and value.startswith(f"{person_name}と"):
            value = value[len(person_name) + 1 :]
        return _clean_entity(value)
    patterns = [
        r"(?:と|、)([一-龥ぁ-んァ-ヶA-Za-z0-9_ー]{1,24})(?:に|へ)(?:行った|行く|いた|いる|着いた|到着)",
        r"(?:と|、)([一-龥ぁ-んァ-ヶA-Za-z0-9_ー]{1,24})(?:で)(?:ご飯|食事|カフェ|食べ)",
        r"([一-龥ぁ-んァ-ヶA-Za-z0-9_ー]{1,24})の写真",
        r"([一-龥ぁ-んァ-ヶA-Za-z0-9_ー]{1,24})(?:に|で)(?:行った|ご飯|食事|カフェ)",
        r"([一-龥ぁ-んァ-ヶA-Za-z0-9_ー]{1,24})に行った",
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            value = _clean_entity(match.group(1))
            if value and value != person_name and value not in {"人物A", "人物B", "場所", "写真"}:
                return value
    raw_terms = entities.get("raw_terms") or []
    for term in raw_terms:
        value = _clean_entity(str(term))
        if (
            value
            and value != person_name
            and not _is_dateish(value)
            and value not in {"写真", "写っている", "写っている写真", "画像", "ご飯", "食事", "カフェ", "LINE", "何していた", "何して"}
        ):
            return value
    return None


def _clean_entity(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"[「」『』\"' 　?？。!！]", "", value)
    cleaned = re.sub(r"(いつ|どこ|何|なに|行った|行く|した|していた|日は|日|写真)$", "", cleaned)
    return cleaned or None


def _is_dateish(value: str) -> bool:
    return bool(re.search(r"\d{4}年|\d{4}-\d{2}|月\d{1,2}日|\d{1,2}月", value))


def _add_person_match(matches: dict[str, dict[str, Any]], row: dict[str, Any], source: str, *, public_mode: bool) -> None:
    person_id = str(row["id"])
    existing = matches.setdefault(person_id, dict(row))
    sources = set(existing.get("match_sources") or [])
    sources.add(source)
    existing["match_sources"] = sorted(sources)
    existing["person_label"] = _person_label(existing, public_mode=public_mode)
    if public_mode:
        existing.pop("display_name", None)


def _add_place_match(matches: dict[str, dict[str, Any]], row: dict[str, Any], source: str, *, public_mode: bool) -> None:
    place_id = str(row["id"])
    existing = matches.setdefault(place_id, dict(row))
    sources = set(existing.get("match_sources") or [])
    sources.add(source)
    existing["match_sources"] = sorted(sources)
    existing["place_label"] = _place_label(existing, public_mode=public_mode, index=len(matches))
    if public_mode:
        existing.pop("display_name", None)


def _person_label(person: dict[str, Any], *, public_mode: bool) -> str:
    if public_mode:
        return public_person_name(person, index=1) or "人物候補"
    return str(person.get("display_name") or person.get("public_name") or "人物候補")


def _place_label(place: dict[str, Any] | None, *, public_mode: bool, index: int) -> str:
    if not place:
        return ""
    if public_mode:
        return public_place_label(place)
    return str(place.get("display_name") or place.get("public_name") or place.get("category") or f"場所{index}")


def _ambiguous_answer(kind: str, candidates: list[dict[str, Any]]) -> str:
    lines = [f"{kind}候補が複数あります。勝手に1つへ確定しません。", "候補:"]
    for index, row in enumerate(candidates[:10], start=1):
        label = row.get("person_label") or row.get("place_label") or row.get("display_name") or row.get("public_name") or row.get("id")
        lines.append(f"- {index}. {label} ({row.get('id')})")
    return "\n".join(lines)


def _source_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in results:
        for source, count in (row.get("source_counts") or {}).items():
            counter[str(source)] += int(count or 0)
        for evidence in row.get("evidence_types") or []:
            counter[str(evidence)] += 1
    return dict(counter)


def _top_dates(results: list[dict[str, Any]]) -> list[str]:
    dates: list[str] = []
    for row in results:
        date_value = str(row.get("date") or row.get("captured_at") or "")[:10]
        if date_value and date_value not in dates:
            dates.append(date_value)
        if len(dates) >= 5:
            break
    return dates


def _overclaim_flags(answer: str) -> list[str]:
    return [term for term in FORBIDDEN_RELATIONSHIP_TERMS if term in answer]


def _privacy_mode(public_mode: bool) -> str:
    return "public" if public_mode else "private"


def _json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []
