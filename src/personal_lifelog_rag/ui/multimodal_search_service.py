"""UI-facing helpers for multimodal search and VLM review handoff."""

from __future__ import annotations

import json
from typing import Any

from personal_lifelog_rag.core.privacy import redact_text
from personal_lifelog_rag.db.repository import connect
from personal_lifelog_rag.db.schema import initialize_schema
from personal_lifelog_rag.embeddings.multimodal_search import format_multimodal_search, multimodal_search
from personal_lifelog_rag.embeddings.schemas import MultimodalSearchOptions
from personal_lifelog_rag.vlm.review_service import (
    VlmOverrideUpdate,
    apply_vlm_override_to_result,
    get_vlm_review_detail,
    save_vlm_override,
)


def multimodal_search_for_ui(
    repository,
    *,
    query: str,
    backend: str = "hybrid",
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 10,
    include_hidden: bool = False,
) -> dict[str, Any]:
    report = multimodal_search(
        repository,
        MultimodalSearchOptions(
            query=query,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            backend=backend,  # type: ignore[arg-type]
            include_hidden=include_hidden,
        ),
    )
    rows = [_result_row(index, row) for index, row in enumerate(report["results"], start=1)]
    return {
        "report": report,
        "summary_text": format_multimodal_search(report),
        "rows": rows,
        "media_ids": [row[1] for row in rows],
    }


def search_result_detail_for_ui(repository, media_id: str | None) -> dict[str, Any]:
    if not media_id:
        return _empty_detail()
    vlm = repository.get_media_vlm(media_id) or {}
    if vlm:
        vlm = apply_vlm_override_to_result(vlm)
    ocr = repository.get_media_ocr(media_id) or {}
    media = _media_item(repository, media_id)
    detail = get_vlm_review_detail(repository, media_id) or {}
    timestamp = str(
        vlm.get("captured_at")
        or ocr.get("captured_at")
        or media.get("captured_at")
        or media.get("fallback_captured_at")
        or ""
    )
    events = detail.get("related_events") or _related_events(repository, media_id, timestamp[:10])
    related_people = _related_people(repository, media_id)
    return {
        "media_id": media_id,
        "thumbnail_path": vlm.get("thumbnail_path") or ocr.get("thumbnail_path") or media.get("thumbnail_path") or "",
        "file_name": redact_text(vlm.get("file_name") or ocr.get("file_name") or media.get("file_name"), max_chars=80),
        "captured_at": timestamp,
        "caption": redact_text(vlm.get("caption") or vlm.get("short_caption"), max_chars=600),
        "ocr_text": redact_text(ocr.get("ocr_text_redacted") or ocr.get("ocr_text"), max_chars=600),
        "score_components": "",
        "evidence": _evidence_text(vlm, ocr, events, related_people),
        "review_status": str(vlm.get("review_status") or detail.get("review_status") or "unreviewed"),
        "events": events,
    }


def detail_values(detail: dict[str, Any]) -> tuple[str, str, str, str, str, str, str, str]:
    return (
        str(detail.get("thumbnail_path") or ""),
        str(detail.get("file_name") or ""),
        str(detail.get("captured_at") or ""),
        str(detail.get("caption") or ""),
        str(detail.get("ocr_text") or ""),
        str(detail.get("evidence") or ""),
        str(detail.get("review_status") or ""),
        str(detail.get("score_components") or ""),
    )


def mark_search_result_for_review(repository, media_id: str | None, action: str) -> dict[str, Any]:
    if not media_id:
        return _empty_detail()
    kwargs: dict[str, Any] = {"media_id": media_id}
    if action == "accepted":
        kwargs.update(review_status="accepted", is_verified=True, is_hidden=False, is_wrong=False, is_searchable=True, is_event_usable=True)
    elif action == "wrong":
        kwargs.update(review_status="wrong", is_wrong=True, is_searchable=False, is_event_usable=False)
    elif action == "hidden":
        kwargs.update(is_hidden=True)
    elif action == "not_searchable":
        kwargs.update(is_searchable=False)
    elif action == "not_event_usable":
        kwargs.update(is_event_usable=False)
    save_vlm_override(repository, VlmOverrideUpdate(**kwargs))
    return search_result_detail_for_ui(repository, media_id)


def _result_row(index: int, row: dict[str, Any]) -> list[Any]:
    return [
        index,
        row.get("media_id") or "",
        row.get("date") or "",
        row.get("captured_at") or "",
        row.get("score_components", {}).get("final_score", 0.0),
        row.get("confidence_label") or "",
        row.get("evidence_strength") or "",
        row.get("caption") or "",
        ", ".join(row.get("matched_terms") or []),
        ", ".join(row.get("food_cues") or []),
        ", ".join(row.get("location_cues") or []),
        row.get("related_event") or "",
        ", ".join(row.get("related_persons") or []),
        ", ".join(row.get("person_evidence_types") or []),
        row.get("thumbnail_path") or "",
        json.dumps(row.get("score_components") or {}, ensure_ascii=False, sort_keys=True),
        row.get("review_status") or "unreviewed",
    ]


def _media_item(repository, media_id: str) -> dict[str, Any]:
    for row in repository.list_media_items(limit=1_000_000):
        if str(row.get("id")) == media_id:
            return row
    return {}


def _related_events(repository, media_id: str, date_value: str) -> list[dict[str, Any]]:
    if not date_value:
        return []
    events = []
    for event in repository.list_events(start_date=date_value, end_date=date_value, include_hidden=True, limit=500):
        evidence = repository.list_event_evidence(str(event["id"]))
        if any(row.get("evidence_type") in {"photo", "vlm", "ocr"} and row.get("evidence_id") == media_id for row in evidence):
            events.append({"event_id": event.get("id"), "title": event.get("title"), "start_time": event.get("start_time")})
    return events


def _related_people(repository, media_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        for row in connection.execute(
            """
            SELECT persons.display_name, persons.public_name, media_people.source, media_people.confidence
            FROM media_people
            JOIN persons ON persons.id = media_people.person_id
            WHERE media_people.media_id = ?
              AND media_people.verified_by_user = 1
              AND COALESCE(media_people.hidden, 0) = 0
              AND persons.manual_verified = 1
              AND COALESCE(persons.hidden, 0) = 0
              AND persons.deleted_at IS NULL
            ORDER BY media_people.confidence DESC, persons.display_name ASC
            LIMIT 5
            """,
            (media_id,),
        ).fetchall():
            rows.append(dict(row))
    return rows


def _evidence_text(vlm: dict[str, Any], ocr: dict[str, Any], events: list[dict[str, Any]], related_people: list[dict[str, Any]] | None = None) -> str:
    lines = []
    if vlm:
        lines.append("VLM: " + redact_text(vlm.get("short_caption") or vlm.get("caption"), max_chars=180))
        if vlm.get("food_cues_json"):
            lines.append("food_cues: " + str(vlm.get("food_cues_json")))
        if vlm.get("location_cues_json"):
            lines.append("location_cues: " + str(vlm.get("location_cues_json")))
    if ocr:
        lines.append("OCR: " + redact_text(ocr.get("ocr_text_redacted") or ocr.get("ocr_text"), max_chars=180))
    if events:
        lines.append("related events: " + ", ".join(str(event.get("title") or event.get("event_id")) for event in events[:5]))
    if related_people:
        labels = [str(row.get("display_name") or row.get("public_name") or "人物候補") for row in related_people[:5]]
        sources = sorted({str(row.get("source") or "manual") for row in related_people})
        lines.append("related persons: " + ", ".join(labels))
        lines.append("person evidence: " + ", ".join(sources) + " (manual links only)")
    return "\n".join(line for line in lines if line.strip())


def _empty_detail() -> dict[str, Any]:
    return {
        "media_id": "",
        "thumbnail_path": "",
        "file_name": "",
        "captured_at": "",
        "caption": "",
        "ocr_text": "",
        "score_components": "",
        "evidence": "",
        "review_status": "",
        "events": [],
    }
