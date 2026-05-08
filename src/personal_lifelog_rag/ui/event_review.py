"""Privacy-conscious services for reviewing and editing timeline events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from personal_lifelog_rag.core.privacy import redact_text


LINE_PREVIEW_CHARS = 80
SUMMARY_PREVIEW_CHARS = 120


def event_review_overview(repository, target_date: str) -> dict[str, Any]:
    events = repository.list_events(
        start_date=target_date,
        end_date=target_date,
        include_hidden=True,
        limit=1_000_000,
    )
    media_items = repository.list_media_items(
        start_date=target_date,
        end_date=target_date,
        limit=1_000_000,
    )
    line_messages = repository.list_line_messages(
        start_date=target_date,
        end_date=target_date,
        limit=1_000_000,
    )
    evidence_count = sum(int(event.get("event_evidence_count") or 0) for event in events)
    return {
        "date": target_date,
        "event_count": len(events),
        "photo_count": len(media_items),
        "line_count": len(line_messages),
        "event_evidence_count": evidence_count,
        "rows": [_event_row(event) for event in events],
        "event_ids": [str(event["id"]) for event in events],
    }


def event_detail(
    repository,
    event_id: str,
    *,
    line_limit: int = 10,
    photo_limit: int = 20,
) -> dict[str, Any]:
    event = repository.get_event(event_id, include_hidden=True)
    if event is None:
        return {
            "event": None,
            "line_evidence": [],
            "photo_evidence": [],
            "ocr_evidence": [],
            "vlm_evidence": [],
            "photo_gallery": [],
            "evidence_summary": "Event not found.",
        }

    evidence = repository.list_event_evidence(event_id)
    line_rows: list[dict[str, Any]] = []
    photo_rows: list[dict[str, Any]] = []
    ocr_rows: list[dict[str, Any]] = []
    vlm_rows: list[dict[str, Any]] = []
    gallery: list[tuple[str, str]] = []
    for row in evidence:
        evidence_type = str(row.get("evidence_type") or "")
        source_id = str(row.get("evidence_id") or "")
        if evidence_type in {"line", "line_message"} and len(line_rows) < line_limit:
            record = repository.get_embedding_record("line_message", source_id)
            if record:
                line_rows.append(_line_evidence_row(record))
        elif evidence_type in {"photo", "media_item"} and len(photo_rows) < photo_limit:
            record = repository.get_embedding_record("media_item", source_id)
            if record:
                photo_row = _photo_evidence_row(record, event)
                photo_rows.append(photo_row)
                ocr_record = repository.get_media_ocr(source_id)
                if ocr_record:
                    ocr_rows.append(_ocr_evidence_row(ocr_record))
                vlm_record = repository.get_media_vlm(source_id)
                if vlm_record:
                    vlm_rows.append(_vlm_evidence_row(vlm_record))
                thumbnail_path = str(record.get("thumbnail_path") or "")
                if thumbnail_path and Path(thumbnail_path).expanduser().exists():
                    gallery.append((thumbnail_path, photo_row["caption"]))

    line_count = sum(1 for row in evidence if row.get("evidence_type") in {"line", "line_message"})
    photo_count = sum(1 for row in evidence if row.get("evidence_type") in {"photo", "media_item"})
    return {
        "event": event,
        "line_evidence": line_rows,
        "photo_evidence": photo_rows,
        "ocr_evidence": ocr_rows,
        "vlm_evidence": vlm_rows,
        "photo_gallery": gallery,
        "evidence_summary": (
            f"evidence total={len(evidence)}, line={line_count}, photo={photo_count}"
        ),
    }


def save_event_review_override(
    repository,
    event_id: str,
    *,
    title: str | None = None,
    summary: str | None = None,
    location_name: str | None = None,
    tags: list[str] | str | None = None,
    is_verified: bool | None = None,
    is_hidden: bool | None = None,
    is_pinned: bool | None = None,
) -> dict[str, Any]:
    parsed_tags = _parse_tags(tags)
    override = repository.upsert_event_override(
        event_id,
        title_override=title,
        summary_override=summary,
        location_name_override=location_name,
        tags=parsed_tags,
        is_verified=is_verified,
        is_hidden=is_hidden,
        is_pinned=is_pinned,
    )
    return override


def clear_event_review_override(repository, event_id: str) -> int:
    return repository.delete_event_override(event_id)


def _event_row(event: dict[str, Any]) -> list[Any]:
    return [
        event.get("id") or "",
        _time_range(event),
        event.get("title") or "",
        redact_text(event.get("summary"), max_chars=SUMMARY_PREVIEW_CHARS),
        event.get("confidence"),
        event.get("location_name") or "",
        int(event.get("event_evidence_count") or 0),
        int(event.get("line_evidence_count") or 0),
        int(event.get("photo_evidence_count") or 0),
        bool(event.get("is_verified")),
        bool(event.get("is_pinned")),
        bool(event.get("is_hidden")),
        bool(event.get("title_override") or event.get("summary_override") or event.get("location_name_override")),
    ]


def _line_evidence_row(record: dict[str, Any]) -> dict[str, str]:
    return {
        "sent_at": str(record.get("sent_at") or ""),
        "sender": redact_text(str(record.get("sender") or record.get("sender_name") or ""), max_chars=24),
        "text": redact_text(str(record.get("text") or record.get("message_text") or ""), max_chars=LINE_PREVIEW_CHARS),
        "message_type": str(record.get("message_type") or ""),
    }


def _photo_evidence_row(record: dict[str, Any], event: dict[str, Any]) -> dict[str, str]:
    has_gps = record.get("gps_lat") is not None and record.get("gps_lon") is not None
    thumbnail_path = str(record.get("thumbnail_path") or "")
    return {
        "thumbnail_path": thumbnail_path,
        "captured_at": str(record.get("captured_at") or record.get("fallback_captured_at") or ""),
        "file_name": redact_text(str(record.get("file_name") or ""), max_chars=80),
        "gps": "GPSあり" if has_gps else "GPSなし",
        "location_name": str(event.get("location_name") or ""),
        "caption": redact_text(str(record.get("file_name") or "photo"), max_chars=80),
    }


def _ocr_evidence_row(record: dict[str, Any]) -> dict[str, str]:
    text = record.get("ocr_text_redacted") or record.get("ocr_text") or ""
    return {
        "media_id": str(record.get("media_id") or ""),
        "captured_at": str(record.get("captured_at") or record.get("fallback_captured_at") or ""),
        "file_name": redact_text(str(record.get("file_name") or ""), max_chars=80),
        "status": str(record.get("status") or ""),
        "engine": str(record.get("ocr_engine") or ""),
        "text": redact_text(str(text), max_chars=160),
    }


def _vlm_evidence_row(record: dict[str, Any]) -> dict[str, str]:
    return {
        "media_id": str(record.get("media_id") or ""),
        "captured_at": str(record.get("captured_at") or record.get("fallback_captured_at") or ""),
        "file_name": redact_text(str(record.get("file_name") or ""), max_chars=80),
        "status": str(record.get("status") or ""),
        "engine": str(record.get("vlm_engine") or ""),
        "caption": redact_text(str(record.get("short_caption") or record.get("caption") or ""), max_chars=160),
        "scene_tags": _join_json_list(record.get("scene_tags_json")),
        "object_tags": _join_json_list(record.get("object_tags_json")),
        "activity_tags": _join_json_list(record.get("activity_tags_json")),
        "food_cues": _join_json_list(record.get("food_cues_json")),
        "location_cues": _join_json_list(record.get("location_cues_json")),
        "safety_flags": _join_json_list(record.get("safety_flags_json")),
    }


def _join_json_list(value: Any) -> str:
    if not value:
        return ""
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return redact_text(str(value), max_chars=80)
    if isinstance(parsed, list):
        return ", ".join(redact_text(str(item), max_chars=32) for item in parsed)
    return redact_text(str(parsed), max_chars=80)


def _time_range(event: dict[str, Any]) -> str:
    start = str(event.get("start_time") or "--:--")
    end = str(event.get("end_time") or "--:--")
    return f"{start[:5]}〜{end[:5]}"


def _parse_tags(value: list[str] | str | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [part.strip() for part in text.replace("、", ",").split(",") if part.strip()]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [str(parsed).strip()] if str(parsed).strip() else []
