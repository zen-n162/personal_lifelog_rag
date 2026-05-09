"""Human review helpers for local VLM outputs.

The review layer never calls a model. It only stores local user decisions and
returns effective VLM metadata for search/event generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from personal_lifelog_rag.core.privacy import redact_text


VALID_REVIEW_STATUSES = {"unreviewed", "accepted", "rejected", "needs_fix", "wrong"}


@dataclass(frozen=True)
class VlmReviewFilters:
    date: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    review_status: str | None = None
    unreviewed: bool = False
    safety_flags: bool = False
    people_present: bool = False
    low_confidence: float | None = None
    food_cues: bool = False
    location_cues: bool = False
    has_ocr: bool = False
    has_embedding: bool = False
    vlm_only: bool = False
    hidden: bool | None = None
    wrong: bool | None = None
    is_searchable: bool | None = None
    is_event_usable: bool | None = None
    event_linked: bool | None = None
    limit: int = 100


@dataclass(frozen=True)
class VlmOverrideUpdate:
    media_id: str
    caption_override: str | None = None
    short_caption_override: str | None = None
    scene_tags_override: list[str] | None = None
    object_tags_override: list[str] | None = None
    activity_tags_override: list[str] | None = None
    food_cues_override: list[str] | None = None
    location_cues_override: list[str] | None = None
    is_verified: bool | None = None
    is_hidden: bool | None = None
    is_wrong: bool | None = None
    is_searchable: bool | None = None
    is_event_usable: bool | None = None
    review_status: str | None = None
    review_note: str | None = None


def list_vlm_review_items(repository, filters: VlmReviewFilters | None = None) -> list[dict[str, Any]]:
    filters = filters or VlmReviewFilters()
    start = filters.date or filters.date_from
    end = filters.date or filters.date_to
    rows = repository.list_media_vlm(start_date=start, end_date=end, limit=100_000)
    embedding_ids = _embedding_media_ids(repository) if filters.has_embedding else set()
    event_ids = _event_media_ids(repository) if filters.event_linked is not None or filters.vlm_only else set()
    output: list[dict[str, Any]] = []
    for row in rows:
        effective = apply_vlm_override_to_result(row)
        if not _matches_filters(effective, filters, embedding_ids=embedding_ids, event_ids=event_ids):
            continue
        output.append(_review_item(effective, has_embedding=str(effective.get("media_id")) in embedding_ids, event_linked=str(effective.get("media_id")) in event_ids))
    return output[: max(filters.limit, 0)]


def get_vlm_review_detail(repository, media_id: str) -> dict[str, Any] | None:
    row = repository.get_media_vlm(media_id)
    if not row:
        return None
    effective = apply_vlm_override_to_result(row)
    events = []
    for event in repository.list_events(start_date=str(effective.get("captured_at") or effective.get("fallback_captured_at") or "")[:10], end_date=str(effective.get("captured_at") or effective.get("fallback_captured_at") or "")[:10], include_hidden=True, limit=500):
        evidence = repository.list_event_evidence(str(event["id"]))
        if any(item.get("evidence_type") in {"photo", "vlm", "ocr"} and item.get("evidence_id") == media_id for item in evidence):
            events.append(
                {
                    "event_id": event["id"],
                    "title": event.get("effective_title") or event.get("title"),
                    "start_time": event.get("start_time"),
                    "end_time": event.get("end_time"),
                }
            )
    effective["related_events"] = events
    return effective


def save_vlm_override(repository, update: VlmOverrideUpdate) -> dict[str, Any]:
    status = update.review_status
    if status and status not in VALID_REVIEW_STATUSES:
        raise ValueError(f"Unknown VLM review_status: {status}")
    repository.upsert_media_vlm_override(
        media_id=update.media_id,
        caption_override=_blank_to_none(update.caption_override),
        short_caption_override=_blank_to_none(update.short_caption_override),
        scene_tags_override=update.scene_tags_override,
        object_tags_override=update.object_tags_override,
        activity_tags_override=update.activity_tags_override,
        food_cues_override=update.food_cues_override,
        location_cues_override=update.location_cues_override,
        is_verified=update.is_verified,
        is_hidden=update.is_hidden,
        is_wrong=update.is_wrong,
        is_searchable=update.is_searchable,
        is_event_usable=update.is_event_usable,
        review_status=status,
        review_note=update.review_note,
    )
    return get_vlm_review_detail(repository, update.media_id) or {"media_id": update.media_id}


def bulk_update_vlm_overrides(
    repository,
    media_ids: list[str],
    *,
    review_status: str | None = None,
    is_verified: bool | None = None,
    is_hidden: bool | None = None,
    is_wrong: bool | None = None,
    is_searchable: bool | None = None,
    is_event_usable: bool | None = None,
    add_tags: list[str] | None = None,
) -> dict[str, Any]:
    updated = 0
    for media_id in [value.strip() for value in media_ids if value.strip()]:
        existing = get_vlm_review_detail(repository, media_id) or {"media_id": media_id}
        food_tags = _json_list(existing.get("effective_food_cues_json") or existing.get("food_cues_json"))
        if add_tags:
            for tag in add_tags:
                if tag and tag not in food_tags:
                    food_tags.append(tag)
        save_vlm_override(
            repository,
            VlmOverrideUpdate(
                media_id=media_id,
                food_cues_override=food_tags if add_tags else None,
                is_verified=is_verified,
                is_hidden=is_hidden,
                is_wrong=is_wrong,
                is_searchable=is_searchable,
                is_event_usable=is_event_usable,
                review_status=review_status,
            ),
        )
        updated += 1
    return {"updated": updated}


def clear_vlm_override(repository, media_id: str) -> int:
    return repository.delete_media_vlm_override(media_id)


def get_effective_vlm_result(repository, media_id: str) -> dict[str, Any] | None:
    row = repository.get_media_vlm(media_id)
    return apply_vlm_override_to_result(row) if row else None


def apply_vlm_override_to_result(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    output["review_status"] = str(row.get("vlm_review_status") or row.get("review_status") or "unreviewed")
    output["is_verified"] = int(row.get("vlm_is_verified") or row.get("is_verified") or 0)
    output["is_hidden"] = int(row.get("vlm_is_hidden") or row.get("is_hidden") or 0)
    output["is_wrong"] = int(row.get("vlm_is_wrong") or row.get("is_wrong") or 0)
    output["is_searchable"] = int(row.get("vlm_is_searchable") if row.get("vlm_is_searchable") is not None else row.get("is_searchable") if row.get("is_searchable") is not None else 1)
    output["is_event_usable"] = int(row.get("vlm_is_event_usable") if row.get("vlm_is_event_usable") is not None else row.get("is_event_usable") if row.get("is_event_usable") is not None else 1)
    output["caption"] = row.get("caption_override") or row.get("caption")
    output["short_caption"] = row.get("short_caption_override") or row.get("short_caption")
    for target, override_key, base_key in (
        ("scene_tags_json", "scene_tags_override_json", "scene_tags_json"),
        ("object_tags_json", "object_tags_override_json", "object_tags_json"),
        ("activity_tags_json", "activity_tags_override_json", "activity_tags_json"),
        ("food_cues_json", "food_cues_override_json", "food_cues_json"),
        ("location_cues_json", "location_cues_override_json", "location_cues_json"),
    ):
        output[target] = row.get(override_key) or row.get(base_key)
        output[f"effective_{target}"] = output[target]
    output["review_note"] = row.get("vlm_review_note") or row.get("review_note")
    return output


def should_use_vlm_for_search(row: dict[str, Any], *, include_hidden: bool = False) -> bool:
    effective = apply_vlm_override_to_result(row)
    if _is_untrusted_vlm_row(effective):
        return False
    if include_hidden:
        return True
    if effective.get("is_hidden") or effective.get("is_wrong") or not effective.get("is_searchable"):
        return False
    return effective.get("review_status") not in {"rejected", "wrong"}


def should_use_vlm_for_events(row: dict[str, Any]) -> bool:
    effective = apply_vlm_override_to_result(row)
    if _is_untrusted_vlm_row(effective):
        return False
    if effective.get("is_hidden") or effective.get("is_wrong") or not effective.get("is_event_usable"):
        return False
    return effective.get("review_status") not in {"rejected", "wrong"}


def _is_untrusted_vlm_row(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "").lower()
    engine = str(row.get("vlm_engine") or row.get("engine") or "").lower()
    model = str(row.get("model_name") or "").lower()
    if status and status != "success":
        return True
    return "fake" in engine or "fake" in model


def generate_vlm_eval_case(*, media_id: str | None = None, query: str | None = None, expected_media_id: str | None = None) -> str:
    if query:
        expected = expected_media_id or media_id or "MEDIA_ID"
        case_id = "manual_vlm_search_" + _slug(expected)
        return "\n".join(
            [
                f"- id: {case_id}",
                "  type: image_search",
                f"  query: \"{query}\"",
                "  expected_evidence_types:",
                "    - \"vlm\"",
                "  should_exclude_media_ids: []",
                "  should_not_include:",
                "    - \"確実に\"",
                f"  expected_media_ids:",
                f"    - \"{expected}\"",
            ]
        )
    media = media_id or "MEDIA_ID"
    return "\n".join(
        [
            f"- id: vlm_review_{_slug(media)}",
            "  type: vlm_review",
            f"  media_id: \"{media}\"",
            "  expected_hidden: false",
            "  expected_searchable: true",
        ]
    )


def format_vlm_review_queue(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "VLM review queue: no records"
    lines = [f"VLM review queue: {len(rows)} record(s)"]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"{index}. {row['media_id']} {row.get('captured_at') or ''} "
            f"status={row['review_status']} verified={row['is_verified']} "
            f"hidden={row['is_hidden']} searchable={row['is_searchable']} "
            f"caption={redact_text(row.get('short_caption') or row.get('caption'), max_chars=80)}"
        )
    return "\n".join(lines)


def review_rows_for_dataframe(rows: list[dict[str, Any]]) -> list[list[Any]]:
    return [
        [
            row.get("media_id"),
            row.get("captured_at") or row.get("fallback_captured_at") or "",
            row.get("file_name") or "",
            row.get("thumbnail_path") or "",
            row.get("short_caption") or row.get("caption") or "",
            row.get("confidence"),
            ", ".join(row.get("safety_flags") or []),
            ", ".join(row.get("scene_tags") or []),
            ", ".join(row.get("activity_tags") or []),
            ", ".join(row.get("food_cues") or []),
            ", ".join(row.get("location_cues") or []),
            row.get("review_status"),
            row.get("is_verified"),
            row.get("is_hidden"),
            row.get("is_wrong"),
            row.get("is_searchable"),
            row.get("is_event_usable"),
        ]
        for row in rows
    ]


def _review_item(row: dict[str, Any], *, has_embedding: bool, event_linked: bool) -> dict[str, Any]:
    return {
        "media_id": row.get("media_id"),
        "captured_at": row.get("captured_at") or row.get("fallback_captured_at") or "",
        "file_name": row.get("file_name") or "",
        "thumbnail_path": row.get("thumbnail_path") or "",
        "caption": row.get("caption") or "",
        "short_caption": row.get("short_caption") or "",
        "confidence": row.get("confidence"),
        "safety_flags": _json_list(row.get("safety_flags_json")),
        "scene_tags": _json_list(row.get("scene_tags_json")),
        "object_tags": _json_list(row.get("object_tags_json")),
        "activity_tags": _json_list(row.get("activity_tags_json")),
        "food_cues": _json_list(row.get("food_cues_json")),
        "location_cues": _json_list(row.get("location_cues_json")),
        "people_count": row.get("people_count"),
        "contains_text_hint": bool(row.get("contains_text_hint")),
        "evidence_strength": row.get("evidence_strength") or "weak",
        "ocr_preview": redact_text(row.get("ocr_text_redacted") or row.get("ocr_text"), max_chars=180),
        "review_status": row.get("review_status") or "unreviewed",
        "review_note": row.get("review_note") or "",
        "is_verified": int(row.get("is_verified") or 0),
        "is_hidden": int(row.get("is_hidden") or 0),
        "is_wrong": int(row.get("is_wrong") or 0),
        "is_searchable": int(row.get("is_searchable") if row.get("is_searchable") is not None else 1),
        "is_event_usable": int(row.get("is_event_usable") if row.get("is_event_usable") is not None else 1),
        "has_embedding": has_embedding,
        "event_linked": event_linked,
    }


def _matches_filters(row: dict[str, Any], filters: VlmReviewFilters, *, embedding_ids: set[str], event_ids: set[str]) -> bool:
    item = _review_item(row, has_embedding=str(row.get("media_id")) in embedding_ids, event_linked=str(row.get("media_id")) in event_ids)
    if filters.unreviewed and item["review_status"] != "unreviewed":
        return False
    if filters.review_status and item["review_status"] != filters.review_status:
        return False
    if filters.safety_flags and not item["safety_flags"]:
        return False
    if filters.people_present and "people_present" not in item["safety_flags"]:
        return False
    if filters.low_confidence is not None and (item["confidence"] is None or float(item["confidence"]) > filters.low_confidence):
        return False
    if filters.food_cues and not item["food_cues"]:
        return False
    if filters.location_cues and not item["location_cues"]:
        return False
    if filters.has_ocr and not (row.get("ocr_text") or row.get("ocr_text_redacted")):
        return False
    if filters.has_embedding and str(row.get("media_id")) not in embedding_ids:
        return False
    if filters.vlm_only and (row.get("ocr_text") or str(row.get("media_id")) in event_ids):
        return False
    for name in ("hidden", "wrong", "is_searchable", "is_event_usable", "event_linked"):
        expected = getattr(filters, name)
        if expected is not None and bool(item[name]) != bool(expected):
            return False
    return True


def _embedding_media_ids(repository) -> set[str]:
    try:
        from personal_lifelog_rag.embeddings.repository import MediaEmbeddingRepository

        rows = MediaEmbeddingRepository(repository.db_path).list_embeddings(statuses=["success"], limit=100_000)
    except Exception:
        return set()
    return {str(row.get("media_id")) for row in rows if row.get("media_id")}


def _event_media_ids(repository) -> set[str]:
    ids: set[str] = set()
    for event in repository.list_events(include_hidden=True, limit=100_000):
        for row in repository.list_event_evidence(str(event["id"])):
            if row.get("evidence_type") in {"photo", "vlm", "ocr"} and row.get("evidence_id"):
                ids.add(str(row["evidence_id"]))
    return ids


def _json_list(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item)]
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return [part.strip() for part in str(raw).replace("\n", ",").split(",") if part.strip()]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    return []


def parse_tag_text(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).replace("\n", ",").split(",") if part.strip()]


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value)[:40].strip("_") or "case"
