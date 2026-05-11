"""Conservative score components for multimodal image search."""

from __future__ import annotations

import json
from typing import Any


VLM_TEXT_KEYS = (
    "caption",
    "short_caption",
    "scene_tags_json",
    "object_tags_json",
    "activity_tags_json",
    "food_cues_json",
    "location_cues_json",
    "text_cues_json",
)
OCR_KEYS = ("ocr_text", "ocr_text_redacted")
PLACE_KEYS = (
    "location_name",
    "effective_location_name",
    "location_name_override",
    "place_display_name",
    "place_public_name",
    "place_category",
    "place_aliases_json",
)
VISUAL_MATCH_EMBEDDING_THRESHOLD = 0.35


def score_multimodal_components(
    row: dict[str, Any],
    *,
    expanded_terms: list[str],
    embedding_score: float,
    related_event: dict[str, Any] | None,
    line_matches: list[dict[str, Any]],
    sql_score: float = 0.0,
    matched_terms: list[str] | None = None,
    visual_match_embedding_threshold: float = VISUAL_MATCH_EMBEDDING_THRESHOLD,
    query_intent: str | None = None,
    specific_terms: list[str] | None = None,
    generic_terms: list[str] | None = None,
) -> dict[str, float]:
    """Return normalized score components and final hybrid score.

    Weights follow the PR35 starting point:
    embedding 0.30, VLM text 0.20, OCR 0.15, LINE 0.15, event 0.15, place 0.05.
    """

    is_specific_food = query_intent == "specific_food_search" or bool(specific_terms)
    specific_terms = [term for term in (specific_terms or []) if str(term).strip()]
    generic_terms = [term for term in (generic_terms or []) if str(term).strip()]

    raw_vlm_text_score = _field_match_score(row, VLM_TEXT_KEYS, expanded_terms, max_score=1.0)
    specific_food_score = _field_match_score(row, VLM_TEXT_KEYS + OCR_KEYS, specific_terms, max_score=1.0) if specific_terms else 0.0
    generic_food_score = _field_match_score(row, VLM_TEXT_KEYS + OCR_KEYS, generic_terms, max_score=1.0) if generic_terms else 0.0
    vlm_text_score = raw_vlm_text_score
    effective_sql_score = sql_score
    specific_visual_match = False
    if is_specific_food:
        specific_visual_match = bool(specific_food_score > 0 or _terms_in_keys(row, specific_terms, VLM_TEXT_KEYS + OCR_KEYS + ("file_name",)))
        if specific_food_score <= 0:
            # Generic food words such as meal/bowl/noodle are useful recall
            # signals, but they must not make a ramen/soba/omurice query look
            # visually confirmed on their own.
            vlm_text_score = min(vlm_text_score, 0.2 if generic_food_score else 0.0)
            effective_sql_score = min(sql_score, 0.2 if generic_food_score else 0.0)

    ocr_score = _field_match_score(row, OCR_KEYS, expanded_terms, max_score=1.0)
    line_score = min(1.0, len(line_matches) * 0.35)
    event_score = _event_score(related_event, expanded_terms)
    place_score = _place_score(row, related_event, expanded_terms)
    override_boost = _override_boost(row, related_event)
    safety_penalty = _safety_penalty(row)
    if is_specific_food:
        visual_match = specific_visual_match
    else:
        visual_match = has_visual_match(
            embedding_score=embedding_score,
            vlm_text_score=vlm_text_score,
            ocr_score=ocr_score,
            sql_score=effective_sql_score,
            matched_terms=matched_terms or [],
            embedding_threshold=visual_match_embedding_threshold,
        )
    if not visual_match:
        line_score = min(line_score, 0.2)
        event_score = min(event_score, 0.2)

    final = (
        max(0.0, min(embedding_score, 1.0)) * 0.30
        + vlm_text_score * 0.20
        + ocr_score * 0.15
        + line_score * 0.15
        + event_score * 0.15
        + place_score * 0.05
        + override_boost
        + safety_penalty
    )
    if is_specific_food and specific_food_score > 0:
        final += min(0.18, specific_food_score * 0.18)

    # Visual-only candidates are useful but should not dominate activity/date
    # answers without LINE/OCR/event/place support.
    has_context = any([ocr_score, line_score, event_score, place_score])
    if embedding_score and not has_context and not vlm_text_score:
        final = min(final, 0.44)
    if vlm_text_score and not has_context and not embedding_score:
        final = min(final, 0.44)
    if _people_present_only(row) and not has_context:
        final = min(final, 0.44)
    if not visual_match:
        final *= 0.5

    return {
        "sql_score": round(max(0.0, min(effective_sql_score, 1.0)), 3),
        "embedding_score": round(max(0.0, min(embedding_score, 1.0)), 3),
        "vlm_text_score": round(vlm_text_score, 3),
        "specific_food_score": round(specific_food_score, 3),
        "generic_food_score": round(generic_food_score, 3),
        "specific_visual_match": 1.0 if specific_visual_match else 0.0,
        "ocr_score": round(ocr_score, 3),
        "line_score": round(line_score, 3),
        "event_score": round(event_score, 3),
        "place_score": round(place_score, 3),
        "override_boost": round(override_boost, 3),
        "safety_penalty": round(safety_penalty, 3),
        "visual_match": 1.0 if visual_match else 0.0,
        "visual_match_threshold": round(visual_match_embedding_threshold, 3),
        "final_score": round(max(0.0, min(final, 0.95)), 3),
    }


def has_visual_match(
    *,
    embedding_score: float,
    vlm_text_score: float,
    ocr_score: float = 0.0,
    sql_score: float = 0.0,
    matched_terms: list[str] | None = None,
    embedding_threshold: float = VISUAL_MATCH_EMBEDDING_THRESHOLD,
) -> bool:
    """Return whether an image result visibly matches the visual query itself."""

    return bool(
        vlm_text_score > 0
        or ocr_score > 0
        or sql_score > 0
        or (matched_terms or [])
        or embedding_score >= embedding_threshold
    )


def matched_terms_for_row(row: dict[str, Any], terms: list[str]) -> list[str]:
    haystack = _joined_fields(row, VLM_TEXT_KEYS + OCR_KEYS + PLACE_KEYS + ("file_name",))
    matched: list[str] = []
    for term in terms:
        term = str(term or "").strip()
        if term and term.lower() in haystack and term not in matched:
            matched.append(term)
    return matched[:20]


def matched_visual_terms_for_row(row: dict[str, Any], terms: list[str]) -> list[str]:
    """Return matches from visual/OCR fields only, excluding event/place context."""

    return _terms_in_keys(row, terms, VLM_TEXT_KEYS + OCR_KEYS + ("file_name",))[:20]


def matched_fields_for_row(row: dict[str, Any], terms: list[str], *, has_embedding: bool) -> list[str]:
    fields = ["embedding"] if has_embedding else []
    for name, keys in {
        "caption": ("caption",),
        "short_caption": ("short_caption",),
        "scene_tags": ("scene_tags_json",),
        "object_tags": ("object_tags_json",),
        "activity_tags": ("activity_tags_json",),
        "food_cues": ("food_cues_json",),
        "location_cues": ("location_cues_json",),
        "text_cues": ("text_cues_json",),
        "ocr": OCR_KEYS,
        "place": PLACE_KEYS,
        "file_name": ("file_name",),
    }.items():
        if _field_match_score(row, keys, terms, max_score=1.0) > 0:
            fields.append(name)
    return fields or ["candidate"]


def _event_score(event: dict[str, Any] | None, terms: list[str]) -> float:
    if not event:
        return 0.0
    confidence = max(0.0, min(float(event.get("confidence") or 0.0), 1.0))
    score = 0.35 + confidence * 0.35
    if _field_match_score(event, ("title", "summary", "effective_title", "effective_summary", "tags_json"), terms, max_score=1.0):
        score += 0.2
    return min(score, 1.0)


def _place_score(row: dict[str, Any], event: dict[str, Any] | None, terms: list[str]) -> float:
    score = 0.0
    if row.get("gps_lat") is not None and row.get("gps_lon") is not None:
        score = max(score, 0.3)
    if event and event.get("location_name"):
        score = max(score, 0.4)
    if event and _field_match_score(event, ("location_name", "effective_location_name"), terms, max_score=1.0):
        score = max(score, 1.0)
    if _field_match_score(row, PLACE_KEYS, terms, max_score=1.0):
        score = max(score, 1.0)
    return score


def _override_boost(row: dict[str, Any], event: dict[str, Any] | None) -> float:
    boost = 0.0
    if row.get("is_verified") or row.get("review_status") == "accepted":
        boost += 0.08
    if event and event.get("is_verified"):
        boost += 0.1
    if event and event.get("is_pinned"):
        boost += 0.12
    return boost


def _safety_penalty(row: dict[str, Any]) -> float:
    penalty = 0.0
    flags = set(_json_list(row.get("safety_flags_json")))
    if flags & {"forbidden_terms_found", "relationship_inference_removed", "emotion_inference_removed"}:
        penalty -= 0.1
    if row.get("is_wrong") or row.get("review_status") in {"rejected", "wrong"}:
        penalty -= 0.5
    return penalty


def _field_match_score(row: dict[str, Any], keys: tuple[str, ...], terms: list[str], *, max_score: float) -> float:
    field_matches = 0
    term_matches: set[str] = set()
    for key in keys:
        value = _stringify(row.get(key)).lower()
        if not value:
            continue
        matched_here = False
        for term in terms:
            normalized = str(term or "").strip().lower()
            if normalized and normalized in value:
                term_matches.add(normalized)
                matched_here = True
        if matched_here:
            field_matches += 1
    if not field_matches:
        return 0.0
    return min(max_score, 0.22 * field_matches + 0.08 * len(term_matches))


def _joined_fields(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    return " ".join(_stringify(row.get(key)).lower() for key in keys)


def _terms_in_keys(row: dict[str, Any], terms: list[str], keys: tuple[str, ...]) -> list[str]:
    haystack = _joined_fields(row, keys)
    matched: list[str] = []
    for term in terms:
        value = str(term or "").strip()
        if value and value.lower() in haystack and value not in matched:
            matched.append(value)
    return matched


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    text = str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(parsed, list):
        return " ".join(str(item) for item in parsed)
    return str(parsed)


def _json_list(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw]
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return [str(raw)]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    return [str(parsed)] if str(parsed).strip() else []


def _people_present_only(row: dict[str, Any]) -> bool:
    flags = set(_json_list(row.get("safety_flags_json")))
    if "people_present" not in flags:
        return False
    return not any(
        _stringify(row.get(key)).strip()
        for key in ("food_cues_json", "location_cues_json", "text_cues_json", "ocr_text", "ocr_text_redacted")
    )
