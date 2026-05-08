"""Safety filters for local VLM outputs."""

from __future__ import annotations

import json
import re
from typing import Any

from personal_lifelog_rag.vlm.schemas import VlmResult


SENSITIVE_TERMS = (
    "恋人",
    "彼氏",
    "彼女",
    "家族",
    "友人",
    "友達",
    "夫",
    "妻",
    "病気",
    "障害",
    "宗教",
    "政治",
    "支持政党",
    "職業",
    "医師",
    "看護師",
    "先生",
    "感情",
    "怒って",
    "悲し",
)

MAX_CAPTION_CHARS = 240
MAX_SHORT_CAPTION_CHARS = 80
MAX_TAGS = 12


def sanitize_vlm_result(result: VlmResult) -> VlmResult:
    """Remove sensitive or over-specific guesses before saving/displaying."""

    safety_flags = list(result.safety_flags)
    caption = _clean_text(result.caption, MAX_CAPTION_CHARS)
    short_caption = _clean_text(result.short_caption or caption, MAX_SHORT_CAPTION_CHARS)
    fields = {
        "scene_tags": _clean_tags(result.scene_tags),
        "object_tags": _clean_tags(result.object_tags),
        "activity_tags": _clean_tags(result.activity_tags),
        "location_cues": _clean_tags(result.location_cues),
        "food_cues": _clean_tags(result.food_cues),
    }
    if _had_sensitive_content(result):
        safety_flags.append("sensitive_terms_removed")
    if result.people_count and result.people_count > 0 and "people_present" not in safety_flags:
        safety_flags.append("people_present")
    return VlmResult(
        caption=caption,
        short_caption=short_caption,
        scene_tags=fields["scene_tags"],
        object_tags=fields["object_tags"],
        activity_tags=fields["activity_tags"],
        location_cues=fields["location_cues"],
        food_cues=fields["food_cues"],
        people_count=result.people_count,
        contains_text_hint=result.contains_text_hint,
        safety_flags=_unique(safety_flags),
        engine=result.engine,
        model_name=result.model_name,
        prompt_version=result.prompt_version,
        confidence=result.confidence,
        status=result.status,
        error_message=_clean_text(result.error_message, 300),
        raw=result.raw,
    )


def result_from_payload(payload: dict[str, Any], *, engine: str, model_name: str | None, prompt_version: str) -> VlmResult:
    """Parse a possibly messy JSON-like VLM payload into a safe result."""

    result = VlmResult(
        caption=_string_or_none(payload.get("caption")),
        short_caption=_string_or_none(payload.get("short_caption")),
        scene_tags=_string_list(payload.get("scene_tags")),
        object_tags=_string_list(payload.get("object_tags")),
        activity_tags=_string_list(payload.get("activity_tags")),
        location_cues=_string_list(payload.get("location_cues")),
        food_cues=_string_list(payload.get("food_cues")),
        people_count=_int_or_none(payload.get("people_count")),
        contains_text_hint=_bool_or_none(payload.get("contains_text_hint")),
        safety_flags=_string_list(payload.get("safety_flags")),
        engine=engine,
        model_name=model_name,
        prompt_version=prompt_version,
        confidence=_float_or_none(payload.get("confidence")),
        status="success",
        raw={"payload_keys": sorted(str(key) for key in payload.keys())},
    )
    return sanitize_vlm_result(result)


def safe_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"caption": text.strip()} if text.strip() else {}
    return value if isinstance(value, dict) else {}


def _clean_text(value: Any, max_chars: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for term in SENSITIVE_TERMS:
        text = text.replace(term, "")
    text = re.sub(r"\s{2,}", " ", text).strip(" 、。")
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text or None


def _clean_tags(values: list[str]) -> list[str]:
    tags: list[str] = []
    for value in values:
        text = _clean_text(value, 48)
        if text and text not in tags:
            tags.append(text)
    return tags[:MAX_TAGS]


def _had_sensitive_content(result: VlmResult) -> bool:
    joined = "\n".join(
        [
            str(result.caption or ""),
            str(result.short_caption or ""),
            "\n".join(result.scene_tags + result.object_tags + result.activity_tags + result.location_cues + result.food_cues),
            str(result.error_message or ""),
        ]
    )
    return any(term in joined for term in SENSITIVE_TERMS)


def _string_or_none(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,、\n]+", value) if part.strip()]
    return [str(value).strip()] if str(value).strip() else []


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return None


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        clean = str(value).strip()
        if clean and clean not in result:
            result.append(clean)
    return result

