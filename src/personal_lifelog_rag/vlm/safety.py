"""Safety filters for local VLM outputs."""

from __future__ import annotations

import json
import re
from typing import Any

from personal_lifelog_rag.retrieval.vlm_evidence import compute_vlm_evidence_strength
from personal_lifelog_rag.vlm.schemas import VlmResult


RELATIONSHIP_TERMS = (
    "恋人",
    "彼氏",
    "彼女",
    "カップル",
    "家族",
    "友人",
    "友達",
    "同僚",
    "母",
    "父",
    "妻",
    "夫",
    "girlfriend",
    "boyfriend",
    "lover",
    "couple",
    "family",
    "friend",
    "coworker",
)
EMOTION_TERMS = (
    "怒っている",
    "悲しんでいる",
    "幸せそう",
    "楽しそう",
    "悲しそう",
    "smiling happily",
    "depressed",
    "angry",
    "sad",
    "happy",
)
SENSITIVE_ATTRIBUTE_TERMS = (
    "病気",
    "障害",
    "宗教",
    "政治",
    "犯罪歴",
    "性的",
    "医療",
    "支持政党",
    "職業",
    "国籍",
    "sick",
    "disabled",
    "religion",
    "religious",
    "political",
    "politics",
    "medical",
    "sexuality",
    "nationality",
)
OVERCLAIM_TERMS = (
    "確実に",
    "間違いなく",
    "definitely",
    "certainly",
)
FORBIDDEN_TERMS = RELATIONSHIP_TERMS + EMOTION_TERMS + SENSITIVE_ATTRIBUTE_TERMS

MAX_CAPTION_CHARS = 240
MAX_SHORT_CAPTION_CHARS = 80
MAX_TAGS = 12


def sanitize_vlm_result(result: VlmResult) -> VlmResult:
    """Remove sensitive or over-specific guesses before saving/displaying."""

    original_text = _joined_result_text(result)
    flags = _safety_flags_for_text(original_text)
    safety_flags = _unique([*result.safety_flags, *flags])
    caption = _clean_text(result.caption, MAX_CAPTION_CHARS)
    short_caption = _clean_text(result.short_caption or caption, MAX_SHORT_CAPTION_CHARS)
    fields = {
        "scene_tags": _clean_tags(result.scene_tags),
        "object_tags": _clean_tags(result.object_tags),
        "activity_tags": _clean_tags(result.activity_tags),
        "location_cues": _clean_tags(result.location_cues),
        "food_cues": _clean_tags(result.food_cues),
        "text_cues": _clean_tags(result.text_cues),
        "uncertainty_notes": _clean_tags(result.uncertainty_notes),
    }
    if result.people_count and result.people_count > 0 and "people_present" not in safety_flags:
        safety_flags.append("people_present")
    evidence_strength = compute_vlm_evidence_strength(result)
    return VlmResult(
        caption=caption,
        short_caption=short_caption,
        scene_tags=fields["scene_tags"],
        object_tags=fields["object_tags"],
        activity_tags=fields["activity_tags"],
        location_cues=fields["location_cues"],
        food_cues=fields["food_cues"],
        text_cues=fields["text_cues"],
        people_count=result.people_count,
        contains_text_hint=result.contains_text_hint,
        uncertainty_notes=fields["uncertainty_notes"],
        safety_flags=_unique(safety_flags),
        evidence_strength=evidence_strength,  # VLM-only is weak by design.
        engine=result.engine,
        model_name=result.model_name,
        prompt_version=result.prompt_version,
        confidence=result.confidence,
        status=result.status,
        error_message=_clean_text(result.error_message, 4000),
        raw=result.raw,
    )


def result_from_payload(payload: dict[str, Any], *, engine: str, model_name: str | None, prompt_version: str) -> VlmResult:
    """Parse a possibly messy JSON-like VLM payload into a safe result."""

    if payload.get("_parse_error"):
        return VlmResult(
            engine=engine,
            model_name=model_name,
            prompt_version=prompt_version,
            status="failed",
            error_message="VLM output was not valid JSON",
            safety_flags=["json_parse_failed"],
            raw={"parse_error": True},
        )

    event_cues = payload.get("event_cues") if isinstance(payload.get("event_cues"), dict) else {}
    result = VlmResult(
        caption=_string_or_none(payload.get("caption")),
        short_caption=_string_or_none(payload.get("short_caption")),
        scene_tags=_string_list(payload.get("scene_tags")) + _true_cue_tags(event_cues, ("outdoor_possible", "indoor_possible")),
        object_tags=_string_list(payload.get("object_tags")),
        activity_tags=_string_list(payload.get("activity_tags"))
        + _true_cue_tags(event_cues, ("travel_possible", "shopping_possible")),
        location_cues=_string_list(payload.get("location_cues")) + _true_cue_tags(event_cues, ("station_possible",)),
        food_cues=_string_list(payload.get("food_cues")) + _true_cue_tags(event_cues, ("meal_possible", "cafe_possible")),
        text_cues=_string_list(payload.get("text_cues")) + _true_cue_tags(event_cues, ("document_or_ticket_possible", "screenshot_possible")),
        people_count=_int_or_none(payload.get("people_count")),
        contains_text_hint=_bool_or_none(payload.get("contains_text_hint")),
        uncertainty_notes=_string_list(payload.get("uncertainty_notes")),
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
    stripped = text.strip()
    if not stripped:
        return {"_parse_error": "empty_output"}
    for candidate in (stripped, _extract_json_object(stripped)):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {"_parse_error": "invalid_json"}


def safety_check_text(text: str) -> dict[str, Any]:
    result = sanitize_vlm_result(
        VlmResult(
            caption=text,
            short_caption=text,
            engine="manual_safety_check",
            status="success",
            confidence=0.0,
        )
    )
    return {
        "input": text,
        "sanitized": result.caption or "",
        "safety_flags": result.safety_flags,
        "violations": _violation_names(result.safety_flags),
        "evidence_strength": result.evidence_strength,
    }


def _clean_text(value: Any, max_chars: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = _soften_overclaim_text(text)
    for term in sorted(FORBIDDEN_TERMS, key=len, reverse=True):
        text = re.sub(re.escape(term), "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip(" 、。,.")
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text or None


def _soften_overclaim_text(text: str) -> str:
    lowered = text.lower()
    if re.fullmatch(r"[a-z0-9_]+_possible", lowered):
        return text
    if any(term in text for term in ("ご飯", "料理", "食事", "ラーメン", "食べ")):
        if any(marker in text for marker in ("食べている", "食べてる", "食べた", "食べています")):
            return "ご飯または食事の可能性がある写真です"
    if any(term in text for term in ("カフェにいる", "cafe", "coffee shop")):
        return "カフェのような場所の可能性があります"
    if any(term in text for term in ("新宿にいる", "駅にいる", "改札にいる")):
        return "都市部または駅周辺の可能性があります"
    if "definitely" in lowered or "certainly" in lowered:
        text = re.sub(r"\b(definitely|certainly)\b", "", text, flags=re.IGNORECASE)
    for term in OVERCLAIM_TERMS:
        text = text.replace(term, "")
    if "している" in text or "した" in text:
        text = text.replace("している", "している可能性があります")
        text = text.replace("した", "した可能性があります")
    return text


def _clean_tags(values: list[str]) -> list[str]:
    tags: list[str] = []
    for value in values:
        text = _clean_text(value, 48)
        if text and text not in tags:
            tags.append(text)
    return tags[:MAX_TAGS]


def _safety_flags_for_text(text: str) -> list[str]:
    flags: list[str] = []
    if _contains_any(text, RELATIONSHIP_TERMS):
        flags.append("relationship_inference_removed")
    if _contains_any(text, EMOTION_TERMS):
        flags.append("emotion_inference_removed")
    if _contains_any(text, SENSITIVE_ATTRIBUTE_TERMS):
        flags.append("sensitive_attribute_removed")
    if _contains_any(text, OVERCLAIM_TERMS) or _contains_overclaim_pattern(text):
        flags.append("overclaim_softened")
    if flags:
        flags.append("sensitive_terms_removed")
        flags.append("forbidden_terms_removed")
    return flags


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _contains_overclaim_pattern(text: str) -> bool:
    return any(pattern in text for pattern in ("している", "食べている", "新宿にいる", "カフェにいる"))


def _joined_result_text(result: VlmResult) -> str:
    return "\n".join(
        [
            str(result.caption or ""),
            str(result.short_caption or ""),
            "\n".join(
                result.scene_tags
                + result.object_tags
                + result.activity_tags
                + result.location_cues
                + result.food_cues
                + result.text_cues
                + result.uncertainty_notes
            ),
            str(result.error_message or ""),
        ]
    )


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _true_cue_tags(event_cues: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    return [key for key in keys if bool(event_cues.get(key))]


def _violation_names(flags: list[str]) -> list[str]:
    mapping = {
        "relationship_inference_removed": "relationship_inference",
        "emotion_inference_removed": "emotion_inference",
        "sensitive_attribute_removed": "sensitive_attribute",
        "overclaim_softened": "overclaim",
    }
    return [mapping[flag] for flag in flags if flag in mapping]


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
