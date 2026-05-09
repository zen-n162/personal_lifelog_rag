"""Evidence-strength helpers for conservative multimodal ranking."""

from __future__ import annotations


STRENGTH_ORDER = {"weak": 0, "medium": 1, "strong": 2}
CONFIDENCE_ORDER = {"低": 0, "中": 1, "高": 2}


def compute_evidence_strength(
    evidence_types: list[str],
    *,
    verified_event: bool = False,
    pinned_event: bool = False,
) -> str:
    """Classify how much support a multimodal result has.

    VLM-only and embedding-only results are intentionally weak because they are
    candidate finders, not factual proof of an activity.
    """

    types = set(evidence_types)
    if verified_event or pinned_event:
        return "strong"
    if {"vlm", "ocr", "line", "event"}.issubset(types):
        return "strong"
    if {"embedding", "vlm", "ocr", "event"}.issubset(types):
        return "strong"
    if {"embedding", "vlm", "ocr", "line"}.issubset(types):
        return "strong"
    if "event" in types and "line" in types and ("gps" in types or "place" in types):
        return "strong"

    medium_pairs = [
        {"vlm", "event"},
        {"vlm", "line"},
        {"vlm", "ocr"},
        {"embedding", "vlm"},
        {"event", "photo"},
        {"line", "photo"},
    ]
    if any(pair.issubset(types) for pair in medium_pairs):
        return "medium"
    return "weak"


def confidence_label_for_score(
    score: float,
    *,
    evidence_types: list[str],
    safety_flags: list[str] | None = None,
) -> str:
    """Return a display confidence label with VLM/embedding-only caps."""

    if score >= 0.75:
        label = "高"
    elif score >= 0.45:
        label = "中"
    else:
        label = "低"

    types = set(evidence_types)
    flags = set(safety_flags or [])
    has_context = bool(types & {"ocr", "line", "event", "place", "gps"})
    visual_only = bool(types & {"vlm", "embedding"}) and not has_context
    people_only = flags == {"people_present"} and not has_context
    sensitive = bool(flags & {"forbidden_terms_found", "relationship_inference_removed", "emotion_inference_removed"})
    if label == "高" and (visual_only or people_only or sensitive):
        return "中"
    return label


def strength_at_least(actual: str, expected: str) -> bool:
    return STRENGTH_ORDER.get(actual, -1) >= STRENGTH_ORDER.get(expected, -1)


def confidence_at_most(actual: str, maximum: str) -> bool:
    return CONFIDENCE_ORDER.get(actual, 99) <= CONFIDENCE_ORDER.get(maximum, -1)
