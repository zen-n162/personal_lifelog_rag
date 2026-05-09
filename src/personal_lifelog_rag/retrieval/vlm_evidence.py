"""Evidence-strength helpers for cautious VLM usage."""

from __future__ import annotations

from typing import Any


def compute_vlm_evidence_strength(
    vlm_result: Any,
    *,
    ocr_match: bool = False,
    line_match: bool = False,
    gps_match: bool = False,
    place_match: bool = False,
    event_evidence_match: bool = False,
) -> str:
    """Classify VLM evidence strength.

    VLM-only evidence stays weak. It can become medium when one independent
    local signal agrees, and strong only when multiple independent signals agree.
    """

    supporting = sum(bool(value) for value in (ocr_match, line_match, gps_match, place_match, event_evidence_match))
    confidence = _confidence(vlm_result)
    if supporting >= 3 and confidence >= 0.7:
        return "strong"
    if supporting >= 1 and confidence >= 0.65:
        return "medium"
    return "weak"


def _confidence(vlm_result: Any) -> float:
    if isinstance(vlm_result, dict):
        value = vlm_result.get("confidence")
    else:
        value = getattr(vlm_result, "confidence", None)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

