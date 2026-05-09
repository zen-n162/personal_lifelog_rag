from __future__ import annotations

from personal_lifelog_rag.retrieval.vlm_evidence import compute_vlm_evidence_strength
from personal_lifelog_rag.vlm.schemas import VlmResult


def test_vlm_only_evidence_is_weak() -> None:
    result = VlmResult(confidence=0.95)

    assert compute_vlm_evidence_strength(result) == "weak"


def test_vlm_with_one_independent_match_can_be_medium() -> None:
    result = VlmResult(confidence=0.8)

    assert compute_vlm_evidence_strength(result, ocr_match=True) == "medium"


def test_vlm_with_multiple_independent_matches_can_be_strong() -> None:
    result = VlmResult(confidence=0.8)

    assert compute_vlm_evidence_strength(result, ocr_match=True, line_match=True, place_match=True) == "strong"

