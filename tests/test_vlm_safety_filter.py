from __future__ import annotations

from personal_lifelog_rag.vlm.safety import sanitize_vlm_result, safety_check_text
from personal_lifelog_rag.vlm.schemas import VlmResult


def test_relationship_and_emotion_inference_are_removed() -> None:
    report = safety_check_text("彼女と楽しそうにご飯を食べている写真です")

    assert "彼女" not in report["sanitized"]
    assert "楽しそう" not in report["sanitized"]
    assert "relationship_inference_removed" in report["safety_flags"]
    assert "emotion_inference_removed" in report["safety_flags"]


def test_overclaim_is_softened() -> None:
    result = sanitize_vlm_result(
        VlmResult(caption="確実に料理を食べている写真です", confidence=0.9)
    )

    assert "確実に" not in (result.caption or "")
    assert "可能性" in (result.caption or "")
    assert "overclaim_softened" in result.safety_flags

