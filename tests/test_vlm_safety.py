from __future__ import annotations

from personal_lifelog_rag.vlm.safety import safe_json_object, sanitize_vlm_result
from personal_lifelog_rag.vlm.schemas import VlmResult


def test_vlm_safety_removes_sensitive_relationship_and_attribute_terms() -> None:
    result = sanitize_vlm_result(
        VlmResult(
            caption="恋人と家族が写る写真で、病気や職業を推定しています",
            scene_tags=["家族", "restaurant"],
            people_count=2,
            safety_flags=[],
        )
    )

    joined = " ".join([result.caption or "", *result.scene_tags])
    assert "恋人" not in joined
    assert "家族" not in joined
    assert "病気" not in joined
    assert "職業" not in joined
    assert "sensitive_terms_removed" in result.safety_flags
    assert "people_present" in result.safety_flags


def test_safe_json_object_falls_back_to_caption_text() -> None:
    payload = safe_json_object("カフェの可能性がある写真")

    assert payload["_parse_error"] == "invalid_json"


def test_safe_json_object_extracts_embedded_json() -> None:
    payload = safe_json_object('prefix {"caption": "カフェの可能性"} suffix')

    assert payload["caption"] == "カフェの可能性"
