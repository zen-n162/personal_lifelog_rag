from __future__ import annotations

from personal_lifelog_rag.vlm.safety import result_from_payload


def test_structured_payload_populates_safe_vlm_schema_fields() -> None:
    result = result_from_payload(
        {
            "caption": "カフェのような場所の可能性",
            "short_caption": "カフェ可能性",
            "scene_tags": ["indoor"],
            "text_cues": ["menu_possible"],
            "uncertainty_notes": ["visual_only"],
            "confidence": 0.8,
        },
        engine="fake",
        model_name="fake",
        prompt_version="lifelog_structured_tags_v1",
    )

    assert result.text_cues == ["menu_possible"]
    assert result.uncertainty_notes == ["visual_only"]
    assert result.evidence_strength == "weak"


def test_event_cues_payload_maps_to_possible_tags() -> None:
    result = result_from_payload(
        {
            "event_cues": {
                "meal_possible": True,
                "station_possible": True,
                "document_or_ticket_possible": True,
            },
            "confidence": 0.7,
        },
        engine="fake",
        model_name="fake",
        prompt_version="lifelog_event_cues_v1",
    )

    assert "meal_possible" in result.food_cues
    assert "station_possible" in result.location_cues
    assert "document_or_ticket_possible" in result.text_cues

