from __future__ import annotations

from personal_lifelog_rag.vlm.prompts import PROMPT_TEMPLATES, get_vlm_prompt_template


def test_required_vlm_prompt_templates_exist() -> None:
    assert "lifelog_safe_caption_v1" in PROMPT_TEMPLATES
    assert "lifelog_structured_tags_v1" in PROMPT_TEMPLATES
    assert "lifelog_event_cues_v1" in PROMPT_TEMPLATES
    assert "food_specific_tags_v1" in PROMPT_TEMPLATES


def test_structured_prompt_contains_safety_rules_and_json_instruction() -> None:
    prompt = get_vlm_prompt_template("lifelog_structured_tags_v1").prompt

    assert "Do not identify people" in prompt
    assert "Do not infer relationships" in prompt
    assert "Return valid JSON only" in prompt
    assert "Do not include explanations outside JSON" in prompt
    assert "Do not include reasoning" in prompt
    assert "Use possible tags instead of definitive claims" in prompt
    assert "text_cues" in prompt


def test_food_specific_prompt_contains_dish_fields() -> None:
    prompt = get_vlm_prompt_template("food_specific_tags_v1").prompt

    assert "food_specific_name" in prompt
    assert "food_candidates" in prompt
    assert "dish_confidence" in prompt
    assert "uncertainty_notes" in prompt
