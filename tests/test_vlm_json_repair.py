from __future__ import annotations

from personal_lifelog_rag.vlm.json_repair import parse_json_object_with_repair
from personal_lifelog_rag.vlm.safety import result_from_payload, safe_json_object


def test_json_repair_parses_markdown_code_block() -> None:
    result = parse_json_object_with_repair(
        '```json\n{"caption": "stage", "scene_tags": ["stage_possible"]}\n```'
    )

    assert result.payload is not None
    assert result.payload["caption"] == "stage"
    assert result.repaired is True


def test_json_repair_removes_trailing_commas() -> None:
    result = parse_json_object_with_repair(
        '{"caption": "food", "food_cues": ["meal_possible",],}'
    )

    assert result.payload is not None
    assert result.payload["food_cues"] == ["meal_possible"]
    assert "removed_trailing_commas" in result.notes


def test_json_repair_extracts_json_from_surrounding_text() -> None:
    result = parse_json_object_with_repair(
        'Here is the JSON:\n{"caption": "ticket", "text_cues": ["ticket_possible"]}\nDone.'
    )

    assert result.payload is not None
    assert result.payload["text_cues"] == ["ticket_possible"]


def test_json_repair_balances_missing_tail() -> None:
    result = parse_json_object_with_repair(
        '{"caption": "cafe", "scene_tags": ["indoor"], "food_cues": ["cafe_possible"]'
    )

    assert result.payload is not None
    assert result.payload["caption"] == "cafe"
    assert "balanced_delimiters" in result.notes


def test_json_repair_escapes_likely_unescaped_quotes() -> None:
    result = parse_json_object_with_repair(
        '{"caption": "A sign says "CAFE" near the table", "short_caption": "cafe sign"}'
    )

    assert result.payload is not None
    assert result.payload["caption"] == 'A sign says "CAFE" near the table'
    assert "escaped_unescaped_quotes" in result.notes


def test_safe_json_object_marks_repaired_payload_and_result_flag() -> None:
    payload = safe_json_object('```json\n{"caption": "food", "food_cues": ["meal_possible",],}\n```')
    result = result_from_payload(
        payload,
        engine="qwen3_vl_transformers",
        model_name="local",
        prompt_version="lifelog_structured_tags_v1",
    )

    assert payload["_json_repaired"] is True
    assert result.status == "success"
    assert "json_repaired" in result.safety_flags


def test_json_repair_returns_failure_for_unrepairable_text() -> None:
    result = parse_json_object_with_repair("no json here")

    assert result.payload is None
    assert result.error_message
