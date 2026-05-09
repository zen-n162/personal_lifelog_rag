from __future__ import annotations

import json

from personal_lifelog_rag.app.cli import main


def test_vlm_prompt_cli_outputs_json(capsys) -> None:
    code = main(["vlm-prompt", "--template", "lifelog_structured_tags_v1", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["name"] == "lifelog_structured_tags_v1"
    assert "Do not identify people" in payload["prompt"]


def test_vlm_safety_check_cli_outputs_flags(capsys) -> None:
    code = main(["vlm-safety-check", "--text", "彼女と楽しそうにご飯を食べている写真です", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert "relationship_inference_removed" in payload["safety_flags"]
    assert "emotion_inference_removed" in payload["safety_flags"]
    assert "彼女" not in payload["sanitized"]

