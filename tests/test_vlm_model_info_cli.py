from __future__ import annotations

import json
from pathlib import Path

from personal_lifelog_rag.app.cli import main


def test_vlm_model_info_cli_json(tmp_path: Path, capsys) -> None:
    config = tmp_path / "model_runtime.yaml"
    config.write_text(
        """
models:
  vlm:
    engine: "qwen3_vl_transformers"
    model_name: "Qwen/Qwen3-VL-8B-Instruct"
    model_path: null
    device: "auto"
  multimodal_embedding:
    engine: "qwen3_vl_embedding_sentence_transformers"
    model_name: "Qwen/Qwen3-VL-Embedding-2B"
    model_path: null
    device: "auto"
""",
        encoding="utf-8",
    )

    code = main(["vlm-model-info", "--config", str(config), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["auto_download"] is False
    assert payload["vlm"]["engine"] == "qwen3_vl_transformers"
    assert payload["multimodal_embedding"]["available"] is False


def test_benchmark_clis_with_fake_engine(tmp_path: Path, capsys) -> None:
    cases = tmp_path / "cases.yaml"
    cases.write_text(
        """
cases:
  - id: food_sample
    image_path: "private_eval/vlm_benchmark/images/food/sample.jpg"
    query_texts:
      - "料理の写真"
    expected_tags_any:
      - "meal_possible"
    forbidden_terms:
      - "恋人"
""",
        encoding="utf-8",
    )

    vlm_code = main(["benchmark-vlm", "--cases", str(cases), "--engine", "fake", "--json"])
    vlm_payload = json.loads(capsys.readouterr().out)
    embedding_code = main(["benchmark-image-embedding", "--cases", str(cases), "--engine", "fake", "--json"])
    embedding_payload = json.loads(capsys.readouterr().out)
    combined_code = main(
        [
            "benchmark-qwen-multimodal",
            "--cases",
            str(cases),
            "--engine",
            "fake",
            "--save",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    combined_output = capsys.readouterr().out

    assert vlm_code == 0
    assert vlm_payload["vlm"]["success_count"] == 1
    assert embedding_code == 0
    assert embedding_payload["embedding"]["metrics"]["top1_accuracy"] == 1.0
    assert combined_code == 0
    assert "Saved:" in combined_output
    assert list((tmp_path / "runs").glob("*.json"))
    assert list((tmp_path / "runs").glob("*.md"))

