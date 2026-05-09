from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.benchmark.benchmark_report import (
    build_multimodal_benchmark_report,
    format_benchmark_markdown,
    model_info,
    write_benchmark_report,
)
from personal_lifelog_rag.benchmark.schemas import BenchmarkCase, ModelRuntimeConfig, ModelSpec


def test_multimodal_report_with_fake_engines_and_markdown(tmp_path: Path) -> None:
    cases = [
        BenchmarkCase(
            id="food_sample",
            image_path=Path("private_eval/vlm_benchmark/images/food/sample.jpg"),
            query_texts=["料理の写真"],
            expected_tags_any=["meal_possible"],
            forbidden_terms=["恋人"],
        )
    ]

    report = build_multimodal_benchmark_report(
        cases,
        ModelRuntimeConfig(),
        engine_override="fake",
    )
    markdown = format_benchmark_markdown(report)
    outputs = write_benchmark_report(report, output_dir=tmp_path)

    assert report["summary"]["vlm_success_rate"] == 1.0
    assert report["summary"]["top1_accuracy"] == 1.0
    assert "Qwen3-VL should be used" in markdown
    assert Path(outputs["json"]).exists()
    assert Path(outputs["markdown"]).exists()


def test_model_info_does_not_mark_remote_model_id_available() -> None:
    config = ModelRuntimeConfig(
        vlm=ModelSpec(engine="qwen3_vl_transformers", model_name="Qwen/Qwen3-VL-8B-Instruct"),
        multimodal_embedding=ModelSpec(
            engine="qwen3_vl_embedding_sentence_transformers",
            model_name="Qwen/Qwen3-VL-Embedding-2B",
        ),
    )

    report = model_info(config)

    assert report["auto_download"] is False
    assert report["vlm"]["available"] is False
    assert report["multimodal_embedding"]["available"] is False

