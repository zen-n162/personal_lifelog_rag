from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.benchmark.schemas import BenchmarkCase, load_benchmark_cases
from personal_lifelog_rag.benchmark.vlm_benchmark import benchmark_vlm
from personal_lifelog_rag.vlm.engines import FakeVlmEngine, get_vlm_engine
from personal_lifelog_rag.vlm.schemas import VlmResult


class UnsafeTokenVlmEngine:
    name = "unsafe_token_fake"
    model_name = "unsafe-token"

    def is_available(self) -> bool:
        return True

    def analyze_image(self, image_path: Path, prompt: str) -> VlmResult:
        return VlmResult(
            caption="unsafe_token appears in this caption",
            short_caption="unsafe_token",
            scene_tags=["unsafe_token"],
            engine=self.name,
            model_name=self.model_name,
            status="success",
        )


def test_fake_vlm_benchmark_matches_expected_tags_and_blocks_forbidden_terms() -> None:
    cases = [
        BenchmarkCase(
            id="food_sample",
            image_path=Path("private_eval/vlm_benchmark/images/food/sample.jpg"),
            expected_tags_any=["meal_possible", "station_possible"],
            forbidden_terms=["恋人", "病気"],
        )
    ]

    report = benchmark_vlm(cases, engine=FakeVlmEngine())

    row = report["case_results"][0]["vlm"]
    assert report["success_count"] == 1
    assert "meal_possible" in row["tags_matched"]
    assert row["forbidden_terms_found"] == []
    assert row["schema_valid"] is True


def test_unavailable_qwen_vlm_does_not_crash() -> None:
    engine = get_vlm_engine("qwen3_vl_transformers", model_name="Qwen/Qwen3-VL-8B-Instruct")

    report = benchmark_vlm(
        [BenchmarkCase(id="dummy", image_path=Path("missing.jpg"))],
        engine=engine,
    )

    assert report["case_results"][0]["vlm"]["status"] == "engine_unavailable"


def test_vlm_benchmark_detects_forbidden_terms(tmp_path: Path) -> None:
    image_path = tmp_path / "dummy.jpg"
    image_path.write_bytes(b"not a real image, engine ignores it")
    report = benchmark_vlm(
        [
            BenchmarkCase(
                id="unsafe",
                image_path=image_path,
                forbidden_terms=["unsafe_token"],
            )
        ],
        engine=UnsafeTokenVlmEngine(),
    )

    assert report["safety_violations"] == 1
    assert report["case_results"][0]["vlm"]["forbidden_terms_found"] == ["unsafe_token"]


def test_load_benchmark_cases_reads_simple_yaml(tmp_path: Path) -> None:
    path = tmp_path / "cases.yaml"
    path.write_text(
        """
cases:
  - id: food_sample
    image_path: "private_eval/vlm_benchmark/images/food/sample.jpg"
    query_texts:
      - "料理の写真"
    expected_tags_any:
      - "meal_possible"
    forbidden_terms:
      - "家族"
""",
        encoding="utf-8",
    )

    cases = load_benchmark_cases(path)

    assert cases[0].id == "food_sample"
    assert cases[0].query_texts == ["料理の写真"]
    assert cases[0].forbidden_terms == ["家族"]
