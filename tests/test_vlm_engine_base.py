from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.vlm.engines import FakeVlmEngine, NoopVlmEngine, get_vlm_engine


def test_fake_vlm_engine_returns_safe_structured_result(tmp_path: Path) -> None:
    image_path = tmp_path / "dummy.jpg"
    image_path.write_bytes(b"not-used-by-fake")

    result = FakeVlmEngine().analyze_image(image_path, "prompt")

    assert result.status == "success"
    assert result.engine == "fake"
    assert "ramen_possible" in result.food_cues
    assert result.caption


def test_noop_vlm_engine_skips_without_crashing(tmp_path: Path) -> None:
    result = NoopVlmEngine().analyze_image(tmp_path / "missing.jpg", "prompt")

    assert result.status == "skipped"
    assert result.engine == "noop"


def test_ollama_engine_requires_localhost_url() -> None:
    try:
        get_vlm_engine("ollama", model_name="dummy")
    except ValueError:
        raise AssertionError("default Ollama URL should be localhost-only")

