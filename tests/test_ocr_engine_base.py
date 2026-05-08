from __future__ import annotations

from pathlib import Path

from PIL import Image

from personal_lifelog_rag.ocr.engines import FakeOcrEngine, NoopOcrEngine, TesseractCliOcrEngine, get_ocr_engine


def test_fake_ocr_engine_returns_deterministic_result(tmp_path: Path) -> None:
    image_path = tmp_path / "dummy.png"
    Image.new("RGB", (8, 8), "white").save(image_path)

    result = FakeOcrEngine(default_text="新宿 OCR").recognize(image_path, ["jpn", "eng"])

    assert result.status == "success"
    assert result.text == "新宿 OCR"
    assert result.blocks[0].text == "新宿 OCR"


def test_noop_ocr_engine_skips_without_failure(tmp_path: Path) -> None:
    image_path = tmp_path / "dummy.png"
    Image.new("RGB", (8, 8), "white").save(image_path)

    result = NoopOcrEngine().recognize(image_path, ["jpn"])

    assert result.status == "skipped"
    assert "noop" in (result.error_message or "")


def test_tesseract_cli_unavailable_is_reported_without_running(tmp_path: Path) -> None:
    image_path = tmp_path / "dummy.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    engine = TesseractCliOcrEngine(binary="definitely_missing_tesseract_for_test")

    assert engine.is_available() is False
    result = engine.recognize(image_path, ["jpn"])

    assert result.status == "engine_unavailable"


def test_get_ocr_engine_supports_safe_defaults() -> None:
    assert get_ocr_engine("fake").name == "fake"
    assert get_ocr_engine("noop").name == "noop"
    assert get_ocr_engine("unknown_engine_name").name == "noop"
