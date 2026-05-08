"""Local OCR engine implementations."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Mapping

from personal_lifelog_rag.ocr.base import OcrEngine
from personal_lifelog_rag.ocr.image_preprocess import preprocessed_image_path
from personal_lifelog_rag.ocr.schemas import OcrBlock, OcrResult


class NoopOcrEngine:
    name = "noop"

    def is_available(self) -> bool:
        return True

    def recognize(self, image_path: Path, languages: list[str]) -> OcrResult:
        return OcrResult(engine=self.name, status="skipped", error_message="OCR engine is noop")


class FakeOcrEngine:
    """Test-only deterministic OCR engine."""

    name = "fake"

    def __init__(self, text_by_name: Mapping[str, str] | None = None, default_text: str = "新宿 OCR テスト") -> None:
        self.text_by_name = dict(text_by_name or {})
        self.default_text = default_text

    def is_available(self) -> bool:
        return True

    def recognize(self, image_path: Path, languages: list[str]) -> OcrResult:
        text = self.text_by_name.get(image_path.name, self.default_text)
        if not text:
            return OcrResult(engine=self.name, status="no_text", confidence=None)
        return OcrResult(
            text=text,
            engine=self.name,
            status="success",
            confidence=0.99,
            blocks=[OcrBlock(text=text, confidence=0.99, bbox=None)],
        )


class TesseractCliOcrEngine:
    name = "tesseract_cli"

    def __init__(self, *, binary: str = "tesseract", timeout_sec: int = 60) -> None:
        self.binary = binary
        self.timeout_sec = timeout_sec

    def is_available(self) -> bool:
        return shutil.which(self.binary) is not None

    def recognize(self, image_path: Path, languages: list[str]) -> OcrResult:
        if not self.is_available():
            return OcrResult(
                engine=self.name,
                status="engine_unavailable",
                error_message="tesseract command is not installed",
            )
        lang = "+".join(languages) if languages else "jpn+eng"
        try:
            with preprocessed_image_path(image_path) as processed_path:
                completed = subprocess.run(
                    [self.binary, str(processed_path), "stdout", "-l", lang],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                )
        except Exception as exc:
            return OcrResult(
                engine=self.name,
                status="failed",
                error_message=f"OCR failed with {exc.__class__.__name__}",
            )
        if completed.returncode != 0:
            return OcrResult(
                engine=self.name,
                status="failed",
                error_message=(completed.stderr or "tesseract failed").strip()[:300],
            )
        text = completed.stdout.strip()
        if not text:
            return OcrResult(engine=self.name, status="no_text")
        return OcrResult(
            text=text,
            engine=self.name,
            status="success",
            confidence=None,
            blocks=[OcrBlock(text=text, confidence=None, bbox=None)],
        )


class PyTesseractOcrEngine:
    name = "pytesseract"

    def is_available(self) -> bool:
        try:
            import pytesseract  # noqa: F401
        except ImportError:
            return False
        return True

    def recognize(self, image_path: Path, languages: list[str]) -> OcrResult:
        if not self.is_available():
            return OcrResult(
                engine=self.name,
                status="engine_unavailable",
                error_message="pytesseract is not installed",
            )
        try:
            import pytesseract
            from PIL import Image

            lang = "+".join(languages) if languages else "jpn+eng"
            with Image.open(image_path) as image:
                text = pytesseract.image_to_string(image, lang=lang).strip()
        except Exception as exc:
            return OcrResult(
                engine=self.name,
                status="failed",
                error_message=f"OCR failed with {exc.__class__.__name__}",
            )
        if not text:
            return OcrResult(engine=self.name, status="no_text")
        return OcrResult(text=text, engine=self.name, status="success", blocks=[OcrBlock(text=text)])


def get_ocr_engine(name: str | None = None) -> OcrEngine:
    resolved = (name or "tesseract_cli").strip().lower()
    if resolved in {"noop", "none", "disabled", "off"}:
        return NoopOcrEngine()
    if resolved == "fake":
        return FakeOcrEngine()
    if resolved in {"tesseract", "tesseract_cli", "tesseract-cli"}:
        return TesseractCliOcrEngine()
    if resolved in {"pytesseract", "python-tesseract"}:
        return PyTesseractOcrEngine()
    return NoopOcrEngine()
