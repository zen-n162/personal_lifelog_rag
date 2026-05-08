"""Local-only OCR adapter boundary.

The default adapter is intentionally disabled. Users must opt in to a local OCR
engine, and missing engines return skipped results instead of breaking the app.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Protocol


OCR_BACKEND_ENV_VAR = "PERSONAL_LIFELOG_RAG_OCR_BACKEND"
OCR_LANG_ENV_VAR = "PERSONAL_LIFELOG_RAG_OCR_LANG"


@dataclass(frozen=True)
class OCRResult:
    text: str | None
    engine: str
    skipped: bool = False
    reason: str | None = None


class LocalOCRAdapter(Protocol):
    engine: str
    available: bool

    def extract_text(self, image_path: str | Path) -> OCRResult:
        """Extract text from a local image file."""


class UnconfiguredOCRAdapter:
    engine = "none"
    available = False

    def __init__(self, reason: str = "未解析: OCR backend is not configured") -> None:
        self.reason = reason

    def extract_text(self, image_path: str | Path) -> OCRResult:
        return OCRResult(text=None, engine=self.engine, skipped=True, reason=self.reason)


class TesseractOCRAdapter:
    engine = "tesseract"
    available = True

    def __init__(self, *, lang: str | None = None) -> None:
        self.lang = lang or os.getenv(OCR_LANG_ENV_VAR, "jpn+eng")

    def extract_text(self, image_path: str | Path) -> OCRResult:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            return OCRResult(
                text=None,
                engine=self.engine,
                skipped=True,
                reason="未解析: pytesseract is not installed",
            )

        try:
            with Image.open(image_path) as image:
                text = pytesseract.image_to_string(image, lang=self.lang).strip()
        except Exception as exc:  # pragma: no cover - depends on local OCR setup
            return OCRResult(
                text=None,
                engine=self.engine,
                skipped=True,
                reason=f"未解析: OCR failed with {exc.__class__.__name__}",
            )

        if not text:
            return OCRResult(text=None, engine=self.engine, skipped=True, reason="未解析: no text detected")
        return OCRResult(text=text, engine=self.engine)


class PaddleOCRAdapter:
    engine = "paddleocr"
    available = True

    def __init__(self, *, lang: str | None = None) -> None:
        self.lang = lang or os.getenv(OCR_LANG_ENV_VAR, "japan")
        self._ocr: Any | None = None

    def extract_text(self, image_path: str | Path) -> OCRResult:
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            return OCRResult(
                text=None,
                engine=self.engine,
                skipped=True,
                reason="未解析: paddleocr is not installed",
            )

        try:
            if self._ocr is None:
                self._ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
            result = self._ocr.ocr(str(image_path), cls=True)
            text = "\n".join(_paddle_text_lines(result)).strip()
        except Exception as exc:  # pragma: no cover - depends on local OCR setup
            return OCRResult(
                text=None,
                engine=self.engine,
                skipped=True,
                reason=f"未解析: OCR failed with {exc.__class__.__name__}",
            )

        if not text:
            return OCRResult(text=None, engine=self.engine, skipped=True, reason="未解析: no text detected")
        return OCRResult(text=text, engine=self.engine)


def get_ocr_adapter(backend: str | None = None) -> LocalOCRAdapter:
    resolved = (backend or os.getenv(OCR_BACKEND_ENV_VAR, "none")).strip().lower()
    if resolved in {"", "none", "disabled", "off"}:
        return UnconfiguredOCRAdapter()
    if resolved == "tesseract":
        return TesseractOCRAdapter()
    if resolved == "paddleocr":
        return PaddleOCRAdapter()
    return UnconfiguredOCRAdapter(reason=f"未解析: unsupported OCR backend '{resolved}'")


def _paddle_text_lines(result: Any) -> list[str]:
    lines: list[str] = []
    for page in result or []:
        for item in page or []:
            if (
                isinstance(item, (list, tuple))
                and len(item) >= 2
                and isinstance(item[1], (list, tuple))
                and item[1]
            ):
                lines.append(str(item[1][0]))
    return lines

