"""Base interface for local-only OCR engines."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from personal_lifelog_rag.ocr.schemas import OcrResult


class OcrEngine(Protocol):
    name: str

    def is_available(self) -> bool:
        """Return whether the local OCR engine can run on this machine."""

    def recognize(self, image_path: Path, languages: list[str]) -> OcrResult:
        """Extract text from a local image without network access."""
