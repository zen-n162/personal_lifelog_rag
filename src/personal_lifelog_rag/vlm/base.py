"""VLM engine protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from personal_lifelog_rag.vlm.schemas import VlmResult


class VlmEngine(Protocol):
    name: str
    model_name: str | None

    def is_available(self) -> bool:
        """Return whether this engine can run locally in the current environment."""

    def analyze_image(self, image_path: Path, prompt: str) -> VlmResult:
        """Analyze a local image without sending it to a cloud service."""

