"""Common interface for local multimodal embedding engines."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from personal_lifelog_rag.embeddings.schemas import EmbeddingResult


class MultimodalEmbeddingEngine(Protocol):
    name: str
    model_name: str | None

    def is_available(self) -> bool:
        """Return whether this local engine can run without network access."""

    def embed_image(self, image_path: Path) -> EmbeddingResult:
        """Embed one local image path."""

    def embed_text(self, text: str) -> EmbeddingResult:
        """Embed one local text query or media-derived text."""

