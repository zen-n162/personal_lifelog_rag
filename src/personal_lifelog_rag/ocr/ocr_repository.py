"""Small repository facade for OCR-specific persistence."""

from __future__ import annotations

from typing import Any


class MediaOcrRepository:
    """Thin facade over LifelogRepository for OCR-specific callers/tests."""

    def __init__(self, repository) -> None:
        self.repository = repository

    def save(self, row: dict[str, Any]) -> None:
        self.repository.upsert_media_ocr(**row)

    def get(self, media_id: str) -> dict[str, Any] | None:
        return self.repository.get_media_ocr(media_id)

    def list(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.repository.list_media_ocr(**kwargs)
