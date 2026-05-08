"""Repository facade for VLM rows."""

from __future__ import annotations

from typing import Any


class MediaVlmRepository:
    def __init__(self, repository) -> None:
        self.repository = repository

    def save(self, **row: Any) -> None:
        self.repository.upsert_media_vlm(**row)

    def get(self, media_id: str) -> dict[str, Any] | None:
        return self.repository.get_media_vlm(media_id)

    def list(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.repository.list_media_vlm(**kwargs)

