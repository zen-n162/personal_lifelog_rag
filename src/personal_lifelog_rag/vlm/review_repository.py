"""Small repository facade for VLM review overrides."""

from __future__ import annotations

from typing import Any


class MediaVlmReviewRepository:
    def __init__(self, repository) -> None:
        self.repository = repository

    def get(self, media_id: str) -> dict[str, Any] | None:
        return self.repository.get_media_vlm_override(media_id)

    def save(self, **kwargs: Any) -> None:
        self.repository.upsert_media_vlm_override(**kwargs)

    def delete(self, media_id: str) -> int:
        return self.repository.delete_media_vlm_override(media_id)
