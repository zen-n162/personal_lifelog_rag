"""Schemas for local multimodal embeddings and hybrid search."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


EmbeddingType = Literal["image", "caption", "ocr", "combined_text"]
EmbeddingStatus = Literal["pending", "success", "skipped", "failed", "engine_unavailable"]
EmbeddingFormat = Literal["float32_numpy", "json"]
SearchBackend = Literal["sql", "vlm_sql", "embedding", "hybrid"]
EvidenceStrength = Literal["weak", "medium", "strong"]


@dataclass(frozen=True)
class EmbeddingResult:
    vector: list[float] = field(default_factory=list)
    model_name: str | None = None
    embedding_dim: int | None = None
    status: EmbeddingStatus = "success"
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BuildMediaEmbeddingsOptions:
    start_date: str | None = None
    end_date: str | None = None
    limit: int = 100
    embedding_type: EmbeddingType = "image"
    engine_name: str | None = None
    model_name: str | None = None
    model_path: str | None = None
    device: str | None = "auto"
    dtype: str | None = "auto"
    local_files_only: bool | None = True
    embedding_dim: int | None = None
    batch_size: int | None = None
    dry_run: bool = False
    force: bool = False
    skip_existing: bool = False
    media_ids: list[str] | None = None


@dataclass
class BuildMediaEmbeddingsReport:
    selected: int = 0
    processed: int = 0
    success: int = 0
    skipped: int = 0
    failed: int = 0
    engine_unavailable: int = 0
    dry_run: bool = False
    embedding_type: str = "image"
    engine: str = "unknown"
    model_name: str | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MultimodalSearchOptions:
    query: str
    date_from: str | None = None
    date_to: str | None = None
    limit: int = 10
    backend: SearchBackend = "hybrid"
    engine_name: str | None = None
    model_name: str | None = None
    model_path: str | None = None
    device: str | None = "auto"
    dtype: str | None = "auto"
    local_files_only: bool | None = True
    embedding_dim: int | None = None
    batch_size: int | None = None
    include_hidden: bool = False
