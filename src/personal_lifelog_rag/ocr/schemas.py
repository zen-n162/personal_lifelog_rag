"""Dataclasses shared by local OCR engines and services."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


OcrStatus = Literal[
    "pending",
    "success",
    "skipped",
    "failed",
    "no_text",
    "no_text_detected",
    "engine_unavailable",
]


@dataclass(frozen=True)
class OcrBlock:
    text: str
    confidence: float | None = None
    bbox: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OcrResult:
    text: str | None = None
    engine: str = "unknown"
    status: OcrStatus = "success"
    confidence: float | None = None
    blocks: list[OcrBlock] = field(default_factory=list)
    error_message: str | None = None

    def to_blocks_json_rows(self) -> list[dict[str, Any]]:
        return [block.to_dict() for block in self.blocks]


@dataclass(frozen=True)
class OcrTarget:
    media_id: str
    image_path: str
    file_name: str | None = None
    captured_at: str | None = None


@dataclass
class OcrImagesReport:
    selected_images: int = 0
    processed: int = 0
    success: int = 0
    no_text: int = 0
    failed: int = 0
    skipped: int = 0
    engine_unavailable: int = 0
    dry_run: bool = False
    engine: str = "unknown"
    languages: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
