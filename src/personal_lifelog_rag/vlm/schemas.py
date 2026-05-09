"""Dataclasses shared by local VLM engines and services."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


VlmStatus = Literal[
    "pending",
    "success",
    "skipped",
    "failed",
    "no_visual_content",
    "engine_unavailable",
]
EvidenceStrength = Literal["weak", "medium", "strong"]


@dataclass(frozen=True)
class VlmResult:
    caption: str | None = None
    short_caption: str | None = None
    scene_tags: list[str] = field(default_factory=list)
    object_tags: list[str] = field(default_factory=list)
    activity_tags: list[str] = field(default_factory=list)
    location_cues: list[str] = field(default_factory=list)
    food_cues: list[str] = field(default_factory=list)
    text_cues: list[str] = field(default_factory=list)
    people_count: int | None = None
    contains_text_hint: bool | None = None
    uncertainty_notes: list[str] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)
    evidence_strength: EvidenceStrength = "weak"
    engine: str = "unknown"
    model_name: str | None = None
    prompt_version: str | None = None
    confidence: float | None = None
    status: VlmStatus = "success"
    error_message: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SafeVlmAnalysis:
    caption: str | None = None
    short_caption: str | None = None
    scene_tags: list[str] = field(default_factory=list)
    object_tags: list[str] = field(default_factory=list)
    activity_tags: list[str] = field(default_factory=list)
    food_cues: list[str] = field(default_factory=list)
    location_cues: list[str] = field(default_factory=list)
    text_cues: list[str] = field(default_factory=list)
    people_count: int | None = None
    contains_text_hint: bool = False
    confidence: float | None = None
    uncertainty_notes: list[str] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)
    evidence_strength: EvidenceStrength = "weak"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VlmImagesReport:
    selected_images: int = 0
    processed: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    no_visual_content: int = 0
    engine_unavailable: int = 0
    dry_run: bool = False
    engine: str = "unknown"
    model_name: str | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ImageSearchOptions:
    query: str
    date_from: str | None = None
    date_to: str | None = None
    limit: int = 20
    include_hidden: bool = False
