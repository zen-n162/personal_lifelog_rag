"""Schemas for local analysis job planning and execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


JobType = Literal["ocr", "vlm", "image_embedding", "text_embedding", "event_rebuild"]
JobStatus = Literal["planned", "running", "completed", "failed", "canceled", "partial"]
JobItemStatus = Literal["pending", "running", "success", "failed", "skipped", "engine_unavailable"]

JOB_TYPES = {"ocr", "vlm", "image_embedding", "text_embedding", "event_rebuild"}
JOB_STATUSES = {"planned", "running", "completed", "failed", "canceled", "partial"}
JOB_ITEM_STATUSES = {"pending", "running", "success", "failed", "skipped", "engine_unavailable"}

PROCESSABLE_JOB_TYPES = {"ocr", "vlm", "image_embedding", "text_embedding", "event_rebuild"}


@dataclass(frozen=True)
class AnalysisPlanOptions:
    job_type: str
    start_date: str | None = None
    end_date: str | None = None
    all_dates: bool = False
    limit: int | None = None
    engine_name: str | None = None
    model_name: str | None = None
    model_path: str | None = None
    prompt_version: str | None = None
    analysis_version: str | None = None
    embedding_type: str = "combined_text"
    force: bool = False
    skip_existing: bool = False
    failed_only: bool = False
    engine_unavailable_only: bool = False
    version_changed_only: bool = False

    def to_scope(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisRunOptions(AnalysisPlanOptions):
    dry_run: bool = False
    job_id: str | None = None
    save_report: bool = False


@dataclass
class PlannedItem:
    item_id: str
    item_type: str
    existing_status: str | None = None
    needs_version_update: bool = False
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisPlan:
    options: AnalysisPlanOptions
    total_candidates: int = 0
    already_success: int = 0
    failed: int = 0
    engine_unavailable: int = 0
    version_changed: int = 0
    selected_items: list[PlannedItem] = field(default_factory=list)
    estimated_storage_bytes: int = 0
    estimated_processing_sec: float | None = None
    command_example: str | None = None

    def to_dict(self, *, include_items: bool = True) -> dict[str, Any]:
        payload = {
            "job_type": self.options.job_type,
            "target_scope": self.options.to_scope(),
            "total_candidates": self.total_candidates,
            "already_success": self.already_success,
            "failed": self.failed,
            "engine_unavailable": self.engine_unavailable,
            "version_changed": self.version_changed,
            "selected_count": len(self.selected_items),
            "estimated_storage_bytes": self.estimated_storage_bytes,
            "estimated_processing_sec": self.estimated_processing_sec,
            "command_example": self.command_example,
        }
        if include_items:
            payload["items"] = [item.to_dict() for item in self.selected_items]
        return payload
