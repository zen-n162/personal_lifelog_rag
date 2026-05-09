"""Schemas for research/portfolio reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


ReportMode = Literal["public", "private"]


@dataclass(frozen=True)
class ReportOptions:
    start_date: str | None = None
    end_date: str | None = None
    mode: ReportMode = "public"
    eval_path: Path | None = None
    eval_run: Path | None = None
    include_examples: bool = False
    save_json: bool = False

    @property
    def public(self) -> bool:
        return self.mode == "public"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["eval_path"] = str(self.eval_path) if self.eval_path else None
        payload["eval_run"] = str(self.eval_run) if self.eval_run else None
        return payload


@dataclass(frozen=True)
class ReportWriteResult:
    markdown_path: Path
    json_path: Path | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "markdown_path": str(self.markdown_path),
            "json_path": str(self.json_path) if self.json_path else None,
        }
