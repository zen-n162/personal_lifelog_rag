"""Find and mark imported media whose original local files are unavailable."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from personal_lifelog_rag.core.privacy import redact_text
from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.db.schema import initialize_schema


@dataclass(frozen=True)
class MissingFileRow:
    media_id: str
    captured_at: str | None
    file_name: str | None
    file_path: str | None
    thumbnail_path: str | None
    missing_file: bool
    missing_thumbnail: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "media_id": self.media_id,
            "captured_at": self.captured_at,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "thumbnail_path": self.thumbnail_path,
            "missing_file": self.missing_file,
            "missing_thumbnail": self.missing_thumbnail,
        }


@dataclass
class MissingFilesReport:
    total_media: int = 0
    missing_files: int = 0
    missing_thumbnails: int = 0
    rows: list[MissingFileRow] = field(default_factory=list)
    exported_path: str | None = None
    marked_unavailable: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_media": self.total_media,
            "missing_files": self.missing_files,
            "missing_thumbnails": self.missing_thumbnails,
            "exported_path": self.exported_path,
            "marked_unavailable": self.marked_unavailable,
            "rows": [row.to_dict() for row in self.rows],
        }


def collect_missing_files(repository: LifelogRepository, *, limit: int = 50) -> MissingFilesReport:
    """Return missing original files and count missing thumbnails separately."""

    rows = repository.list_media_items(limit=1_000_000)
    report = MissingFilesReport(total_media=len(rows))
    sample_limit = max(limit, 0)
    for row in rows:
        file_path = str(row.get("file_path") or "")
        thumbnail_path = str(row.get("thumbnail_path") or "")
        missing_file = bool(file_path) and not Path(file_path).expanduser().exists()
        missing_thumbnail = bool(thumbnail_path) and not Path(thumbnail_path).expanduser().exists()
        if missing_file:
            report.missing_files += 1
        if missing_thumbnail:
            report.missing_thumbnails += 1
        if (missing_file or missing_thumbnail) and len(report.rows) < sample_limit:
            report.rows.append(
                MissingFileRow(
                    media_id=str(row.get("id") or ""),
                    captured_at=str(row.get("captured_at") or row.get("fallback_captured_at") or "") or None,
                    file_name=str(row.get("file_name") or "") or None,
                    file_path=file_path or None,
                    thumbnail_path=thumbnail_path or None,
                    missing_file=missing_file,
                    missing_thumbnail=missing_thumbnail,
                )
            )
    return report


def export_missing_files(report: MissingFilesReport, output_path: Path) -> Path:
    """Write a CSV report for local operational cleanup."""

    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "media_id",
                "captured_at",
                "file_name",
                "file_path",
                "thumbnail_path",
                "missing_file",
                "missing_thumbnail",
            ],
        )
        writer.writeheader()
        for row in report.rows:
            writer.writerow(row.to_dict())
    report.exported_path = str(output_path)
    return output_path


def mark_missing_media_unavailable(repository: LifelogRepository) -> int:
    """Mark all media rows with missing original files as unavailable in analysis_json."""

    rows = repository.list_media_items(limit=1_000_000)
    checked_at = datetime.now(timezone.utc).isoformat()
    updated = 0
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        for row in rows:
            file_path = str(row.get("file_path") or "")
            if not file_path or Path(file_path).expanduser().exists():
                continue
            analysis_json = _analysis_json_with_unavailable(row.get("analysis_json"), checked_at=checked_at)
            cursor = connection.execute(
                "UPDATE media_items SET analysis_json = ? WHERE id = ?",
                (json.dumps(analysis_json, ensure_ascii=False, sort_keys=True), row.get("id")),
            )
            updated += cursor.rowcount
        connection.commit()
    return updated


def format_missing_files(report: MissingFilesReport, *, mark_unavailable: bool = False, yes: bool = False) -> str:
    lines = [
        "Missing Files",
        f"- total media: {report.total_media}",
        f"- missing original files: {report.missing_files}",
        f"- missing thumbnails: {report.missing_thumbnails}",
    ]
    if report.exported_path:
        lines.append(f"- exported: {report.exported_path}")
    if mark_unavailable:
        if yes:
            lines.append(f"- marked unavailable: {report.marked_unavailable}")
        else:
            lines.append("- mark unavailable: dry-run only; pass --yes to write analysis_json flags")
    if report.rows:
        lines.extend(["", "sample:"])
        for row in report.rows:
            flags = []
            if row.missing_file:
                flags.append("missing_file")
            if row.missing_thumbnail:
                flags.append("missing_thumbnail")
            lines.append(
                "- "
                + " | ".join(
                    [
                        row.media_id,
                        row.captured_at or "",
                        row.file_name or "",
                        ",".join(flags),
                        redact_text(row.file_path or "", max_chars=120),
                    ]
                )
            )
    return "\n".join(lines)


def _analysis_json_with_unavailable(value: Any, *, checked_at: str) -> dict[str, Any]:
    if isinstance(value, dict):
        payload = dict(value)
    else:
        try:
            payload = json.loads(str(value)) if value else {}
            if not isinstance(payload, dict):
                payload = {}
        except (TypeError, ValueError):
            payload = {}
    payload.update(
        {
            "file_unavailable": True,
            "file_unavailable_reason": "missing_file",
            "file_unavailable_checked_at": checked_at,
        }
    )
    return payload
