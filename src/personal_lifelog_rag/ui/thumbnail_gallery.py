"""Shared local thumbnail gallery helpers for the Gradio UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def sort_media_by_capture_date(rows: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """Return media rows in chronological order for thumbnail galleries."""

    return sorted(
        [dict(row) for row in rows],
        key=lambda row: (
            _date_value(row),
            str(row.get("media_id") or row.get("id") or ""),
            str(row.get("file_name") or ""),
        ),
    )


def media_thumbnail_gallery_items(
    rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    limit: int = 50,
) -> list[tuple[str, str]]:
    """Build Gradio Gallery items from local thumbnails, sorted by date."""

    items: list[tuple[str, str]] = []
    for row in sort_media_by_capture_date(rows):
        thumbnail_path = str(row.get("thumbnail_path") or "")
        if not thumbnail_path:
            continue
        resolved_path = Path(thumbnail_path).expanduser()
        if not resolved_path.exists():
            continue
        items.append((str(resolved_path), _gallery_caption(row)))
        if len(items) >= limit:
            break
    return items


def _date_value(row: dict[str, Any]) -> str:
    return str(
        row.get("captured_at")
        or row.get("fallback_captured_at")
        or row.get("date")
        or ""
    )


def _gallery_caption(row: dict[str, Any]) -> str:
    date = _date_value(row)[:10]
    media_id = str(row.get("media_id") or row.get("id") or "")
    caption = str(row.get("caption") or row.get("file_name") or "")
    parts = [part for part in [date, media_id, caption] if part]
    return " / ".join(parts)
