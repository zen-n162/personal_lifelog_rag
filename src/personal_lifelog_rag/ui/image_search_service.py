"""UI-facing helpers for image search results."""

from __future__ import annotations

from typing import Any

from personal_lifelog_rag.ui.thumbnail_gallery import media_thumbnail_gallery_items
from personal_lifelog_rag.vlm.schemas import ImageSearchOptions
from personal_lifelog_rag.vlm.vlm_service import format_image_search, image_search


def image_search_for_ui(
    repository,
    *,
    query: str,
    limit: int = 20,
) -> dict[str, Any]:
    report = image_search(repository, ImageSearchOptions(query=query, limit=limit))
    return {
        "report": report,
        "summary_text": format_image_search(report),
        "rows": image_search_rows(report),
        "gallery": image_search_gallery_items(report),
    }


def image_search_rows(report: dict[str, Any]) -> list[list[Any]]:
    return [
        [
            row.get("date") or "",
            row.get("media_id") or "",
            row.get("file_name") or "",
            row.get("captured_at") or "",
            row.get("caption") or "",
            row.get("ocr_preview") or "",
            ", ".join(row.get("matched_fields") or []),
            ", ".join(row.get("related_persons") or []),
            ", ".join(row.get("person_evidence_types") or []),
            row.get("person_score") or 0.0,
            row.get("thumbnail_path") or "",
        ]
        for row in report.get("results", [])
    ]


def image_search_gallery_items(report: dict[str, Any]) -> list[tuple[str, str]]:
    return media_thumbnail_gallery_items(list(report.get("results", [])), limit=100)
