"""Image OCR/VLM analysis orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from personal_lifelog_rag.captioning.local_vlm import LocalVLMAdapter, VLMAnalysisResult, get_vlm_adapter
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.ocr.local_ocr import LocalOCRAdapter, OCRResult, get_ocr_adapter


@dataclass
class ImageAnalysisReport:
    scanned: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0
    ocr_used: int = 0
    vlm_used: int = 0
    reason: str | None = None


def analyze_images(
    repository: LifelogRepository,
    *,
    limit: int = 100,
    ocr_adapter: LocalOCRAdapter | None = None,
    vlm_adapter: LocalVLMAdapter | None = None,
) -> ImageAnalysisReport:
    """Analyze local media rows with optional local OCR/VLM engines."""

    ocr = ocr_adapter or get_ocr_adapter()
    vlm = vlm_adapter or get_vlm_adapter()
    rows = repository.list_media_items_for_analysis(limit=limit)
    report = ImageAnalysisReport(scanned=len(rows))

    if not getattr(ocr, "available", True) and not getattr(vlm, "available", True):
        report.skipped = report.scanned
        report.reason = "未解析: OCR/VLM backends are not configured"
        return report

    for row in rows:
        image_path = row.get("file_path")
        if not image_path or not Path(image_path).exists():
            report.skipped += 1
            continue

        try:
            ocr_result = ocr.extract_text(image_path)
            vlm_result = vlm.analyze_image(image_path, ocr_text=ocr_result.text)
        except Exception:  # pragma: no cover - adapter bugs should not stop a batch
            report.errors += 1
            continue

        if ocr_result.skipped and vlm_result.skipped:
            report.skipped += 1
            continue

        analysis = build_analysis_json(ocr_result, vlm_result)
        repository.update_media_analysis(
            str(row["id"]),
            caption=vlm_result.caption,
            ocr_text=ocr_result.text,
            analysis=analysis,
        )
        report.updated += 1
        if ocr_result.text:
            report.ocr_used += 1
        if not vlm_result.skipped:
            report.vlm_used += 1

    return report


def build_analysis_json(ocr_result: OCRResult, vlm_result: VLMAnalysisResult) -> dict[str, Any]:
    fields = vlm_result.to_analysis_fields()
    fields["text_in_image"] = ocr_result.text or fields.get("text_in_image")
    fields["ocr"] = _status_payload(ocr_result.engine, ocr_result.skipped, ocr_result.reason)
    fields["vlm"] = _status_payload(vlm_result.engine, vlm_result.skipped, vlm_result.reason)
    if vlm_result.raw:
        fields["raw"] = vlm_result.raw
    return fields


def format_analysis_report(report: ImageAnalysisReport) -> str:
    suffix = f", reason={report.reason}" if report.reason else ""
    return (
        "Analyzed images: "
        f"{report.updated} updated, {report.skipped} 未解析/skipped, "
        f"{report.errors} error(s), {report.scanned} file(s), "
        f"OCR text={report.ocr_used}, VLM={report.vlm_used}"
        f"{suffix}"
    )


def _status_payload(engine: str, skipped: bool, reason: str | None) -> dict[str, Any]:
    return {
        "engine": engine,
        "status": "skipped" if skipped else "ok",
        "reason": reason,
    }

