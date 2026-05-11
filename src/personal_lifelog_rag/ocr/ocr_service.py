"""Batch OCR orchestration for local media items."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from personal_lifelog_rag.core.privacy import redact_text
from personal_lifelog_rag.ocr.base import OcrEngine
from personal_lifelog_rag.ocr.engines import get_ocr_engine
from personal_lifelog_rag.ocr.redaction import redact_ocr_text
from personal_lifelog_rag.ocr.schemas import OcrImagesReport, OcrResult


ANALYSIS_VERSION = "ocr_v1"


@dataclass(frozen=True)
class OcrImagesOptions:
    start_date: str | None = None
    end_date: str | None = None
    all_dates: bool = False
    limit: int = 100
    engine_name: str | None = None
    languages: list[str] | None = None
    dry_run: bool = False
    force: bool = False
    skip_existing: bool = False
    media_ids: list[str] | None = None
    text_cues_only: bool = False
    vlm_text_hint_only: bool = False
    has_vlm_text: bool = False
    ocr_priority: bool = False
    contains_text_hint: bool = False
    caption_keywords: tuple[str, ...] = ()
    min_vlm_confidence: float | None = None
    only_existing_files: bool = True


def run_ocr_images(
    repository,
    options: OcrImagesOptions,
    *,
    engine: OcrEngine | None = None,
    progress_callback=None,
) -> OcrImagesReport:
    """Run local OCR over selected media rows with safe skip behavior."""

    resolved_engine = engine or get_ocr_engine(options.engine_name)
    languages = options.languages or ["jpn", "eng"]
    rows = repository.list_media_items(
        start_date=options.start_date,
        end_date=options.end_date,
        limit=1_000_000,
    )
    if options.media_ids:
        allowed_ids = set(options.media_ids)
        rows = [row for row in rows if str(row.get("id")) in allowed_ids]
    rows = _filter_ocr_targets(repository, rows, options)[: max(options.limit, 0)]
    report = OcrImagesReport(
        selected_images=len(rows),
        dry_run=options.dry_run,
        engine=resolved_engine.name,
        languages=languages,
    )
    available = resolved_engine.is_available()
    for index, row in enumerate(rows, start=1):
        media_id = str(row["id"])
        if progress_callback is not None and not options.dry_run:
            progress_callback(f"OCR {index}/{len(rows)} {media_id}")
        existing = repository.get_media_ocr(media_id)
        if _should_skip_existing(existing, options):
            _add_report_row(report, row, "skipped", existing=existing)
            report.skipped += 1
            continue
        if options.dry_run:
            _add_report_row(report, row, "pending")
            continue
        if not available:
            result = OcrResult(
                engine=resolved_engine.name,
                status="engine_unavailable",
                error_message=f"OCR engine '{resolved_engine.name}' is not available",
            )
            _save_result(repository, row, result, languages)
            _add_report_row(report, row, result.status, result=result)
            report.engine_unavailable += 1
            report.processed += 1
            continue
        image_path = Path(str(row.get("file_path") or "")).expanduser()
        try:
            result = resolved_engine.recognize(image_path, languages)
        except Exception as exc:  # pragma: no cover - engine bugs should not stop a batch
            result = OcrResult(
                engine=resolved_engine.name,
                status="failed",
                error_message=f"OCR failed with {exc.__class__.__name__}",
            )
        _save_result(repository, row, result, languages)
        _add_report_row(report, row, result.status, result=result)
        report.processed += 1
        if result.status == "success":
            report.success += 1
        elif _is_no_text_status(result.status):
            report.no_text += 1
        elif result.status == "engine_unavailable":
            report.engine_unavailable += 1
        elif result.status == "failed":
            report.failed += 1
        else:
            report.skipped += 1
    return report


def ocr_stats(repository, *, start_date: str | None = None, end_date: str | None = None) -> dict[str, Any]:
    rows = repository.list_media_ocr(start_date=start_date, end_date=end_date, limit=1_000_000)
    status_counts: dict[str, int] = {}
    engine_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    daily_success: dict[str, int] = {}
    text_lengths: list[int] = []
    success_media_ids: set[str] = set()
    for row in rows:
        status = str(row.get("status") or "unknown")
        engine = str(row.get("ocr_engine") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        engine_counts[engine] = engine_counts.get(engine, 0) + 1
        for lang in str(row.get("ocr_languages") or "").split("+"):
            lang = lang.strip()
            if lang:
                language_counts[lang] = language_counts.get(lang, 0) + 1
        text = str(row.get("ocr_text") or "")
        if text:
            text_lengths.append(len(text))
        if status == "success":
            success_media_ids.add(str(row.get("media_id")))
            date_key = str(row.get("captured_at") or row.get("fallback_captured_at") or "")[:10]
            if date_key:
                daily_success[date_key] = daily_success.get(date_key, 0) + 1
    events = repository.list_events(start_date=start_date, end_date=end_date, include_hidden=True, limit=1_000_000)
    ocr_event_count = 0
    for event in events:
        evidence = repository.list_event_evidence(str(event["id"]))
        if any(row.get("evidence_type") == "photo" and str(row.get("evidence_id")) in success_media_ids for row in evidence):
            ocr_event_count += 1
    return {
        "range": {"from": start_date, "to": end_date},
        "total_media_ocr": len(rows),
        "ocr_done_photos": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "engine_counts": dict(sorted(engine_counts.items())),
        "language_counts": dict(sorted(language_counts.items())),
        "text_present_count": len(text_lengths),
        "average_text_length": round(sum(text_lengths) / len(text_lengths), 2) if text_lengths else 0.0,
        "text_length_distribution": _text_length_distribution(text_lengths),
        "daily_success_counts": dict(sorted(daily_success.items())),
        "ocr_event_count": ocr_event_count,
    }


def format_ocr_report(report: OcrImagesReport) -> str:
    return "\n".join(
        [
            "OCR summary:",
            f"- selected images: {report.selected_images}",
            f"- processed: {report.processed}",
            f"- success: {report.success}",
            f"- no_text_detected: {report.no_text}",
            f"- failed: {report.failed}",
            f"- skipped: {report.skipped}",
            f"- engine_unavailable: {report.engine_unavailable}",
            f"- engine: {report.engine}",
            f"- languages: {'+'.join(report.languages)}",
            f"- dry_run: {report.dry_run}",
        ]
    )


def format_ocr_stats(report: dict[str, Any]) -> str:
    lines = [
        "OCR Stats",
        f"- range: {report['range']['from'] or 'all'}..{report['range']['to'] or 'all'}",
        f"- OCR済み写真数: {report['ocr_done_photos']}",
        f"- textあり件数: {report['text_present_count']}",
        f"- average text length: {report['average_text_length']}",
        f"- OCRありイベント件数: {report['ocr_event_count']}",
        "status counts:",
    ]
    lines.extend(_counts(report["status_counts"]))
    lines.append("engine counts:")
    lines.extend(_counts(report["engine_counts"]))
    lines.append("language counts:")
    lines.extend(_counts(report.get("language_counts", {})))
    lines.append("text length distribution:")
    lines.extend(_counts(report.get("text_length_distribution", {})))
    lines.append("daily success counts:")
    lines.extend(_counts(report["daily_success_counts"]))
    return "\n".join(lines)


def format_ocr_show(rows: list[dict[str, Any]], *, full: bool = False, show_errors: bool = False) -> str:
    if not rows:
        return "OCR records: none"
    lines = [f"OCR records: {len(rows)}"]
    for row in rows:
        text = row.get("ocr_text") if full else row.get("ocr_text_redacted")
        preview = redact_ocr_text(str(text or ""), max_chars=None if full else 160)
        lines.extend(
            [
                "",
                f"- media_id: {row.get('media_id')}",
                f"  file_name: {redact_text(row.get('file_name'), max_chars=80)}",
                f"  captured_at: {row.get('captured_at') or row.get('fallback_captured_at') or ''}",
                f"  status: {row.get('status') or ''}",
                f"  engine: {row.get('ocr_engine') or ''}",
                f"  confidence: {row.get('confidence') if row.get('confidence') is not None else ''}",
                f"  text: {preview}",
            ]
        )
        if show_errors:
            lines.append(f"  error_message: {redact_text(row.get('error_message'), max_chars=400)}")
    return "\n".join(lines)


def _save_result(repository, media_row: dict[str, Any], result: OcrResult, languages: list[str]) -> None:
    text = result.text if result.status == "success" else None
    repository.upsert_media_ocr(
        media_id=str(media_row["id"]),
        ocr_text=text,
        ocr_text_redacted=redact_ocr_text(text) if text else None,
        ocr_engine=result.engine,
        ocr_languages=languages,
        confidence=result.confidence,
        blocks_json=result.to_blocks_json_rows(),
        status=result.status,
        error_message=result.error_message,
        analysis_version=ANALYSIS_VERSION,
    )


def _should_skip_existing(existing: dict[str, Any] | None, options: OcrImagesOptions) -> bool:
    if options.force or existing is None:
        return False
    if options.skip_existing:
        return str(existing.get("status") or "") == "success"
    return str(existing.get("status") or "") == "success"


def _filter_ocr_targets(repository, rows: list[dict[str, Any]], options: OcrImagesOptions) -> list[dict[str, Any]]:
    scored_rows: list[tuple[int, dict[str, Any]]] = []
    for row in rows:
        if str(row.get("media_type") or "image") != "image":
            continue
        file_path = Path(str(row.get("file_path") or "")).expanduser()
        if options.only_existing_files and not file_path.exists():
            continue
        media_id = str(row.get("id") or "")
        override = repository.get_media_vlm_override(media_id) or {}
        if int(override.get("is_hidden") or 0) or int(override.get("is_wrong") or 0):
            continue
        vlm = repository.get_media_vlm(media_id) or {}
        existing = repository.get_media_ocr(media_id)
        score, _ = _ocr_priority_score_and_reasons(vlm, caption_keywords=options.caption_keywords)
        text_hint = bool(vlm.get("contains_text_hint"))
        has_vlm_text = score > 0
        if options.contains_text_hint and not text_hint:
            continue
        if options.min_vlm_confidence is not None:
            try:
                confidence = float(vlm.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < options.min_vlm_confidence:
                continue
        if options.vlm_text_hint_only and not text_hint:
            continue
        if options.has_vlm_text and not has_vlm_text:
            continue
        if options.text_cues_only and not has_vlm_text:
            continue
        if options.ocr_priority:
            status = str((existing or {}).get("status") or "")
            if score <= 0:
                continue
            if status and status not in {"engine_unavailable", "failed"} and not options.force:
                continue
        scored_rows.append((score, row))
    if options.ocr_priority:
        scored_rows.sort(
            key=lambda item: (
                -item[0],
                str(item[1].get("captured_at") or item[1].get("fallback_captured_at") or ""),
                str(item[1].get("id") or ""),
            )
        )
    return [row for _, row in scored_rows]


def _ocr_priority_score(vlm: dict[str, Any]) -> int:
    return _ocr_priority_score_and_reasons(vlm)[0]


def _ocr_priority_score_and_reasons(vlm: dict[str, Any], *, caption_keywords: tuple[str, ...] = ()) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    text_cues = _json_list(vlm.get("text_cues_json"))
    if text_cues:
        score += 5
        reasons.append("text_cues")
    if vlm.get("contains_text_hint"):
        score += 4
        reasons.append("contains_text_hint")
    caption_blob = " ".join(
        str(vlm.get(key) or "")
        for key in (
            "caption",
            "short_caption",
            "scene_tags_json",
            "object_tags_json",
            "activity_tags_json",
            "food_cues_json",
            "location_cues_json",
            "text_cues_json",
        )
    ).lower()
    default_keywords = {
        "sign",
        "text",
        "menu",
        "receipt",
        "document",
        "label",
        "poster",
        "ticket",
        "screen",
        "screenshot",
        "storefront",
        "billboard",
        "handwritten",
        "logo",
        "package",
    }
    keywords = set(caption_keywords) if caption_keywords else default_keywords
    matched_keywords = sorted(keyword for keyword in keywords if keyword and keyword.lower() in caption_blob)
    score += len(matched_keywords)
    reasons.extend(f"caption:{keyword}" for keyword in matched_keywords[:8])
    return score, reasons


def ocr_priority_candidates(
    repository,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 50,
    caption_keywords: tuple[str, ...] = (),
    min_vlm_confidence: float | None = None,
) -> list[dict[str, Any]]:
    rows = repository.list_media_items(start_date=start_date, end_date=end_date, limit=1_000_000)
    candidates: list[tuple[int, dict[str, Any]]] = []
    for row in rows:
        if str(row.get("media_type") or "image") != "image":
            continue
        file_path = Path(str(row.get("file_path") or "")).expanduser()
        if not file_path.exists():
            continue
        media_id = str(row.get("id") or "")
        override = repository.get_media_vlm_override(media_id) or {}
        if int(override.get("is_hidden") or 0) or int(override.get("is_wrong") or 0):
            continue
        vlm = repository.get_media_vlm(media_id) or {}
        if min_vlm_confidence is not None:
            try:
                confidence = float(vlm.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < min_vlm_confidence:
                continue
        score, reasons = _ocr_priority_score_and_reasons(vlm, caption_keywords=caption_keywords)
        if score <= 0:
            continue
        existing = repository.get_media_ocr(media_id) or {}
        candidates.append(
            (
                score,
                {
                    "media_id": media_id,
                    "captured_at": row.get("captured_at") or row.get("fallback_captured_at") or "",
                    "file_name": row.get("file_name") or "",
                    "caption": redact_text(vlm.get("short_caption") or vlm.get("caption"), max_chars=120),
                    "text_cues": _json_list(vlm.get("text_cues_json")),
                    "priority_score": score,
                    "priority_reason": ", ".join(reasons),
                    "already_ocr_status": existing.get("status") or "",
                },
            )
        )
    candidates.sort(key=lambda item: (-item[0], str(item[1]["captured_at"]), str(item[1]["media_id"])))
    return [row for _, row in candidates[: max(limit, 0)]]


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _add_report_row(
    report: OcrImagesReport,
    media_row: dict[str, Any],
    status: str,
    *,
    result: OcrResult | None = None,
    existing: dict[str, Any] | None = None,
) -> None:
    report.rows.append(
        {
            "media_id": media_row.get("id"),
            "file_name": media_row.get("file_name"),
            "status": status,
            "engine": (result.engine if result else (existing or {}).get("ocr_engine") or report.engine),
            "text_preview": redact_ocr_text((result.text if result else (existing or {}).get("ocr_text")), max_chars=80),
        }
    )


def _counts(counts: dict[str, int]) -> list[str]:
    if not counts:
        return ["- none"]
    return [f"- {key}: {value}" for key, value in counts.items()]


def _is_no_text_status(status: str) -> bool:
    return status in {"no_text", "no_text_detected"}


def _text_length_distribution(lengths: list[int]) -> dict[str, int]:
    buckets = {"0-80": 0, "81-400": 0, "401-1200": 0, "1201+": 0}
    for length in lengths:
        if length <= 80:
            buckets["0-80"] += 1
        elif length <= 400:
            buckets["81-400"] += 1
        elif length <= 1200:
            buckets["401-1200"] += 1
        else:
            buckets["1201+"] += 1
    return buckets
