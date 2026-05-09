"""Batch VLM orchestration and image-content search."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import traceback
from typing import Any

from personal_lifelog_rag.core.privacy import redact_text
from personal_lifelog_rag.retrieval.visual_query_expansion import expand_visual_query_terms
from personal_lifelog_rag.vlm.base import VlmEngine
from personal_lifelog_rag.vlm.engines import get_vlm_engine
from personal_lifelog_rag.vlm.prompts import (
    SAFE_IMAGE_ANALYSIS_PROMPT,
    SAFE_IMAGE_ANALYSIS_PROMPT_VERSION,
    get_vlm_prompt_template,
)
from personal_lifelog_rag.vlm.safety import sanitize_vlm_result
from personal_lifelog_rag.vlm.review_service import apply_vlm_override_to_result, should_use_vlm_for_search
from personal_lifelog_rag.vlm.schemas import ImageSearchOptions, VlmImagesReport, VlmResult


ANALYSIS_VERSION = "vlm_v1"


@dataclass(frozen=True)
class VlmImagesOptions:
    start_date: str | None = None
    end_date: str | None = None
    all_dates: bool = False
    limit: int = 100
    engine_name: str | None = None
    model_name: str | None = None
    dry_run: bool = False
    force: bool = False
    skip_existing: bool = False
    only_with_ocr: bool = False
    only_gps: bool = False
    prompt_template: str | None = None
    media_ids: list[str] | None = None


def run_vlm_images(
    repository,
    options: VlmImagesOptions,
    *,
    engine: VlmEngine | None = None,
    progress_callback=None,
) -> VlmImagesReport:
    """Run local VLM analysis over selected media rows with safe skip behavior."""

    resolved_engine = engine or get_vlm_engine(options.engine_name, model_name=options.model_name)
    rows = repository.list_media_items(
        start_date=options.start_date,
        end_date=options.end_date,
        limit=1_000_000,
    )
    rows = _filter_targets(repository, rows, options)[: max(options.limit, 0)]
    report = VlmImagesReport(
        selected_images=len(rows),
        dry_run=options.dry_run,
        engine=resolved_engine.name,
        model_name=getattr(resolved_engine, "model_name", None),
    )
    available = resolved_engine.is_available()
    prompt_template = get_vlm_prompt_template(options.prompt_template)
    for index, row in enumerate(rows, start=1):
        media_id = str(row["id"])
        if progress_callback is not None and not options.dry_run:
            progress_callback(f"VLM {index}/{len(rows)} {media_id}")
        existing = repository.get_media_vlm(media_id)
        if _should_skip_existing(existing, options):
            _add_report_row(report, row, "skipped", existing=existing)
            report.skipped += 1
            continue
        if options.dry_run:
            _add_report_row(report, row, "pending")
            continue
        if not available:
            result = VlmResult(
                engine=resolved_engine.name,
                model_name=getattr(resolved_engine, "model_name", None),
                status="engine_unavailable",
                error_message=f"VLM engine '{resolved_engine.name}' is not available",
            )
            _save_result(repository, row, result)
            _count_result(report, result.status)
            _add_report_row(report, row, result.status, result=result)
            report.processed += 1
            continue
        image_path = Path(str(row.get("file_path") or "")).expanduser()
        if not image_path.exists():
            result = VlmResult(
                engine=resolved_engine.name,
                model_name=getattr(resolved_engine, "model_name", None),
                status="failed",
                error_message="image file does not exist",
            )
            _save_result(repository, row, result)
            _count_result(report, result.status)
            _add_report_row(report, row, result.status, result=result)
            report.processed += 1
            continue
        try:
            result = sanitize_vlm_result(resolved_engine.analyze_image(image_path, prompt_template.prompt))
            result = VlmResult(**{**result.to_dict(), "prompt_version": prompt_template.name})
        except Exception as exc:  # pragma: no cover - engine bugs should not stop a batch
            result = VlmResult(
                engine=resolved_engine.name,
                model_name=getattr(resolved_engine, "model_name", None),
                status="failed",
                error_message=_exception_message("VLM failed", exc),
            )
        _save_result(repository, row, result)
        _count_result(report, result.status)
        _add_report_row(report, row, result.status, result=result)
        report.processed += 1
    return report


def vlm_stats(repository, *, start_date: str | None = None, end_date: str | None = None) -> dict[str, Any]:
    rows = repository.list_media_vlm(start_date=start_date, end_date=end_date, limit=1_000_000)
    status_counts: dict[str, int] = {}
    engine_counts: dict[str, int] = {}
    daily_success: dict[str, int] = {}
    scene_tags: dict[str, int] = {}
    object_tags: dict[str, int] = {}
    activity_tags: dict[str, int] = {}
    food_cues: dict[str, int] = {}
    people_present_count = 0
    ocr_and_vlm_count = 0
    for row in rows:
        status = str(row.get("status") or "unknown")
        engine = str(row.get("vlm_engine") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        engine_counts[engine] = engine_counts.get(engine, 0) + 1
        _add_tag_counts(scene_tags, row.get("scene_tags_json"))
        _add_tag_counts(object_tags, row.get("object_tags_json"))
        _add_tag_counts(activity_tags, row.get("activity_tags_json"))
        _add_tag_counts(food_cues, row.get("food_cues_json"))
        if int(row.get("people_count") or 0) > 0 or "people_present" in _json_list(row.get("safety_flags_json")):
            people_present_count += 1
        if row.get("ocr_text") and status == "success":
            ocr_and_vlm_count += 1
        if status == "success":
            date_key = str(row.get("captured_at") or row.get("fallback_captured_at") or "")[:10]
            if date_key:
                daily_success[date_key] = daily_success.get(date_key, 0) + 1
    return {
        "range": {"from": start_date, "to": end_date},
        "total_media_vlm": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "engine_counts": dict(sorted(engine_counts.items())),
        "scene_tags_top": _top_counts(scene_tags),
        "object_tags_top": _top_counts(object_tags),
        "activity_tags_top": _top_counts(activity_tags),
        "food_cues_top": _top_counts(food_cues),
        "people_present_count": people_present_count,
        "ocr_and_vlm_count": ocr_and_vlm_count,
        "daily_success_counts": dict(sorted(daily_success.items())),
    }


def image_search(repository, options: ImageSearchOptions) -> dict[str, Any]:
    terms = expand_visual_query_terms(options.query)
    records = repository.search_text_records(
        terms=terms,
        start_date=options.date_from,
        end_date=options.date_to,
        limit=20_000,
        include_hidden=options.include_hidden,
    )
    rows_by_media: dict[str, dict[str, Any]] = {}
    for media in records.get("media_items", []):
        media_id = str(media.get("id") or media.get("media_id") or "")
        if media_id:
            rows_by_media.setdefault(media_id, {}).update(media)
    for ocr in records.get("media_ocr", []):
        media_id = str(ocr.get("media_id") or ocr.get("media_item_id") or "")
        if media_id:
            rows_by_media.setdefault(media_id, {}).update(ocr)
    for vlm in records.get("media_vlm", []):
        media_id = str(vlm.get("media_id") or vlm.get("media_item_id") or "")
        if media_id:
            rows_by_media.setdefault(media_id, {}).update(vlm)
    for media_id, row in list(rows_by_media.items()):
        _merge_vlm_override_columns(repository, media_id, row)

    effective_rows = [
        apply_vlm_override_to_result(row)
        for row in rows_by_media.values()
        if should_use_vlm_for_search(row, include_hidden=options.include_hidden)
    ]
    effective_rows = [row for row in effective_rows if _row_matches_visual_terms(row, raw_query=options.query, terms=terms)]
    results = [_image_result(repository, row, options.query, terms=terms) for row in effective_rows]
    results = [row for row in results if row["date"]]
    results.sort(key=lambda row: (-row["score"], row["date"], row["media_id"]))
    return {
        "query": options.query,
        "backend": "sqlite_like",
        "expanded_terms": terms,
        "date_from": options.date_from,
        "date_to": options.date_to,
        "total": len(results),
        "results": results[: max(options.limit, 0)],
    }


def _merge_vlm_override_columns(repository, media_id: str, row: dict[str, Any]) -> None:
    override = repository.get_media_vlm_override(media_id)
    if not override:
        return
    row.update(
        {
            "caption_override": override.get("caption_override"),
            "short_caption_override": override.get("short_caption_override"),
            "scene_tags_override_json": override.get("scene_tags_override_json"),
            "object_tags_override_json": override.get("object_tags_override_json"),
            "activity_tags_override_json": override.get("activity_tags_override_json"),
            "food_cues_override_json": override.get("food_cues_override_json"),
            "location_cues_override_json": override.get("location_cues_override_json"),
            "vlm_is_verified": override.get("is_verified", 0),
            "vlm_is_hidden": override.get("is_hidden", 0),
            "vlm_is_wrong": override.get("is_wrong", 0),
            "vlm_is_searchable": override.get("is_searchable", 1),
            "vlm_is_event_usable": override.get("is_event_usable", 1),
            "vlm_review_status": override.get("review_status") or "unreviewed",
            "vlm_review_note": override.get("review_note"),
        }
    )


def format_vlm_report(report: VlmImagesReport) -> str:
    return "\n".join(
        [
            "VLM analysis summary:",
            f"- selected images: {report.selected_images}",
            f"- processed: {report.processed}",
            f"- success: {report.success}",
            f"- failed: {report.failed}",
            f"- skipped: {report.skipped}",
            f"- no_visual_content: {report.no_visual_content}",
            f"- engine_unavailable: {report.engine_unavailable}",
            f"- engine: {report.engine}",
            f"- model: {report.model_name or ''}",
            f"- dry_run: {report.dry_run}",
            f"- file(s): {report.selected_images}",
        ]
    )


def format_vlm_stats(report: dict[str, Any]) -> str:
    lines = [
        "VLM Stats",
        f"- range: {report['range']['from'] or 'all'}..{report['range']['to'] or 'all'}",
        f"- media_vlm total: {report['total_media_vlm']}",
        f"- people_present: {report['people_present_count']}",
        f"- OCRあり+VLMあり: {report['ocr_and_vlm_count']}",
        "status counts:",
    ]
    lines.extend(_counts(report["status_counts"]))
    lines.append("engine counts:")
    lines.extend(_counts(report["engine_counts"]))
    lines.append("scene tags top:")
    lines.extend(_counts(report["scene_tags_top"]))
    lines.append("object tags top:")
    lines.extend(_counts(report["object_tags_top"]))
    lines.append("activity tags top:")
    lines.extend(_counts(report["activity_tags_top"]))
    lines.append("food cues top:")
    lines.extend(_counts(report["food_cues_top"]))
    lines.append("daily success counts:")
    lines.extend(_counts(report["daily_success_counts"]))
    return "\n".join(lines)


def format_vlm_show(rows: list[dict[str, Any]], *, full: bool = False, show_errors: bool = False) -> str:
    if not rows:
        return "VLM records: none"
    lines = [f"VLM records: {len(rows)}"]
    for row in rows:
        caption = row.get("caption") if full else (row.get("short_caption") or row.get("caption"))
        lines.extend(
            [
                "",
                f"- media_id: {row.get('media_id')}",
                f"  file_name: {redact_text(row.get('file_name'), max_chars=80)}",
                f"  captured_at: {row.get('captured_at') or row.get('fallback_captured_at') or ''}",
                f"  status: {row.get('status') or ''}",
                f"  engine: {row.get('vlm_engine') or ''}",
                f"  prompt_template: {row.get('prompt_version') or ''}",
                f"  analyzed_at: {row.get('analyzed_at') or ''}",
                f"  confidence: {row.get('confidence') if row.get('confidence') is not None else ''}",
                f"  caption: {redact_text(caption, max_chars=240 if full else 120)}",
                f"  scene_tags: {', '.join(_json_list(row.get('scene_tags_json')))}",
                f"  object_tags: {', '.join(_json_list(row.get('object_tags_json')))}",
                f"  activity_tags: {', '.join(_json_list(row.get('activity_tags_json')))}",
                f"  food_cues: {', '.join(_json_list(row.get('food_cues_json')))}",
                f"  location_cues: {', '.join(_json_list(row.get('location_cues_json')))}",
                f"  text_cues: {', '.join(_json_list(row.get('text_cues_json')))}",
                f"  evidence_strength: {row.get('evidence_strength') or 'weak'}",
                f"  safety_flags: {', '.join(_json_list(row.get('safety_flags_json')))}",
            ]
        )
        if show_errors and row.get("error_message"):
            lines.append(f"  error_message: {redact_text(row.get('error_message'), max_chars=4000 if full else 1200)}")
    return "\n".join(lines)


def format_image_search(report: dict[str, Any]) -> str:
    if not report["results"]:
        return f"画像検索結果は見つかりませんでした: {report['query']}"
    lines = [
        f"Image Search: {report['query']}",
        f"- backend: {report['backend']}",
        f"- results: {report['total']}",
        "- note: VLM/OCR evidence is local automatic analysis; verify photos before treating it as fact.",
    ]
    for index, row in enumerate(report["results"], start=1):
        lines.append(
            f"{index}. {row['date']} {row['media_id']} confidence={row['confidence']} "
            f"fields={', '.join(row['matched_fields'])}"
        )
        lines.append(f"   file: {row['file_name']}")
        if row.get("caption"):
            lines.append(f"   caption: {row['caption']}")
        if row.get("ocr_preview"):
            lines.append(f"   OCR: {row['ocr_preview']}")
        if row.get("related_event"):
            lines.append(f"   event: {row['related_event']}")
        if row.get("food_cues"):
            lines.append(f"   food_cues: {', '.join(row['food_cues'])}")
        if row.get("evidence_strength"):
            lines.append(f"   evidence_strength: {row['evidence_strength']}")
    return "\n".join(lines)


def _save_result(repository, media_row: dict[str, Any], result: VlmResult) -> None:
    repository.upsert_media_vlm(
        media_id=str(media_row["id"]),
        caption=result.caption if result.status == "success" else None,
        short_caption=result.short_caption if result.status == "success" else None,
        scene_tags=result.scene_tags,
        object_tags=result.object_tags,
        activity_tags=result.activity_tags,
        location_cues=result.location_cues,
        food_cues=result.food_cues,
        text_cues=result.text_cues,
        uncertainty_notes=result.uncertainty_notes,
        evidence_strength=result.evidence_strength,
        people_count=result.people_count,
        contains_text_hint=result.contains_text_hint,
        safety_flags=result.safety_flags,
        vlm_engine=result.engine,
        model_name=result.model_name,
        prompt_version=result.prompt_version or SAFE_IMAGE_ANALYSIS_PROMPT_VERSION,
        confidence=result.confidence,
        status=result.status,
        error_message=result.error_message,
        analysis_version=ANALYSIS_VERSION,
    )


def _filter_targets(repository, rows: list[dict[str, Any]], options: VlmImagesOptions) -> list[dict[str, Any]]:
    media_id_order: dict[str, int] | None = None
    if options.media_ids is not None:
        media_id_order = {str(media_id): index for index, media_id in enumerate(options.media_ids)}

    filtered: list[dict[str, Any]] = []
    for row in rows:
        if media_id_order is not None and str(row.get("id")) not in media_id_order:
            continue
        if options.only_gps and (row.get("gps_lat") is None or row.get("gps_lon") is None):
            continue
        if options.only_with_ocr:
            ocr = repository.get_media_ocr(str(row["id"]))
            if not ocr or ocr.get("status") != "success":
                continue
        filtered.append(row)
    if media_id_order is not None:
        filtered.sort(key=lambda row: media_id_order.get(str(row.get("id")), 1_000_000))
    return filtered


def _should_skip_existing(existing: dict[str, Any] | None, options: VlmImagesOptions) -> bool:
    if options.force or existing is None:
        return False
    if options.skip_existing:
        return str(existing.get("status") or "") == "success"
    return str(existing.get("status") or "") == "success"


def _count_result(report: VlmImagesReport, status: str) -> None:
    if status == "success":
        report.success += 1
    elif status == "failed":
        report.failed += 1
    elif status == "engine_unavailable":
        report.engine_unavailable += 1
    elif status == "no_visual_content":
        report.no_visual_content += 1
    else:
        report.skipped += 1


def _add_report_row(
    report: VlmImagesReport,
    media_row: dict[str, Any],
    status: str,
    *,
    result: VlmResult | None = None,
    existing: dict[str, Any] | None = None,
) -> None:
    report.rows.append(
        {
            "media_id": media_row.get("id"),
            "file_name": media_row.get("file_name"),
            "status": status,
            "engine": (result.engine if result else (existing or {}).get("vlm_engine") or report.engine),
            "caption_preview": redact_text(
                (result.short_caption or result.caption if result else (existing or {}).get("short_caption") or (existing or {}).get("caption")),
                max_chars=80,
            ),
        }
    )


def _image_result(repository, row: dict[str, Any], query: str, *, terms: list[str] | None = None) -> dict[str, Any]:
    media_id = str(row.get("media_id") or row.get("media_item_id") or row.get("id") or "")
    timestamp = str(row.get("captured_at") or row.get("fallback_captured_at") or "")
    fields = _matched_fields(row, query, terms=terms)
    caption = redact_text(row.get("short_caption") or row.get("caption"), max_chars=120)
    ocr_preview = redact_text(row.get("ocr_text_redacted") or row.get("ocr_text"), max_chars=100)
    score = 0.2 + len(fields) * 0.15
    if row.get("confidence") is not None:
        score += min(float(row.get("confidence") or 0.0), 1.0) * 0.25
    if row.get("gps_lat") is not None and row.get("gps_lon") is not None:
        score += 0.1
    related_event = _related_event(repository, media_id, timestamp[:10])
    if related_event:
        score += 0.1
    if row.get("is_verified") or row.get("review_status") == "accepted":
        score += 0.08
    if row.get("is_wrong") or row.get("review_status") in {"rejected", "wrong"}:
        score -= 0.5
    final_score = max(0.0, min(score, 0.95))
    return {
        "date": timestamp[:10],
        "media_id": media_id,
        "file_name": redact_text(row.get("file_name"), max_chars=80),
        "captured_at": timestamp,
        "thumbnail_path": row.get("thumbnail_path") or "",
        "caption": caption,
        "ocr_preview": ocr_preview,
        "matched_fields": fields or ["unknown"],
        "matched_terms": _matched_terms(row, terms or [query]),
        "confidence": _confidence_label(final_score),
        "confidence_score": round(final_score, 3),
        "score": round(final_score, 3),
        "related_event": related_event,
        "evidence_types": _evidence_types(row),
        "evidence_strength": _image_evidence_strength(row, related_event=related_event),
        "scene_tags": _json_list(row.get("scene_tags_json")),
        "activity_tags": _json_list(row.get("activity_tags_json")),
        "food_cues": _json_list(row.get("food_cues_json")),
        "location_cues": _json_list(row.get("location_cues_json")),
        "review_status": row.get("review_status") or "unreviewed",
        "is_verified": int(row.get("is_verified") or 0),
        "is_hidden": int(row.get("is_hidden") or 0),
        "is_wrong": int(row.get("is_wrong") or 0),
        "is_searchable": int(row.get("is_searchable") if row.get("is_searchable") is not None else 1),
        "is_event_usable": int(row.get("is_event_usable") if row.get("is_event_usable") is not None else 1),
    }


def _exception_message(prefix: str, exc: Exception, *, max_chars: int = 4000) -> str:
    traceback_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    message = f"{prefix}: {exc.__class__.__name__}: {exc}\n{traceback_text}".strip()
    return message[:max_chars]


def _matched_fields(row: dict[str, Any], query: str, *, terms: list[str] | None = None) -> list[str]:
    search_terms = [term for term in (terms or [query]) if str(term).strip()]
    fields = {
        "caption": row.get("caption"),
        "short_caption": row.get("short_caption"),
        "scene_tags": row.get("scene_tags_json"),
        "object_tags": row.get("object_tags_json"),
        "activity_tags": row.get("activity_tags_json"),
        "location_cues": row.get("location_cues_json"),
        "food_cues": row.get("food_cues_json"),
        "text_cues": row.get("text_cues_json"),
        "ocr": row.get("ocr_text") or row.get("ocr_text_redacted"),
        "file_name": row.get("file_name"),
    }
    return [name for name, value in fields.items() if any(term in str(value or "") for term in search_terms)]


def _row_matches_visual_terms(row: dict[str, Any], *, raw_query: str, terms: list[str]) -> bool:
    visual_haystack = " ".join(
        str(row.get(key) or "")
        for key in (
            "caption",
            "short_caption",
            "scene_tags_json",
            "object_tags_json",
            "activity_tags_json",
            "location_cues_json",
            "food_cues_json",
            "text_cues_json",
            "ocr_text",
            "ocr_text_redacted",
        )
    )
    if any(term and term in visual_haystack for term in terms):
        return True
    # File names are allowed only for the literal query to avoid UUID-like false
    # positives such as "CAFE" inside camera-generated names.
    literal = (raw_query or "").strip()
    return bool(literal and literal in str(row.get("file_name") or ""))


def _matched_terms(row: dict[str, Any], terms: list[str]) -> list[str]:
    haystack = " ".join(
        str(row.get(key) or "")
        for key in (
            "caption",
            "short_caption",
            "scene_tags_json",
            "object_tags_json",
            "activity_tags_json",
            "location_cues_json",
            "food_cues_json",
            "text_cues_json",
            "ocr_text",
            "ocr_text_redacted",
            "file_name",
        )
    )
    matched: list[str] = []
    for term in terms:
        term = str(term or "").strip()
        if term and term in haystack and term not in matched:
            matched.append(term)
    return matched[:20]


def _related_event(repository, media_id: str, date_value: str) -> str | None:
    if not media_id or not date_value:
        return None
    for event in repository.list_events(start_date=date_value, end_date=date_value, include_hidden=False, limit=1000):
        evidence = repository.list_event_evidence(str(event["id"]))
        if any(row.get("evidence_type") == "photo" and row.get("evidence_id") == media_id for row in evidence):
            return redact_text(f"{event.get('start_time') or ''} {event.get('title') or ''}".strip(), max_chars=80)
    return None


def _evidence_types(row: dict[str, Any]) -> list[str]:
    types = []
    if row.get("caption") or row.get("scene_tags_json") or row.get("food_cues_json"):
        types.append("vlm")
    if row.get("ocr_text") or row.get("ocr_text_redacted"):
        types.append("ocr")
    if row.get("file_name"):
        types.append("photo")
    return types or ["photo"]


def _image_evidence_strength(row: dict[str, Any], *, related_event: str | None) -> str:
    types = set(_evidence_types(row))
    if row.get("is_verified") or row.get("review_status") == "accepted":
        return "medium"
    if "vlm" in types and related_event:
        return "medium"
    if "vlm" in types and "ocr" in types:
        return "medium"
    return "weak"


def _add_tag_counts(counts: dict[str, int], raw: Any) -> None:
    for tag in _json_list(raw):
        counts[tag] = counts.get(tag, 0) + 1


def _top_counts(counts: dict[str, int], *, limit: int = 20) -> dict[str, int]:
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit])


def _json_list(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw]
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return [str(raw)]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    return [str(parsed)] if str(parsed).strip() else []


def _counts(counts: dict[str, int]) -> list[str]:
    if not counts:
        return ["- none"]
    return [f"- {key}: {value}" for key, value in counts.items()]


def _confidence_label(score: float) -> str:
    if score >= 0.75:
        return "高"
    if score >= 0.45:
        return "中"
    return "低"
