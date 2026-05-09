"""Planning helpers for local analysis jobs."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.embeddings.repository import MediaEmbeddingRepository
from personal_lifelog_rag.jobs.schemas import AnalysisPlan, AnalysisPlanOptions, PlannedItem
from personal_lifelog_rag.ocr.ocr_service import ANALYSIS_VERSION as OCR_ANALYSIS_VERSION
from personal_lifelog_rag.vlm.vlm_service import ANALYSIS_VERSION as VLM_ANALYSIS_VERSION


STORAGE_BYTES_PER_ITEM = {
    "ocr": 4_096,
    "vlm": 8_192,
    "image_embedding": 32_768,
    "text_embedding": 16_384,
    "event_rebuild": 2_048,
}

SECONDS_PER_ITEM = {
    "ocr": 1.0,
    "vlm": 8.0,
    "image_embedding": 2.0,
    "text_embedding": 0.5,
    "event_rebuild": 1.0,
}


def plan_analysis(repository: LifelogRepository, options: AnalysisPlanOptions) -> AnalysisPlan:
    """Return selected items and conservative processing estimates."""

    if options.job_type == "event_rebuild":
        return _plan_event_rebuild(repository, options)
    media_rows = [
        row
        for row in repository.list_media_items(
            start_date=options.start_date,
            end_date=options.end_date,
            limit=1_000_000,
        )
        if str(row.get("media_type") or "image") == "image"
    ]
    candidates: list[PlannedItem] = []
    already_success = 0
    failed = 0
    engine_unavailable = 0
    version_changed = 0
    for row in media_rows:
        media_id = str(row["id"])
        existing = _existing_record(repository, media_id, options)
        existing_status = str(existing.get("status") or "") if existing else None
        if existing_status == "success":
            already_success += 1
        if existing_status == "failed":
            failed += 1
        if existing_status == "engine_unavailable":
            engine_unavailable += 1
        needs_version_update = _needs_version_update(existing, options)
        if needs_version_update:
            version_changed += 1
        if _should_select(existing_status, needs_version_update, options):
            candidates.append(
                PlannedItem(
                    item_id=media_id,
                    item_type="media",
                    existing_status=existing_status,
                    needs_version_update=needs_version_update,
                    label=str(row.get("file_name") or media_id),
                )
            )
    if options.limit is not None:
        candidates = candidates[: max(options.limit, 0)]
    selected_count = len(candidates)
    return AnalysisPlan(
        options=options,
        total_candidates=len(media_rows),
        already_success=already_success,
        failed=failed,
        engine_unavailable=engine_unavailable,
        version_changed=version_changed,
        selected_items=candidates,
        estimated_storage_bytes=selected_count * STORAGE_BYTES_PER_ITEM.get(options.job_type, 1_024),
        estimated_processing_sec=round(selected_count * SECONDS_PER_ITEM.get(options.job_type, 1.0), 2),
        command_example=_command_example(options),
    )


def _plan_event_rebuild(repository: LifelogRepository, options: AnalysisPlanOptions) -> AnalysisPlan:
    days = _date_items(options.start_date, options.end_date)
    if not days and options.all_dates:
        date_values = {
            str(row.get("captured_at") or row.get("fallback_captured_at") or "")[:10]
            for row in repository.list_media_items(limit=1_000_000)
        }
        days = sorted(day for day in date_values if len(day) == 10)
    if options.limit is not None:
        days = days[: max(options.limit, 0)]
    items = [PlannedItem(item_id=day, item_type="date", existing_status=None, label=day) for day in days]
    return AnalysisPlan(
        options=options,
        total_candidates=len(days),
        selected_items=items,
        estimated_storage_bytes=len(items) * STORAGE_BYTES_PER_ITEM["event_rebuild"],
        estimated_processing_sec=round(len(items) * SECONDS_PER_ITEM["event_rebuild"], 2),
        command_example=_command_example(options),
    )


def _date_items(start_date: str | None, end_date: str | None) -> list[str]:
    if not start_date:
        return []
    end = end_date or start_date
    current = date.fromisoformat(start_date)
    last = date.fromisoformat(end)
    days: list[str] = []
    while current <= last:
        days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def _existing_record(
    repository: LifelogRepository,
    media_id: str,
    options: AnalysisPlanOptions,
) -> dict[str, Any] | None:
    if options.job_type == "ocr":
        return repository.get_media_ocr(media_id)
    if options.job_type == "vlm":
        return repository.get_media_vlm(media_id)
    if options.job_type in {"image_embedding", "text_embedding"}:
        embedding_type = "image" if options.job_type == "image_embedding" else options.embedding_type
        model_name = _embedding_model_name(options)
        return MediaEmbeddingRepository(repository.db_path).get_embedding(media_id, embedding_type, model_name)
    return None


def _embedding_model_name(options: AnalysisPlanOptions) -> str:
    return str(options.model_name or options.model_path or options.engine_name or "unknown")


def _needs_version_update(existing: dict[str, Any] | None, options: AnalysisPlanOptions) -> bool:
    if not existing:
        return False
    if options.job_type == "ocr":
        expected_version = options.analysis_version or OCR_ANALYSIS_VERSION
        return bool(
            (options.engine_name and str(existing.get("ocr_engine") or "") != options.engine_name)
            or str(existing.get("analysis_version") or "") != expected_version
        )
    if options.job_type == "vlm":
        expected_version = options.analysis_version or VLM_ANALYSIS_VERSION
        return bool(
            (options.engine_name and str(existing.get("vlm_engine") or "") != options.engine_name)
            or (options.model_name and str(existing.get("model_name") or "") != options.model_name)
            or (options.prompt_version and str(existing.get("prompt_version") or "") != options.prompt_version)
            or str(existing.get("analysis_version") or "") != expected_version
        )
    if options.job_type in {"image_embedding", "text_embedding"}:
        expected_model = _embedding_model_name(options)
        return str(existing.get("embedding_model") or "") != expected_model
    return False


def _should_select(
    existing_status: str | None,
    needs_version_update: bool,
    options: AnalysisPlanOptions,
) -> bool:
    if options.force:
        return True
    if options.failed_only:
        return existing_status == "failed"
    if options.engine_unavailable_only:
        return existing_status == "engine_unavailable"
    if options.version_changed_only:
        return needs_version_update
    if options.skip_existing:
        return existing_status != "success"
    return existing_status != "success"


def _command_example(options: AnalysisPlanOptions) -> str:
    parts = ["python -m personal_lifelog_rag.app.cli analysis-run", f"--type {options.job_type}"]
    if options.start_date and options.end_date and options.start_date != options.end_date:
        parts.extend([f"--from {options.start_date}", f"--to {options.end_date}"])
    elif options.start_date:
        parts.append(f"--date {options.start_date}")
    elif options.all_dates:
        parts.append("--all")
    if options.limit is not None:
        parts.append(f"--limit {options.limit}")
    if options.engine_name:
        parts.append(f"--engine {options.engine_name}")
    if options.force:
        parts.append("--force")
    if options.skip_existing:
        parts.append("--skip-existing")
    if options.failed_only:
        parts.append("--failed-only")
    if options.engine_unavailable_only:
        parts.append("--engine-unavailable-only")
    if options.version_changed_only:
        parts.append("--version-changed-only")
    return " ".join(parts)
