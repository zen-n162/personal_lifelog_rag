"""Analysis job runners that wrap existing local OCR/VLM/embedding services."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import uuid
from typing import Any

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.embeddings.embedding_service import build_image_embeddings, build_text_embeddings
from personal_lifelog_rag.embeddings.engines import get_multimodal_embedding_engine
from personal_lifelog_rag.embeddings.schemas import BuildMediaEmbeddingsOptions
from personal_lifelog_rag.jobs.job_repository import AnalysisJobRepository
from personal_lifelog_rag.jobs.planners import plan_analysis
from personal_lifelog_rag.jobs.schemas import AnalysisPlanOptions, AnalysisRunOptions
from personal_lifelog_rag.ocr.engines import get_ocr_engine
from personal_lifelog_rag.ocr.ocr_service import OcrImagesOptions, run_ocr_images
from personal_lifelog_rag.timeline.event_builder import EventBuildConfig, build_events
from personal_lifelog_rag.vlm.engines import get_vlm_engine
from personal_lifelog_rag.vlm.vlm_service import VlmImagesOptions, run_vlm_images


DEFAULT_ANALYSIS_JOB_OUTPUT_DIR = Path("eval_outputs/analysis_jobs")


def run_analysis_job(
    repository: LifelogRepository,
    options: AnalysisRunOptions,
    *,
    output_dir: Path = DEFAULT_ANALYSIS_JOB_OUTPUT_DIR,
    progress_callback=None,
) -> dict[str, Any]:
    """Run one analysis job. Dry-runs only return a plan and do not touch the DB."""

    plan = plan_analysis(repository, AnalysisPlanOptions(**_base_options(options)))
    if options.dry_run:
        return {
            "dry_run": True,
            "job_id": options.job_id,
            "job_type": options.job_type,
            "plan": plan.to_dict(),
            "report_paths": {},
        }

    job_repo = AnalysisJobRepository(repository.db_path)
    job_repo.initialize()
    job_id = options.job_id or _new_job_id(options.job_type)
    job_repo.create_job(
        job_id=job_id,
        job_type=options.job_type,
        status="planned",
        target_scope=options.to_scope(),
        engine=options.engine_name,
        model_name=options.model_name or options.model_path,
        prompt_version=options.prompt_version,
        analysis_version=options.analysis_version,
        total_items=len(plan.selected_items),
    )
    for item in plan.selected_items:
        job_repo.upsert_item(job_id=job_id, item_id=item.item_id, item_type=item.item_type, status="pending")
    job_repo.update_job_status(job_id, "running", mark_started=True)
    try:
        service_report = _run_service(repository, options, [item.item_id for item in plan.selected_items], progress_callback=progress_callback)
        _sync_items_from_report(job_repo, job_id, plan.selected_items, service_report)
        counts = job_repo.recalculate_counts(job_id)
        final_status = "completed" if counts["failed"] == 0 else ("failed" if counts["success"] == 0 else "partial")
        job_repo.update_job_status(job_id, final_status, mark_finished=True)
        job = job_repo.get_job(job_id)
        payload = {
            "dry_run": False,
            "job_id": job_id,
            "job_type": options.job_type,
            "status": final_status,
            "plan": plan.to_dict(include_items=False),
            "service_report": _to_dict(service_report),
            "job": job,
            "report_paths": {},
        }
        if options.save_report:
            payload["report_paths"] = write_analysis_job_report(payload, output_dir=output_dir)
        return payload
    except Exception as exc:
        job_repo.update_job_status(job_id, "failed", error_message=f"{exc.__class__.__name__}: {exc}", mark_finished=True)
        job_repo.recalculate_counts(job_id)
        raise


def resume_analysis_job(
    repository: LifelogRepository,
    job_id: str,
    *,
    failed_only: bool = False,
    engine_unavailable_only: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    save_report: bool = False,
) -> dict[str, Any]:
    job_repo = AnalysisJobRepository(repository.db_path)
    job = job_repo.get_job(job_id)
    if not job:
        raise ValueError(f"analysis job not found: {job_id}")
    scope = _scope_from_job(job)
    scope["limit"] = limit if limit is not None else scope.get("limit")
    scope["failed_only"] = failed_only or not engine_unavailable_only
    scope["engine_unavailable_only"] = engine_unavailable_only
    options = AnalysisRunOptions(**scope, dry_run=dry_run, save_report=save_report)
    return run_analysis_job(repository, options)


def retry_failed_analysis(
    repository: LifelogRepository,
    options: AnalysisRunOptions,
) -> dict[str, Any]:
    retry_options = AnalysisRunOptions(**{**options.to_scope(), "failed_only": True, "dry_run": options.dry_run, "save_report": options.save_report})
    return run_analysis_job(repository, retry_options)


def write_analysis_job_report(report: dict[str, Any], *, output_dir: Path = DEFAULT_ANALYSIS_JOB_OUTPUT_DIR) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(report.get("job_id") or "analysis_job")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{job_id}_{timestamp}.json"
    md_path = output_dir / f"{job_id}_{timestamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def _run_service(
    repository: LifelogRepository,
    options: AnalysisRunOptions,
    media_ids: list[str],
    *,
    progress_callback=None,
) -> Any:
    limit = len(media_ids) if media_ids else 0
    if options.job_type == "ocr":
        return run_ocr_images(
            repository,
            OcrImagesOptions(
                start_date=options.start_date,
                end_date=options.end_date,
                limit=limit,
                engine_name=options.engine_name,
                dry_run=False,
                force=options.force,
                skip_existing=options.skip_existing,
                media_ids=media_ids,
            ),
            engine=get_ocr_engine(options.engine_name),
            progress_callback=progress_callback,
        )
    if options.job_type == "vlm":
        return run_vlm_images(
            repository,
            VlmImagesOptions(
                start_date=options.start_date,
                end_date=options.end_date,
                limit=limit,
                engine_name=options.engine_name,
                model_name=options.model_path or options.model_name,
                dry_run=False,
                force=options.force,
                skip_existing=options.skip_existing,
                prompt_template=options.prompt_version,
                media_ids=media_ids,
            ),
            engine=get_vlm_engine(options.engine_name, model_name=options.model_name, model_path=options.model_path),
            progress_callback=progress_callback,
        )
    if options.job_type == "image_embedding":
        return build_image_embeddings(
            repository,
            BuildMediaEmbeddingsOptions(
                start_date=options.start_date,
                end_date=options.end_date,
                limit=limit,
                embedding_type="image",
                engine_name=options.engine_name,
                model_name=options.model_name,
                model_path=options.model_path,
                dry_run=False,
                force=options.force,
                skip_existing=options.skip_existing,
                media_ids=media_ids,
            ),
            engine=get_multimodal_embedding_engine(options.engine_name, model_name=options.model_name, model_path=options.model_path),
            progress_callback=progress_callback,
        )
    if options.job_type == "text_embedding":
        return build_text_embeddings(
            repository,
            BuildMediaEmbeddingsOptions(
                start_date=options.start_date,
                end_date=options.end_date,
                limit=limit,
                embedding_type=options.embedding_type,  # type: ignore[arg-type]
                engine_name=options.engine_name,
                model_name=options.model_name,
                model_path=options.model_path,
                dry_run=False,
                force=options.force,
                skip_existing=options.skip_existing,
                media_ids=media_ids,
            ),
            engine=get_multimodal_embedding_engine(options.engine_name, model_name=options.model_name, model_path=options.model_path),
            progress_callback=progress_callback,
        )
    if options.job_type == "event_rebuild":
        rows: list[dict[str, Any]] = []
        for day in media_ids:
            build_events(repository, start_date=day, config=EventBuildConfig(), force=options.force)
            rows.append({"media_id": day, "status": "success"})
        return {"rows": rows, "success": len(rows), "failed": 0, "skipped": 0, "processed": len(rows)}
    raise ValueError(f"unknown analysis job type: {options.job_type}")


def _sync_items_from_report(
    job_repo: AnalysisJobRepository,
    job_id: str,
    planned_items,
    service_report: Any,
) -> None:
    rows = _rows_from_report(service_report)
    seen: set[str] = set()
    for row in rows:
        item_id = str(row.get("media_id") or row.get("item_id") or "")
        if not item_id:
            continue
        seen.add(item_id)
        status = _normalize_item_status(str(row.get("status") or "skipped"))
        job_repo.upsert_item(
            job_id=job_id,
            item_id=item_id,
            item_type="date" if item_id[:4].isdigit() and len(item_id) == 10 else "media",
            status=status,
            error_message=row.get("error_message"),
            finished_at=datetime.now().replace(microsecond=0).isoformat(),
        )
    for item in planned_items:
        if item.item_id not in seen:
            job_repo.upsert_item(job_id=job_id, item_id=item.item_id, item_type=item.item_type, status="skipped")


def _rows_from_report(service_report: Any) -> list[dict[str, Any]]:
    if isinstance(service_report, dict):
        return list(service_report.get("rows") or [])
    return list(getattr(service_report, "rows", []) or [])


def _normalize_item_status(status: str) -> str:
    if status in {"success", "failed", "skipped", "engine_unavailable"}:
        return status
    if status in {"pending", "running"}:
        return status
    return "skipped"


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"value": str(value)}


def _base_options(options: AnalysisRunOptions) -> dict[str, Any]:
    return {
        key: value
        for key, value in options.to_scope().items()
        if key not in {"dry_run", "job_id", "save_report"}
    }


def _scope_from_job(job: dict[str, Any]) -> dict[str, Any]:
    try:
        scope = json.loads(str(job.get("target_scope_json") or "{}"))
    except json.JSONDecodeError:
        scope = {}
    scope.setdefault("job_type", job.get("job_type"))
    scope.setdefault("engine_name", job.get("engine"))
    scope.setdefault("model_name", job.get("model_name"))
    scope.setdefault("prompt_version", job.get("prompt_version"))
    scope.setdefault("analysis_version", job.get("analysis_version"))
    for key in ("dry_run", "job_id", "save_report"):
        scope.pop(key, None)
    return scope


def _new_job_id(job_type: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"analysis_{job_type}_{timestamp}_{uuid.uuid4().hex[:8]}"


def _markdown_report(report: dict[str, Any]) -> str:
    job = report.get("job") or {}
    service = report.get("service_report") or {}
    lines = [
        "# Analysis Job Report",
        "",
        "## Summary",
        f"- job_id: {report.get('job_id')}",
        f"- job_type: {report.get('job_type')}",
        f"- status: {report.get('status')}",
        f"- total_items: {job.get('total_items')}",
        f"- processed_items: {job.get('processed_items')}",
        f"- success_items: {job.get('success_items')}",
        f"- failed_items: {job.get('failed_items')}",
        f"- skipped_items: {job.get('skipped_items')}",
        "",
        "## Service Report",
        f"- processed: {service.get('processed')}",
        f"- success: {service.get('success')}",
        f"- failed: {service.get('failed')}",
        f"- skipped: {service.get('skipped')}",
        f"- engine_unavailable: {service.get('engine_unavailable')}",
        "",
        "## Next Recommended Command",
        f"`python -m personal_lifelog_rag.app.cli analysis-status --job-id {report.get('job_id')}`",
    ]
    return "\n".join(lines) + "\n"
