"""Collect privacy-safe metrics for generated reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from personal_lifelog_rag.db.checks import run_db_check
from personal_lifelog_rag.embeddings.embedding_service import embedding_stats
from personal_lifelog_rag.evaluation.private_eval import (
    evaluate_private_questions,
    load_private_eval_questions,
)
from personal_lifelog_rag.jobs.job_repository import AnalysisJobRepository
from personal_lifelog_rag.jobs.storage import storage_stats
from personal_lifelog_rag.line.call_index import call_stats
from personal_lifelog_rag.ocr.ocr_service import ocr_stats
from personal_lifelog_rag.places.stats import place_stats
from personal_lifelog_rag.reporting.examples import build_example_queries
from personal_lifelog_rag.reporting.redaction import ReportRedactor
from personal_lifelog_rag.reporting.schemas import ReportOptions
from personal_lifelog_rag.timeline.event_reports import event_stats
from personal_lifelog_rag.vlm.vlm_service import vlm_stats


def collect_report_data(repository, options: ReportOptions) -> dict[str, Any]:
    """Collect all report metrics without exposing raw personal records."""

    start_date = options.start_date
    end_date = options.end_date or options.start_date
    db_check = run_db_check(repository.db_path)
    redactor = ReportRedactor(public=options.public)
    data: dict[str, Any] = {
        "options": options.to_dict(),
        "db_summary": _dataset_summary(db_check),
        "db_check_summary": _db_check_summary(db_check),
        "event_stats": event_stats(repository, start_date=start_date, end_date=end_date),
        "ocr_stats": ocr_stats(repository, start_date=start_date, end_date=end_date),
        "vlm_stats": vlm_stats(repository, start_date=start_date, end_date=end_date),
        "embedding_stats": embedding_stats(repository, start_date=start_date, end_date=end_date),
        "call_stats": call_stats(repository, start_date=start_date, end_date=end_date),
        "place_stats": place_stats(repository, start_date=start_date, end_date=end_date),
        "analysis_jobs": _analysis_job_summary(repository.db_path),
        "storage_stats": storage_stats(repository.db_path),
        "private_eval": _private_eval_summary(repository, options),
    }
    data["examples"] = build_example_queries(data, redactor=redactor) if options.include_examples else []
    data["redaction"] = {
        "mode": options.mode,
        "gps": "hidden",
        "file_paths": "hidden",
        "line_text": "omitted or shortened",
        "media_ids": "redacted" if options.public else "shortened",
    }
    return _public_sanitize(data, redactor=redactor) if options.public else _private_sanitize(data, redactor=redactor)


def _dataset_summary(db_check: dict[str, Any]) -> dict[str, Any]:
    media = db_check["media_items"]
    line = db_check["line_messages"]
    events = db_check["events"]
    evidence = db_check["event_evidence"]
    ocr = db_check["media_ocr"]
    vlm = db_check["media_vlm"]
    embeddings = db_check["media_embeddings"]
    calls = db_check["line_call_events"]
    return {
        "media_items": media["total"],
        "gps_photos": media["gps_present"],
        "line_messages": line["total"],
        "events": events["total"],
        "event_evidence": evidence["total"],
        "ocr_analyzed": ocr["total"],
        "vlm_analyzed": vlm["total"],
        "embeddings": embeddings["total"],
        "call_events": calls["total"],
    }


def _db_check_summary(db_check: dict[str, Any]) -> dict[str, Any]:
    return {
        "strict_ok": bool(db_check.get("strict", {}).get("ok")),
        "strict_issue_count": len(db_check.get("strict", {}).get("issues") or []),
        "missing_files": db_check.get("media_items", {}).get("missing_file_count", 0),
        "orphan_event_evidence": db_check.get("event_evidence", {}).get("orphan_event_refs", 0),
    }


def _analysis_job_summary(db_path: str | Path) -> dict[str, Any]:
    repository = AnalysisJobRepository(db_path)
    jobs = repository.list_jobs(recent=20)
    status_counts: dict[str, int] = {}
    for job in jobs:
        status = str(job.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "recent_count": len(jobs),
        "status_counts": dict(sorted(status_counts.items())),
        "recent_jobs": [
            {
                "job_id": job.get("job_id"),
                "job_type": job.get("job_type"),
                "status": job.get("status"),
                "total_items": job.get("total_items"),
                "success_items": job.get("success_items"),
                "failed_items": job.get("failed_items"),
            }
            for job in jobs[:10]
        ],
    }


def _private_eval_summary(repository, options: ReportOptions) -> dict[str, Any] | None:
    if options.eval_run:
        return _load_eval_run(options.eval_run)
    if options.eval_path:
        questions = load_private_eval_questions(options.eval_path)
        report = evaluate_private_questions(repository, questions)
        return _compact_eval(report)
    return None


def _load_eval_run(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return _compact_eval(payload)


def _compact_eval(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": report.get("run_id"),
        "created_at": report.get("created_at"),
        "summary": report.get("summary", {}),
        "by_type": report.get("by_type", {}),
        "ranking_metrics": report.get("ranking_metrics", {}),
        "safety_metrics": report.get("safety_metrics", {}),
    }


def _public_sanitize(data: dict[str, Any], *, redactor: ReportRedactor) -> dict[str, Any]:
    event_report = data.get("event_stats") or {}
    event_report["title_counts"] = _public_title_counts(event_report.get("title_counts") or {})
    event_report["low_confidence_events"] = [
        {
            "date": redactor.date(row.get("date")),
            "title": _public_event_title(row.get("title")),
            "confidence": row.get("confidence"),
        }
        for row in event_report.get("low_confidence_events", [])[:10]
    ]
    place_report = data.get("place_stats") or {}
    place_report["top_locations"] = [
        {
            "location_name": redactor.place(row.get("location_name")),
            "event_count": row.get("event_count"),
        }
        for row in place_report.get("top_locations", [])[:20]
    ]
    place_report["location_counts"] = {
        redactor.place(name): count
        for name, count in (place_report.get("location_counts") or {}).items()
    }
    data["storage_stats"].pop("db_path", None)
    return data


def _private_sanitize(data: dict[str, Any], *, redactor: ReportRedactor) -> dict[str, Any]:
    # Private reports still avoid raw paths, exact GPS, and full LINE text.
    data["storage_stats"].pop("db_path", None)
    return data


def _public_title_counts(raw_counts: dict[str, Any]) -> dict[str, int]:
    output: dict[str, int] = {}
    for title, count in raw_counts.items():
        safe_title = _public_event_title(title)
        output[safe_title] = output.get(safe_title, 0) + int(count or 0)
    return dict(sorted(output.items(), key=lambda item: (-item[1], item[0])))


def _public_event_title(value: Any) -> str:
    title = str(value or "").strip()
    safe_keywords = (
        "LINE",
        "通話",
        "連絡",
        "写真",
        "記録",
        "食事",
        "カフェ",
        "移動",
        "待ち合わせ",
        "位置情報",
        "未分類",
    )
    if any(keyword in title for keyword in safe_keywords):
        return title[:80]
    return "EVENT_TITLE_REDACTED"
