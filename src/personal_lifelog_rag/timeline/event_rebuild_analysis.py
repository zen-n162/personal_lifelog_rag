"""Rebuild events after OCR/VLM analysis and compare before/after quality."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from personal_lifelog_rag.core.config import load_event_building_config
from personal_lifelog_rag.core.privacy import redact_text
from personal_lifelog_rag.db.backup import DEFAULT_BACKUP_DIR, backup_sqlite_db
from personal_lifelog_rag.db.checks import run_db_check
from personal_lifelog_rag.evaluation.private_eval import (
    evaluate_private_questions,
    load_private_eval_questions,
)
from personal_lifelog_rag.retrieval.answer_builder import build_answer
from personal_lifelog_rag.retrieval.date_parser import parse_date_query
from personal_lifelog_rag.retrieval.local_search import LocalSearchOptions, local_text_search
from personal_lifelog_rag.retrieval.query_router import route_query
from personal_lifelog_rag.retrieval.temporal_search import search_timeline
from personal_lifelog_rag.timeline.event_builder import EventBuildConfig, build_events
from personal_lifelog_rag.vlm.schemas import ImageSearchOptions
from personal_lifelog_rag.vlm.vlm_service import image_search


DEFAULT_EVENT_REBUILD_OUTPUT_DIR = Path("eval_outputs/event_rebuild")
DEFAULT_REBUILD_SEARCH_QUERIES = ("ご飯", "新宿")
DEFAULT_REBUILD_IMAGE_QUERIES = ("ご飯", "新宿")
DEFAULT_REBUILD_QA_QUERIES = ("ご飯を食べた写真はいつ？", "新宿に行ったのはいつ？")


@dataclass(frozen=True)
class EventRebuildOptions:
    date: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    dry_run: bool = False
    save_report: bool = False
    force: bool = False
    eval_path: Path | None = None
    output_dir: Path = DEFAULT_EVENT_REBUILD_OUTPUT_DIR
    backup_dir: Path = DEFAULT_BACKUP_DIR


def rebuild_events_with_analysis(
    repository,
    db_path: str | Path,
    options: EventRebuildOptions,
    *,
    config: EventBuildConfig | None = None,
    now: datetime | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    created_at = now or datetime.now()
    start_date, end_date = _range_from_options(options)
    before = analysis_snapshot(repository, start_date=start_date, end_date=end_date)
    before_eval = _private_eval(repository, options.eval_path)
    backup_path = None
    backup_size = None
    if not options.dry_run:
        backup = backup_sqlite_db(
            db_path,
            label=f"before_event_rebuild_{start_date.replace('-', '')}",
            output_dir=options.backup_dir,
            now=created_at,
        )
        backup_path = str(backup.backup_path)
        backup_size = backup.size_bytes

    build_config = config or EventBuildConfig.from_mapping(load_event_building_config())
    build_report = build_events(
        repository,
        start_date=start_date,
        end_date=end_date,
        config=build_config,
        dry_run=options.dry_run,
        force=options.force,
        progress_callback=progress_callback,
    )
    after = analysis_snapshot(repository, start_date=start_date, end_date=end_date)
    after_eval = before_eval if options.dry_run else _private_eval(repository, options.eval_path)
    diff = diff_event_snapshots(before, after)
    db_check = run_db_check(db_path)
    report = {
        "run_info": {
            "date": options.date,
            "from": start_date,
            "to": end_date,
            "created_at": created_at.isoformat(timespec="seconds"),
            "dry_run": options.dry_run,
            "force": options.force,
            "eval_path": str(options.eval_path) if options.eval_path else None,
        },
        "db_safety": {
            "backup_path": backup_path,
            "backup_size_bytes": backup_size,
            "strict_ok": bool(db_check.get("strict", {}).get("ok")),
            "strict_issues": list(db_check.get("strict", {}).get("issues") or []),
        },
        "build_report": _build_report_dict(build_report),
        "before_snapshot": before,
        "after_snapshot": after,
        "event_diff": diff,
        "private_eval": {
            "before": _eval_summary(before_eval),
            "after": _eval_summary(after_eval),
            "delta": _eval_delta(before_eval, after_eval),
        },
        "recommendation": _recommendation(diff, db_check),
    }
    if options.save_report:
        report["output_paths"] = write_event_rebuild_report(report, output_dir=options.output_dir, now=created_at)
    return report


def analysis_snapshot(
    repository,
    *,
    start_date: str,
    end_date: str | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    resolved_end = end_date or start_date
    events = repository.list_events(start_date=start_date, end_date=resolved_end, include_hidden=True, limit=100_000)
    event_rows = [_snapshot_event(repository, event) for event in events]
    return {
        "created_at": (created_at or datetime.now()).isoformat(timespec="seconds"),
        "date": start_date if start_date == resolved_end else None,
        "from": start_date,
        "to": resolved_end,
        "summary": _snapshot_summary(event_rows),
        "events": event_rows,
        "search_samples": _search_samples(repository, start_date=start_date, end_date=resolved_end),
        "qa_samples": _qa_samples(repository),
        "ask_sample": _ask_sample(repository, start_date=start_date, end_date=resolved_end),
    }


def diff_event_snapshots(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_events = {str(event["event_id"]): event for event in before.get("events", [])}
    after_events = {str(event["event_id"]): event for event in after.get("events", [])}
    added = [event for event_id, event in after_events.items() if event_id not in before_events]
    removed = [event for event_id, event in before_events.items() if event_id not in after_events]
    changed_titles = []
    changed_summaries = []
    changed_confidence = []
    changed_locations = []
    changed_evidence = []
    for event_id in sorted(set(before_events) & set(after_events)):
        before_event = before_events[event_id]
        after_event = after_events[event_id]
        if before_event.get("title") != after_event.get("title"):
            changed_titles.append(_change_row(event_id, before_event, after_event, "title"))
        if before_event.get("summary_short") != after_event.get("summary_short"):
            changed_summaries.append(_change_row(event_id, before_event, after_event, "summary_short"))
        if before_event.get("confidence") != after_event.get("confidence"):
            changed_confidence.append(_change_row(event_id, before_event, after_event, "confidence"))
        if before_event.get("location_name") != after_event.get("location_name"):
            changed_locations.append(_change_row(event_id, before_event, after_event, "location_name"))
        if before_event.get("evidence_counts") != after_event.get("evidence_counts"):
            changed_evidence.append(_change_row(event_id, before_event, after_event, "evidence_counts"))
    before_summary = before.get("summary", {})
    after_summary = after.get("summary", {})
    return {
        "event_count_delta": int(after_summary.get("event_count", 0)) - int(before_summary.get("event_count", 0)),
        "event_evidence_count_delta": int(after_summary.get("event_evidence_count", 0)) - int(before_summary.get("event_evidence_count", 0)),
        "ocr_evidence_delta": int(after_summary.get("ocr_evidence_count", 0)) - int(before_summary.get("ocr_evidence_count", 0)),
        "vlm_evidence_delta": int(after_summary.get("vlm_evidence_count", 0)) - int(before_summary.get("vlm_evidence_count", 0)),
        "vlm_only_high_confidence_count": _vlm_only_high_confidence(after.get("events", [])),
        "added_events": [_compact_event(event) for event in added],
        "removed_events": [_compact_event(event) for event in removed],
        "changed_titles": changed_titles,
        "changed_summaries": changed_summaries,
        "changed_confidence": changed_confidence,
        "changed_locations": changed_locations,
        "changed_evidence": changed_evidence,
        "title_distribution_before": before_summary.get("title_counts", {}),
        "title_distribution_after": after_summary.get("title_counts", {}),
        "low_confidence_delta": int(after_summary.get("low_confidence_events", 0)) - int(before_summary.get("low_confidence_events", 0)),
        "override_status": _override_status(before.get("events", []), after.get("events", [])),
    }


def write_event_rebuild_report(
    report: dict[str, Any],
    *,
    output_dir: str | Path = DEFAULT_EVENT_REBUILD_OUTPUT_DIR,
    now: datetime | None = None,
) -> dict[str, str]:
    created = now or datetime.now()
    run = report.get("run_info", {})
    date_label = str(run.get("date") or f"{run.get('from')}_{run.get('to')}").replace("-", "")
    timestamp = created.strftime("%Y%m%d_%H%M%S")
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / f"event_rebuild_{date_label}_{timestamp}.json"
    markdown_path = destination / f"event_rebuild_{date_label}_{timestamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    markdown_path.write_text(format_event_rebuild_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def format_event_rebuild_report(report: dict[str, Any]) -> str:
    run = report["run_info"]
    diff = report["event_diff"]
    paths = report.get("output_paths") or {}
    lines = [
        "Event Rebuild Analysis",
        f"- range: {run['from']}..{run['to']}",
        f"- dry_run: {run['dry_run']}",
        f"- force: {run['force']}",
        f"- backup: {report['db_safety'].get('backup_path') or ''}",
        f"- db_check_strict_ok: {report['db_safety'].get('strict_ok')}",
        f"- event_count_delta: {diff['event_count_delta']:+d}",
        f"- evidence_count_delta: {diff['event_evidence_count_delta']:+d}",
        f"- OCR evidence delta: {diff['ocr_evidence_delta']:+d}",
        f"- VLM evidence delta: {diff['vlm_evidence_delta']:+d}",
        f"- VLM-only high confidence events: {diff['vlm_only_high_confidence_count']}",
        f"- changed titles: {len(diff['changed_titles'])}",
        f"- changed confidence: {len(diff['changed_confidence'])}",
    ]
    if paths:
        lines.extend(["Reports:", f"- json: {paths.get('json')}", f"- markdown: {paths.get('markdown')}"])
    return "\n".join(lines)


def format_event_diff(diff: dict[str, Any]) -> str:
    lines = [
        "Event Diff",
        f"- event_count_delta: {diff.get('event_count_delta', 0):+d}",
        f"- event_evidence_count_delta: {diff.get('event_evidence_count_delta', 0):+d}",
        f"- OCR evidence delta: {diff.get('ocr_evidence_delta', 0):+d}",
        f"- VLM evidence delta: {diff.get('vlm_evidence_delta', 0):+d}",
        f"- VLM-only high confidence events: {diff.get('vlm_only_high_confidence_count', 0)}",
        "",
        "changed titles:",
    ]
    rows = diff.get("changed_titles") or []
    if rows:
        for row in rows[:20]:
            lines.append(f"- {row['event_id']}: {row['before']} -> {row['after']}")
    else:
        lines.append("- none")
    return "\n".join(lines)


def format_event_rebuild_markdown(report: dict[str, Any]) -> str:
    run = report["run_info"]
    before = report["before_snapshot"]["summary"]
    after = report["after_snapshot"]["summary"]
    diff = report["event_diff"]
    private_eval = report.get("private_eval") or {}
    lines = [
        "# Event Rebuild Analysis Report",
        "",
        "## Run Info",
        f"- range: {run['from']}..{run['to']}",
        f"- created_at: {run['created_at']}",
        f"- backup path: {report['db_safety'].get('backup_path') or ''}",
        f"- dry_run: {run['dry_run']}",
        f"- force: {run['force']}",
        "",
        "## Before/After Summary",
        f"- event count: {before.get('event_count', 0)} -> {after.get('event_count', 0)}",
        f"- evidence count: {before.get('event_evidence_count', 0)} -> {after.get('event_evidence_count', 0)}",
        f"- confidence avg: {_fmt(before.get('confidence_avg'))} -> {_fmt(after.get('confidence_avg'))}",
        f"- OCR evidence count: {before.get('ocr_evidence_count', 0)} -> {after.get('ocr_evidence_count', 0)}",
        f"- VLM evidence count: {before.get('vlm_evidence_count', 0)} -> {after.get('vlm_evidence_count', 0)}",
        f"- VLM-only high confidence count: {diff.get('vlm_only_high_confidence_count', 0)}",
        "",
        "## Event Changes",
        f"- added events: {len(diff.get('added_events') or [])}",
        f"- removed events: {len(diff.get('removed_events') or [])}",
        f"- changed titles: {len(diff.get('changed_titles') or [])}",
        f"- changed summaries: {len(diff.get('changed_summaries') or [])}",
        f"- changed confidence: {len(diff.get('changed_confidence') or [])}",
        "",
        "## QA/Search Changes",
        f"- search samples before: {len(report['before_snapshot'].get('search_samples') or {})}",
        f"- search samples after: {len(report['after_snapshot'].get('search_samples') or {})}",
        f"- qa samples before: {len(report['before_snapshot'].get('qa_samples') or {})}",
        f"- qa samples after: {len(report['after_snapshot'].get('qa_samples') or {})}",
        "",
        "## Private Eval",
        f"- before passed/failed/skipped: {_eval_line(private_eval.get('before'))}",
        f"- after passed/failed/skipped: {_eval_line(private_eval.get('after'))}",
        f"- delta: {private_eval.get('delta') or {}}",
        "",
        "## Safety",
        f"- overclaim violations: {_overclaim_violations(report)}",
        f"- VLM-only strong claims: {diff.get('vlm_only_high_confidence_count', 0)}",
        "",
        "## Recommendation",
        f"- {report.get('recommendation')}",
    ]
    return "\n".join(lines) + "\n"


def load_snapshot_or_report(path: str | Path, *, slot: str | None = None) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if slot and payload.get(slot):
        return payload[slot]
    if payload.get("events") is not None:
        return payload
    if payload.get("before_snapshot") and slot != "after_snapshot":
        return payload["before_snapshot"]
    if payload.get("after_snapshot"):
        return payload["after_snapshot"]
    return payload


def _snapshot_event(repository, event: dict[str, Any]) -> dict[str, Any]:
    evidence = repository.list_event_evidence(str(event["id"]))
    counts = Counter(str(row.get("evidence_type") or "unknown") for row in evidence)
    has_line = counts.get("line", 0) > 0
    has_photo = counts.get("photo", 0) > 0
    has_ocr = counts.get("ocr", 0) > 0
    has_vlm = counts.get("vlm", 0) > 0
    return {
        "event_id": event.get("id"),
        "date": event.get("date"),
        "start_time": event.get("start_time"),
        "end_time": event.get("end_time"),
        "title": redact_text(event.get("title"), max_chars=120),
        "summary_short": redact_text(event.get("summary"), max_chars=220),
        "location_name": redact_text(event.get("location_name"), max_chars=80),
        "confidence": event.get("confidence"),
        "evidence_counts": dict(sorted(counts.items())),
        "event_evidence_count": len(evidence),
        "evidence_strength": {
            "vlm_only": has_vlm and not has_line and not has_ocr,
            "has_line": has_line,
            "has_photo": has_photo,
            "has_ocr": has_ocr,
            "has_vlm": has_vlm,
        },
        "is_hidden": int(event.get("is_hidden") or 0),
        "is_pinned": int(event.get("is_pinned") or 0),
        "is_verified": int(event.get("is_verified") or 0),
        "title_override": bool(event.get("title_override")),
        "summary_override": bool(event.get("summary_override")),
        "location_name_override": bool(event.get("location_name_override")),
    }


def _snapshot_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    confidence_values = [float(event["confidence"]) for event in events if event.get("confidence") is not None]
    evidence_counts = Counter()
    for event in events:
        evidence_counts.update(event.get("evidence_counts") or {})
    return {
        "event_count": len(events),
        "event_evidence_count": sum(int(event.get("event_evidence_count") or 0) for event in events),
        "title_counts": dict(sorted(Counter(str(event.get("title") or "") for event in events).items())),
        "confidence_avg": round(sum(confidence_values) / len(confidence_values), 3) if confidence_values else None,
        "evidence_type_counts": dict(sorted(evidence_counts.items())),
        "ocr_evidence_count": int(evidence_counts.get("ocr", 0)),
        "vlm_evidence_count": int(evidence_counts.get("vlm", 0)),
        "low_confidence_events": sum(1 for event in events if float(event.get("confidence") or 0.0) < 0.5),
    }


def _search_samples(repository, *, start_date: str, end_date: str) -> dict[str, Any]:
    samples: dict[str, Any] = {}
    for query in DEFAULT_REBUILD_SEARCH_QUERIES:
        report = local_text_search(
            repository,
            LocalSearchOptions(query=query, date_from=start_date, date_to=end_date, limit=5, mode="all"),
        )
        samples[f"search {query}"] = _compact_search_report(report)
    for query in DEFAULT_REBUILD_IMAGE_QUERIES:
        report = image_search(repository, ImageSearchOptions(query=query, date_from=start_date, date_to=end_date, limit=5))
        samples[f"image-search {query}"] = {
            "total": report.get("total", 0),
            "results": [_compact_image_result(row) for row in report.get("results", [])[:5]],
        }
    return samples


def _qa_samples(repository) -> dict[str, Any]:
    samples: dict[str, Any] = {}
    for query in DEFAULT_REBUILD_QA_QUERIES:
        result = route_query(repository, query, limit=5).to_dict()
        samples[query] = {
            "intent": result.get("intent"),
            "routing": result.get("routing"),
            "results_count": len(result.get("results") or []),
            "answer_preview": redact_text(result.get("answer"), max_chars=500),
        }
    return samples


def _ask_sample(repository, *, start_date: str, end_date: str) -> dict[str, Any]:
    question = f"{start_date}は何していた？" if start_date == end_date else f"{start_date}から{end_date}は何していた？"
    date_range = parse_date_query(question)
    result = search_timeline(repository, question, date_range=date_range)
    return {"question": question, "answer_preview": redact_text(build_answer(question, result), max_chars=700)}


def _compact_search_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_dates": len(report.get("results") or []),
        "results": [
            {
                "date": row.get("date"),
                "classification": row.get("classification"),
                "confidence": row.get("confidence"),
                "score": row.get("score"),
                "evidence_types": row.get("evidence_types") or [],
            }
            for row in (report.get("results") or [])[:5]
        ],
    }


def _compact_image_result(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": row.get("date"),
        "media_id": row.get("media_id"),
        "confidence": row.get("confidence"),
        "score": row.get("score"),
        "evidence_types": row.get("evidence_types") or [],
        "matched_fields": row.get("matched_fields") or [],
        "caption": redact_text(row.get("caption"), max_chars=100),
    }


def _compact_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event.get("event_id"),
        "date": event.get("date"),
        "start_time": event.get("start_time"),
        "end_time": event.get("end_time"),
        "title": event.get("title"),
        "confidence": event.get("confidence"),
        "evidence_counts": event.get("evidence_counts"),
    }


def _change_row(event_id: str, before: dict[str, Any], after: dict[str, Any], key: str) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "before": before.get(key),
        "after": after.get(key),
    }


def _vlm_only_high_confidence(events: list[dict[str, Any]]) -> int:
    return sum(
        1
        for event in events
        if (event.get("evidence_strength") or {}).get("vlm_only")
        and float(event.get("confidence") or 0.0) >= 0.8
    )


def _override_status(before_events: list[dict[str, Any]], after_events: list[dict[str, Any]]) -> dict[str, Any]:
    before_by_id = {str(event.get("event_id")): event for event in before_events}
    after_by_id = {str(event.get("event_id")): event for event in after_events}
    checks = {"verified_preserved": 0, "pinned_preserved": 0, "hidden_preserved": 0, "lost_overrides": []}
    for event_id, before in before_by_id.items():
        after = after_by_id.get(event_id)
        if after is None:
            continue
        for key, counter in (("is_verified", "verified_preserved"), ("is_pinned", "pinned_preserved"), ("is_hidden", "hidden_preserved")):
            if before.get(key) and after.get(key):
                checks[counter] += 1
            elif before.get(key) and not after.get(key):
                checks["lost_overrides"].append({"event_id": event_id, "field": key})
    return checks


def _private_eval(repository, eval_path: Path | None) -> dict[str, Any] | None:
    if eval_path is None or not Path(eval_path).expanduser().exists():
        return None
    questions = load_private_eval_questions(eval_path)
    return evaluate_private_questions(repository, questions)


def _eval_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "summary": report.get("summary"),
        "ranking_metrics": report.get("ranking_metrics"),
        "safety_metrics": report.get("safety_metrics"),
    }


def _eval_delta(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    if before is None or after is None:
        return {}
    before_summary = before.get("summary") or {}
    after_summary = after.get("summary") or {}
    return {
        "passed": int(after_summary.get("passed") or 0) - int(before_summary.get("passed") or 0),
        "failed": int(after_summary.get("failed") or 0) - int(before_summary.get("failed") or 0),
        "skipped": int(after_summary.get("skipped") or 0) - int(before_summary.get("skipped") or 0),
    }


def _build_report_dict(report: Any) -> dict[str, Any]:
    return {
        "start_date": report.start_date,
        "end_date": report.end_date,
        "days_scanned": report.days_scanned,
        "days_skipped": report.days_skipped,
        "dry_run": report.dry_run,
        "events_planned": report.events_planned,
        "evidence_planned": report.evidence_planned,
        "events_created": report.events_created,
        "evidence_saved": report.evidence_saved,
        "events_deleted": report.events_deleted,
        "day_reports": report.day_reports,
    }


def _recommendation(diff: dict[str, Any], db_check: dict[str, Any]) -> str:
    if not db_check.get("strict", {}).get("ok"):
        return "db-check strict failed; inspect integrity before trusting rebuilt events"
    if diff.get("vlm_only_high_confidence_count"):
        return "reduce VLM trust before wider rebuild; VLM-only high confidence events were found"
    if diff.get("ocr_evidence_delta", 0) or diff.get("vlm_evidence_delta", 0):
        return "review OCR/VLM-assisted event changes in UI before expanding to a wider period"
    return "no major OCR/VLM event evidence change; inspect search and QA snapshots before further tuning"


def _range_from_options(options: EventRebuildOptions) -> tuple[str, str]:
    if options.date:
        return options.date, options.date
    if options.start_date:
        return options.start_date, options.end_date or options.start_date
    raise ValueError("rebuild-events-with-analysis requires --date or --from")


def _fmt(value: Any) -> str:
    if value is None:
        return "none"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _eval_line(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "not run"
    row = summary.get("summary") or {}
    return f"{row.get('passed', 0)}/{row.get('failed', 0)}/{row.get('skipped', 0)}"


def _overclaim_violations(report: dict[str, Any]) -> int:
    text = json.dumps(report.get("after_snapshot", {}).get("events", []), ensure_ascii=False)
    return sum(1 for phrase in ("確実に食事した", "確実に新宿に行った", "確実に行った") if phrase in text)
