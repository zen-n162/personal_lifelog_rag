"""Small, safe local VLM pilot orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Literal

from personal_lifelog_rag.core.privacy import redact_text
from personal_lifelog_rag.db.backup import DEFAULT_BACKUP_DIR, backup_sqlite_db
from personal_lifelog_rag.db.checks import run_db_check
from personal_lifelog_rag.retrieval.query_router import route_query
from personal_lifelog_rag.vlm.engines import get_vlm_engine
from personal_lifelog_rag.vlm.pilot_report import DEFAULT_VLM_PILOT_OUTPUT_DIR, write_vlm_pilot_report
from personal_lifelog_rag.vlm.safety import FORBIDDEN_TERMS
from personal_lifelog_rag.vlm.schemas import ImageSearchOptions
from personal_lifelog_rag.vlm.vlm_service import (
    VlmImagesOptions,
    image_search,
    run_vlm_images,
    vlm_stats,
)


PilotStrategy = Literal["time_spread", "event_evidence", "ocr_first", "gps_first"]
VALID_STRATEGIES: set[str] = {"time_spread", "event_evidence", "ocr_first", "gps_first"}
DEFAULT_PILOT_IMAGE_QUERIES = ["ご飯", "カフェ", "新宿"]
DEFAULT_PILOT_QA_QUERY = "ご飯を食べた写真はいつ？"


@dataclass(frozen=True)
class VlmPilotOptions:
    date: str
    limit: int = 20
    engine_name: str | None = None
    model_name: str | None = None
    prompt_template: str | None = "lifelog_structured_tags_v1"
    dry_run: bool = False
    save_report: bool = False
    force: bool = False
    skip_existing: bool = False
    include_hidden: bool = False
    strategy: PilotStrategy = "time_spread"
    output_dir: Path = DEFAULT_VLM_PILOT_OUTPUT_DIR
    backup_dir: Path = DEFAULT_BACKUP_DIR


def run_vlm_pilot(
    repository,
    db_path: str | Path,
    options: VlmPilotOptions,
    *,
    engine=None,
    now: datetime | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    """Run a local-only, limited VLM pilot and return a compact report."""

    created_at = now or datetime.now()
    if options.strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unknown VLM pilot strategy: {options.strategy}")

    selected = select_vlm_pilot_images(
        repository,
        date=options.date,
        limit=options.limit,
        strategy=options.strategy,
        include_hidden=options.include_hidden,
    )
    resolved_engine = engine or get_vlm_engine(options.engine_name, model_name=options.model_name)
    before_stats = vlm_stats(repository, start_date=options.date, end_date=options.date)
    db_safety: dict[str, Any] = {
        "backup_path": None,
        "backup_size_bytes": None,
        "strict_ok": None,
        "strict_issues": [],
    }

    vlm_report_payload = _empty_vlm_report_payload(
        selected_count=len(selected),
        engine_name=resolved_engine.name,
        model_name=getattr(resolved_engine, "model_name", None),
        dry_run=options.dry_run,
    )
    if not options.dry_run:
        backup = backup_sqlite_db(
            db_path,
            label=f"before_vlm_pilot_{options.date.replace('-', '')}",
            output_dir=options.backup_dir,
            now=created_at,
        )
        db_safety["backup_path"] = str(backup.backup_path)
        db_safety["backup_size_bytes"] = backup.size_bytes

    db_check = run_db_check(db_path)
    db_safety["strict_ok"] = bool(db_check.get("strict", {}).get("ok"))
    db_safety["strict_issues"] = list(db_check.get("strict", {}).get("issues") or [])
    if not db_safety["strict_ok"]:
        safety_summary = _safety_summary([])
        vlm_report_payload = _add_pilot_summary(vlm_report_payload, safety_summary)
        report = _base_report(
            options=options,
            created_at=created_at,
            engine_name=resolved_engine.name,
            model_name=getattr(resolved_engine, "model_name", None),
            selected=selected,
            before_stats=before_stats,
            after_stats=before_stats,
            db_safety=db_safety,
            vlm_report=vlm_report_payload,
            processed_images=[],
            safety_summary=safety_summary,
            search_smoke_tests={},
            recommendation="inspect_failures",
        )
        report["blocked_reason"] = "db-check strict failed; resolve DB integrity issues before running VLM pilot"
        if options.save_report:
            report["output_paths"] = write_vlm_pilot_report(report, output_dir=options.output_dir, now=created_at)
        return report

    if not options.dry_run:

        media_ids = [str(row["media_id"]) for row in selected]
        vlm_report = run_vlm_images(
            repository,
            VlmImagesOptions(
                start_date=options.date,
                end_date=options.date,
                limit=len(media_ids),
                engine_name=options.engine_name,
                model_name=options.model_name,
                dry_run=False,
                force=options.force,
                skip_existing=options.skip_existing,
                prompt_template=options.prompt_template,
                media_ids=media_ids,
            ),
            engine=resolved_engine,
            progress_callback=progress_callback,
        )
        vlm_report_payload = {
            "selected_images": vlm_report.selected_images,
            "processed": vlm_report.processed,
            "success": vlm_report.success,
            "failed": vlm_report.failed,
            "skipped": vlm_report.skipped,
            "no_visual_content": vlm_report.no_visual_content,
            "engine_unavailable": vlm_report.engine_unavailable,
            "dry_run": vlm_report.dry_run,
            "engine": vlm_report.engine,
            "model_name": vlm_report.model_name,
            "rows": vlm_report.rows,
        }

    after_stats = vlm_stats(repository, start_date=options.date, end_date=options.date)
    processed_images = _processed_images(repository, selected)
    safety_summary = _safety_summary(processed_images)
    vlm_report_payload = _add_pilot_summary(vlm_report_payload, safety_summary)
    search_smoke_tests = {} if options.dry_run else _search_smoke_tests(repository, options.date)
    recommendation = _recommendation(vlm_report_payload, safety_summary)
    vlm_report_payload["recommendation"] = recommendation
    report = _base_report(
        options=options,
        created_at=created_at,
        engine_name=resolved_engine.name,
        model_name=getattr(resolved_engine, "model_name", None),
        selected=selected,
        before_stats=before_stats,
        after_stats=after_stats,
        db_safety=db_safety,
        vlm_report=vlm_report_payload,
        processed_images=processed_images,
        safety_summary=safety_summary,
        search_smoke_tests=search_smoke_tests,
        recommendation=recommendation,
    )
    if options.save_report:
        report["output_paths"] = write_vlm_pilot_report(report, output_dir=options.output_dir, now=created_at)
    return report


def select_vlm_pilot_images(
    repository,
    *,
    date: str,
    limit: int,
    strategy: PilotStrategy = "time_spread",
    include_hidden: bool = False,
) -> list[dict[str, Any]]:
    """Select a small, diversified set of existing local images for pilot analysis."""

    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unknown VLM pilot strategy: {strategy}")
    rows = repository.list_media_items(start_date=date, end_date=date, limit=1_000_000)
    metadata = _pilot_metadata(repository, date=date)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        media_id = str(row.get("id") or "")
        if not media_id:
            continue
        path = Path(str(row.get("file_path") or "")).expanduser()
        if not path.exists() or not path.is_file():
            continue
        hidden_statuses = metadata["event_hidden_by_media_id"].get(media_id, [])
        if hidden_statuses and all(hidden_statuses) and not include_hidden:
            continue
        timestamp = str(row.get("captured_at") or row.get("fallback_captured_at") or "")
        candidates.append(
            {
                "media_id": media_id,
                "file_name": row.get("file_name"),
                "file_path": str(path),
                "captured_at": timestamp,
                "thumbnail_path": row.get("thumbnail_path") or "",
                "has_event_evidence": media_id in metadata["event_evidence_media_ids"],
                "has_ocr": media_id in metadata["ocr_media_ids"],
                "has_gps": row.get("gps_lat") is not None and row.get("gps_lon") is not None,
                "is_hidden_event_only": bool(hidden_statuses and all(hidden_statuses)),
            }
        )

    if strategy == "time_spread":
        selected = _time_spread(candidates, max(limit, 0))
        return _ensure_priority_coverage(selected, candidates, max(limit, 0))

    key_name = {
        "event_evidence": "has_event_evidence",
        "ocr_first": "has_ocr",
        "gps_first": "has_gps",
    }[strategy]
    candidates.sort(key=lambda row: (not row.get(key_name), row.get("captured_at") or "", row.get("media_id") or ""))
    return candidates[: max(limit, 0)]


def _pilot_metadata(repository, *, date: str) -> dict[str, Any]:
    event_hidden_by_media_id: dict[str, list[bool]] = {}
    event_evidence_media_ids: set[str] = set()
    for event in repository.list_events(start_date=date, end_date=date, include_hidden=True, limit=20_000):
        hidden = bool(int(event.get("is_hidden") or 0))
        for evidence in repository.list_event_evidence(str(event.get("id") or "")):
            if evidence.get("evidence_type") != "photo":
                continue
            media_id = str(evidence.get("evidence_id") or "")
            if not media_id:
                continue
            event_evidence_media_ids.add(media_id)
            event_hidden_by_media_id.setdefault(media_id, []).append(hidden)
    ocr_media_ids = {
        str(row.get("media_id") or "")
        for row in repository.list_media_ocr(start_date=date, end_date=date, statuses=["success"], limit=1_000_000)
        if row.get("media_id")
    }
    return {
        "event_hidden_by_media_id": event_hidden_by_media_id,
        "event_evidence_media_ids": event_evidence_media_ids,
        "ocr_media_ids": ocr_media_ids,
    }


def _time_spread(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    ordered = sorted(candidates, key=lambda row: (row.get("captured_at") or "", row.get("media_id") or ""))
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[0]]
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    max_index = len(ordered) - 1
    for slot in range(limit):
        index = round(slot * max_index / (limit - 1))
        candidate = ordered[index]
        media_id = str(candidate.get("media_id"))
        if media_id in used:
            continue
        selected.append(candidate)
        used.add(media_id)
    for candidate in ordered:
        if len(selected) >= limit:
            break
        media_id = str(candidate.get("media_id"))
        if media_id not in used:
            selected.append(candidate)
            used.add(media_id)
    return sorted(selected, key=lambda row: (row.get("captured_at") or "", row.get("media_id") or ""))


def _ensure_priority_coverage(
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if not selected or limit <= 0:
        return selected[:limit]
    working = list(selected[:limit])
    for key in ["has_event_evidence", "has_ocr", "has_gps"]:
        if any(row.get(key) for row in working):
            continue
        replacement = next((row for row in candidates if row.get(key)), None)
        if replacement is None or any(row["media_id"] == replacement["media_id"] for row in working):
            continue
        replace_index = next((idx for idx, row in enumerate(reversed(working)) if not row.get("has_event_evidence") and not row.get("has_ocr") and not row.get("has_gps")), 0)
        actual_index = max(len(working) - 1 - replace_index, 0)
        working[actual_index] = replacement
    return sorted(working, key=lambda row: (row.get("captured_at") or "", row.get("media_id") or ""))


def _processed_images(repository, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for selected_row in selected:
        media_id = str(selected_row["media_id"])
        vlm = repository.get_media_vlm(media_id)
        if vlm:
            rows.append(
                {
                    **selected_row,
                    "status": vlm.get("status") or "",
                    "caption": redact_text(vlm.get("caption"), max_chars=160),
                    "short_caption": redact_text(vlm.get("short_caption"), max_chars=100),
                    "scene_tags": _json_list(vlm.get("scene_tags_json")),
                    "object_tags": _json_list(vlm.get("object_tags_json")),
                    "activity_tags": _json_list(vlm.get("activity_tags_json")),
                    "food_cues": _json_list(vlm.get("food_cues_json")),
                    "location_cues": _json_list(vlm.get("location_cues_json")),
                    "safety_flags": _json_list(vlm.get("safety_flags_json")),
                    "evidence_strength": vlm.get("evidence_strength") or "weak",
                }
            )
        else:
            rows.append({**selected_row, "status": "pending"})
    return rows


def _safety_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flag_counts: dict[str, int] = {}
    forbidden_terms_found = 0
    for row in rows:
        text = " ".join(
            str(value or "")
            for value in [
                row.get("caption"),
                row.get("short_caption"),
                " ".join(row.get("scene_tags") or []),
                " ".join(row.get("activity_tags") or []),
                " ".join(row.get("food_cues") or []),
                " ".join(row.get("location_cues") or []),
            ]
        )
        if any(term in text for term in FORBIDDEN_TERMS):
            forbidden_terms_found += 1
        for flag in row.get("safety_flags") or []:
            flag_counts[str(flag)] = flag_counts.get(str(flag), 0) + 1
    return {
        "forbidden_terms_found": forbidden_terms_found,
        "relationship_inference_removed": flag_counts.get("relationship_inference_removed", 0),
        "emotion_inference_removed": flag_counts.get("emotion_inference_removed", 0),
        "people_present_count": flag_counts.get("people_present", 0),
        "overclaim_softened": flag_counts.get("overclaim_softened", 0),
        "safety_flags_count": sum(flag_counts.values()),
        "flag_counts": dict(sorted(flag_counts.items())),
    }


def _search_smoke_tests(repository, date: str) -> dict[str, Any]:
    smoke: dict[str, Any] = {}
    for query in DEFAULT_PILOT_IMAGE_QUERIES:
        report = image_search(repository, ImageSearchOptions(query=query, date_from=date, date_to=date, limit=5))
        smoke[f"image-search {query}"] = {
            "total": report.get("total", 0),
            "results": _compact_image_results(report.get("results") or []),
        }
    routed = route_query(repository, DEFAULT_PILOT_QA_QUERY, limit=5, include_hidden=False).to_dict()
    smoke[f"qa {DEFAULT_PILOT_QA_QUERY}"] = {
        "intent": routed.get("intent"),
        "routing": routed.get("routing"),
        "total": len(routed.get("results") or []),
        "results": routed.get("results") or [],
        "answer_preview": redact_text(routed.get("answer"), max_chars=500),
    }
    return smoke


def _compact_image_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rows[:5]:
        compact.append(
            {
                "date": row.get("date"),
                "media_id": row.get("media_id"),
                "file_name": redact_text(row.get("file_name"), max_chars=80),
                "confidence": row.get("confidence"),
                "score": row.get("score"),
                "evidence_types": row.get("evidence_types") or [],
                "matched_fields": row.get("matched_fields") or [],
                "caption": redact_text(row.get("caption"), max_chars=120),
                "ocr_preview": redact_text(row.get("ocr_preview"), max_chars=100),
            }
        )
    return compact


def _base_report(
    *,
    options: VlmPilotOptions,
    created_at: datetime,
    engine_name: str,
    model_name: str | None,
    selected: list[dict[str, Any]],
    before_stats: dict[str, Any],
    after_stats: dict[str, Any],
    db_safety: dict[str, Any],
    vlm_report: dict[str, Any],
    processed_images: list[dict[str, Any]],
    safety_summary: dict[str, Any],
    search_smoke_tests: dict[str, Any],
    recommendation: str,
) -> dict[str, Any]:
    return {
        "run_info": {
            "date": options.date,
            "engine": engine_name,
            "model": model_name,
            "prompt_template": options.prompt_template,
            "limit": options.limit,
            "strategy": options.strategy,
            "created_at": created_at.isoformat(timespec="seconds"),
            "dry_run": options.dry_run,
            "force": options.force,
            "skip_existing": options.skip_existing,
            "include_hidden": options.include_hidden,
        },
        "db_safety": db_safety,
        "vlm_stats_before": before_stats,
        "vlm_stats_after": after_stats,
        "selected_images": selected,
        "vlm_report": vlm_report,
        "processed_images": processed_images,
        "safety_summary": safety_summary,
        "search_smoke_tests": search_smoke_tests,
        "recommendation": recommendation,
    }


def _empty_vlm_report_payload(
    *,
    selected_count: int,
    engine_name: str,
    model_name: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "selected_images": selected_count,
        "processed": 0,
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "no_visual_content": 0,
        "engine_unavailable": 0,
        "dry_run": dry_run,
        "engine": engine_name,
        "model_name": model_name,
        "rows": [],
    }


def _add_pilot_summary(vlm_report: dict[str, Any], safety_summary: dict[str, Any]) -> dict[str, Any]:
    output = dict(vlm_report)
    denominator = max(
        int(output.get("processed") or 0),
        sum(
            int(output.get(key) or 0)
            for key in ["success", "failed", "engine_unavailable", "skipped", "no_visual_content"]
        ),
    )
    success = int(output.get("success") or 0)
    failed = int(output.get("failed") or 0)
    engine_unavailable = int(output.get("engine_unavailable") or 0)
    output["success_rate"] = _rate(success, denominator)
    output["failed_rate"] = _rate(failed, denominator)
    output["engine_unavailable_rate"] = _rate(engine_unavailable, denominator)
    output["people_present_count"] = int(safety_summary.get("people_present_count") or 0)
    output["safety_flags_count"] = int(safety_summary.get("safety_flags_count") or 0)
    output["json_parse_failure_count"] = _json_parse_failure_count(output, safety_summary)
    output["recommendation"] = _recommendation(output, safety_summary)
    return output


def _rate(count: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(count / denominator, 3)


def _json_parse_failure_count(vlm_report: dict[str, Any], safety_summary: dict[str, Any]) -> int:
    count = int((safety_summary.get("flag_counts") or {}).get("json_parse_failed", 0))
    for row in vlm_report.get("rows") or []:
        error_text = str(row.get("error_message") or row.get("error") or "")
        lowered = error_text.lower()
        if "json" in lowered or "parse" in lowered:
            count += 1
    return count


def _recommendation(vlm_report: dict[str, Any], safety_summary: dict[str, Any]) -> str:
    failed = int(vlm_report.get("failed") or 0)
    engine_unavailable = int(vlm_report.get("engine_unavailable") or 0)
    success = int(vlm_report.get("success") or 0)
    rows = vlm_report.get("rows") or []
    error_text = " ".join(str(row.get("error_message") or row.get("error") or "") for row in rows).lower()
    if any(term in error_text for term in ["cuda out of memory", "outofmemory", "oom", "image size", "pixel"]):
        return "reduce_image_size"
    if int(vlm_report.get("json_parse_failure_count") or 0) > 0:
        return "adjust_prompt"
    if engine_unavailable > 0:
        return "inspect_failures"
    if success == 0 and not bool(vlm_report.get("dry_run")):
        return "inspect_failures"
    if failed and float(vlm_report.get("failed_rate") or 0.0) >= 0.5:
        return "inspect_failures"
    risky = (
        int(safety_summary.get("relationship_inference_removed") or 0)
        + int(safety_summary.get("emotion_inference_removed") or 0)
        + int(safety_summary.get("forbidden_terms_found") or 0)
    )
    if risky:
        return "adjust_prompt"
    return "continue_to_20"


def _json_list(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return [str(raw)]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    return [str(parsed)] if str(parsed).strip() else []
