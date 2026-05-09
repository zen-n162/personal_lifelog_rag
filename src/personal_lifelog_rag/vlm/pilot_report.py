"""Privacy-conscious VLM pilot report formatting and persistence."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from personal_lifelog_rag.core.privacy import redact_text


DEFAULT_VLM_PILOT_OUTPUT_DIR = Path("eval_outputs/vlm_pilot")


def write_vlm_pilot_report(
    report: dict[str, Any],
    *,
    output_dir: str | Path = DEFAULT_VLM_PILOT_OUTPUT_DIR,
    now: datetime | None = None,
) -> dict[str, str]:
    """Write JSON and Markdown reports under an ignored eval output directory."""

    created = now or datetime.now()
    date_label = str(report.get("run_info", {}).get("date") or "unknown").replace("-", "")
    timestamp = created.strftime("%Y%m%d_%H%M%S")
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / f"vlm_pilot_{date_label}_{timestamp}.json"
    markdown_path = destination / f"vlm_pilot_{date_label}_{timestamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    markdown_path.write_text(format_vlm_pilot_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def format_vlm_pilot_report(report: dict[str, Any]) -> str:
    run = report.get("run_info", {})
    summary = report.get("vlm_report", {})
    safety = report.get("safety_summary", {})
    output_paths = report.get("output_paths") or {}
    lines = [
        "VLM Pilot",
        f"- date: {run.get('date')}",
        f"- engine: {run.get('engine')}",
        f"- model: {run.get('model') or ''}",
        f"- prompt_template: {run.get('prompt_template')}",
        f"- limit: {run.get('limit')}",
        f"- strategy: {run.get('strategy')}",
        f"- dry_run: {run.get('dry_run')}",
        f"- selected images: {len(report.get('selected_images', []))}",
        f"- processed: {summary.get('processed', 0)}",
        f"- success: {summary.get('success', 0)}",
        f"- failed: {summary.get('failed', 0)}",
        f"- skipped: {summary.get('skipped', 0)}",
        f"- engine_unavailable: {summary.get('engine_unavailable', 0)}",
        f"- success_rate: {summary.get('success_rate', 0.0)}",
        f"- failed_rate: {summary.get('failed_rate', 0.0)}",
        f"- engine_unavailable_rate: {summary.get('engine_unavailable_rate', 0.0)}",
        f"- json_parse_failure_count: {summary.get('json_parse_failure_count', 0)}",
        f"- db_check_strict_ok: {report.get('db_safety', {}).get('strict_ok')}",
        f"- backup: {report.get('db_safety', {}).get('backup_path') or ''}",
        "Safety:",
        f"- forbidden_terms_found: {safety.get('forbidden_terms_found', 0)}",
        f"- relationship_inference_removed: {safety.get('relationship_inference_removed', 0)}",
        f"- emotion_inference_removed: {safety.get('emotion_inference_removed', 0)}",
        f"- people_present: {safety.get('people_present_count', 0)}",
        f"- safety_flags_count: {safety.get('safety_flags_count', summary.get('safety_flags_count', 0))}",
        f"- overclaim_softened: {safety.get('overclaim_softened', 0)}",
        f"- recommendation: {report.get('recommendation') or summary.get('recommendation') or ''}",
    ]
    if output_paths:
        lines.extend(["Reports:", f"- json: {output_paths.get('json')}", f"- markdown: {output_paths.get('markdown')}"])
    if run.get("dry_run"):
        lines.append("Selected images:")
        for row in (report.get("selected_images") or [])[:50]:
            lines.append(
                f"- {row.get('captured_at') or ''} {row.get('media_id')} "
                f"{redact_text(row.get('file_name'), max_chars=80)} "
                f"event={row.get('has_event_evidence')} ocr={row.get('has_ocr')} gps={row.get('has_gps')}"
            )
    return "\n".join(lines)


def format_vlm_pilot_markdown(report: dict[str, Any]) -> str:
    run = report.get("run_info", {})
    safety = report.get("safety_summary", {})
    db_safety = report.get("db_safety", {})
    before = report.get("vlm_stats_before", {})
    after = report.get("vlm_stats_after", {})
    vlm_report = report.get("vlm_report", {})
    lines = [
        "# VLM Pilot Report",
        "",
        "## Run Info",
        f"- date: {run.get('date')}",
        f"- engine: {run.get('engine')}",
        f"- model: {run.get('model') or ''}",
        f"- prompt_template: {run.get('prompt_template')}",
        f"- limit: {run.get('limit')}",
        f"- strategy: {run.get('strategy')}",
        f"- created_at: {run.get('created_at')}",
        f"- dry_run: {run.get('dry_run')}",
        "",
        "## DB Safety",
        f"- backup path: {db_safety.get('backup_path') or ''}",
        f"- db-check strict result: {db_safety.get('strict_ok')}",
        "",
        "## VLM Stats Before/After",
        f"- total media_vlm: {before.get('total_media_vlm', 0)} -> {after.get('total_media_vlm', 0)}",
        f"- success: {_status_count(before, 'success')} -> {_status_count(after, 'success')}",
        f"- failed: {_status_count(before, 'failed')} -> {_status_count(after, 'failed')}",
        f"- engine_unavailable: {_status_count(before, 'engine_unavailable')} -> {_status_count(after, 'engine_unavailable')}",
        f"- processed: {vlm_report.get('processed', 0)}",
        f"- success rate: {vlm_report.get('success_rate', 0.0)}",
        f"- failed rate: {vlm_report.get('failed_rate', 0.0)}",
        f"- engine_unavailable rate: {vlm_report.get('engine_unavailable_rate', 0.0)}",
        f"- JSON parse failure count: {vlm_report.get('json_parse_failure_count', 0)}",
        "",
        "## Processed Images",
    ]
    images = report.get("processed_images") or report.get("selected_images") or []
    if images:
        for image in images:
            lines.extend(
                [
                    f"- media_id: {image.get('media_id')}",
                    f"  - captured_at: {image.get('captured_at') or ''}",
                    f"  - file_name: {redact_text(image.get('file_name'), max_chars=80)}",
                    f"  - status: {image.get('status') or ''}",
                    f"  - short_caption: {redact_text(image.get('short_caption') or image.get('caption'), max_chars=100)}",
                    f"  - scene_tags: {', '.join(image.get('scene_tags', []))}",
                    f"  - activity_tags: {', '.join(image.get('activity_tags', []))}",
                    f"  - food_cues: {', '.join(image.get('food_cues', []))}",
                    f"  - location_cues: {', '.join(image.get('location_cues', []))}",
                    f"  - safety_flags: {', '.join(image.get('safety_flags', []))}",
                    f"  - evidence_strength: {image.get('evidence_strength') or 'weak'}",
                ]
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Safety Summary",
            f"- forbidden terms found: {safety.get('forbidden_terms_found', 0)}",
            f"- relationship inference removed: {safety.get('relationship_inference_removed', 0)}",
            f"- emotion inference removed: {safety.get('emotion_inference_removed', 0)}",
            f"- people_present count: {safety.get('people_present_count', 0)}",
            f"- safety_flags count: {safety.get('safety_flags_count', vlm_report.get('safety_flags_count', 0))}",
            f"- overclaim softened count: {safety.get('overclaim_softened', 0)}",
            "",
            "## Search Smoke Test",
        ]
    )
    smoke = report.get("search_smoke_tests", {})
    for key in ["image-search ご飯", "image-search カフェ", "image-search 新宿", "qa ご飯を食べた写真はいつ？"]:
        value = smoke.get(key) or {}
        lines.append(f"- {key}: {value.get('total', len(value.get('results', [])) if isinstance(value, dict) else 0)} result(s)")
    lines.extend(["", "## Recommendation", f"- {report.get('recommendation') or 'review results before expanding the batch'}"])
    return "\n".join(lines) + "\n"


def _status_count(report: dict[str, Any], status: str) -> int:
    return int((report.get("status_counts") or {}).get(status, 0))
