from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from personal_lifelog_rag.vlm.pilot_report import format_vlm_pilot_markdown, write_vlm_pilot_report


def test_vlm_pilot_markdown_contains_safety_and_smoke_sections() -> None:
    report = _sample_report()

    markdown = format_vlm_pilot_markdown(report)

    assert "# VLM Pilot Report" in markdown
    assert "## Safety Summary" in markdown
    assert "## Search Smoke Test" in markdown
    assert "relationship inference removed: 1" in markdown
    assert "success rate: 1.0" in markdown
    assert "continue_to_20" in markdown


def test_write_vlm_pilot_report_writes_json_and_markdown(tmp_path: Path) -> None:
    report = _sample_report()

    paths = write_vlm_pilot_report(
        report,
        output_dir=tmp_path,
        now=datetime(2026, 5, 9, 6, 0, 0),
    )

    json_path = Path(paths["json"])
    markdown_path = Path(paths["markdown"])
    assert json_path.name.startswith("vlm_pilot_20241224_20260509_060000")
    assert markdown_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["run_info"]["date"] == "2024-12-24"


def _sample_report() -> dict:
    return {
        "run_info": {
            "date": "2024-12-24",
            "engine": "fake",
            "model": "fake-vlm",
            "prompt_template": "lifelog_structured_tags_v1",
            "limit": 1,
            "strategy": "time_spread",
            "created_at": "2026-05-09T06:00:00",
            "dry_run": False,
        },
        "db_safety": {"backup_path": "backups/lifelog.sqlite", "strict_ok": True},
        "vlm_stats_before": {"total_media_vlm": 0, "status_counts": {}},
        "vlm_stats_after": {"total_media_vlm": 1, "status_counts": {"success": 1}},
        "selected_images": [],
        "vlm_report": {
            "processed": 1,
            "success": 1,
            "failed": 0,
            "skipped": 0,
            "engine_unavailable": 0,
            "success_rate": 1.0,
            "failed_rate": 0.0,
            "engine_unavailable_rate": 0.0,
            "json_parse_failure_count": 0,
            "safety_flags_count": 1,
            "recommendation": "continue_to_20",
        },
        "processed_images": [
            {
                "media_id": "media_1",
                "captured_at": "2024-12-24T10:00:00+09:00",
                "file_name": "dummy.png",
                "status": "success",
                "short_caption": "dummy",
                "scene_tags": ["indoor"],
                "activity_tags": ["meal_possible"],
                "food_cues": ["ramen_possible"],
                "location_cues": [],
                "safety_flags": ["relationship_inference_removed"],
                "evidence_strength": "weak",
            }
        ],
        "safety_summary": {
            "forbidden_terms_found": 0,
            "relationship_inference_removed": 1,
            "emotion_inference_removed": 0,
            "people_present_count": 0,
            "overclaim_softened": 0,
            "safety_flags_count": 1,
        },
        "search_smoke_tests": {"image-search ご飯": {"total": 1, "results": []}},
        "recommendation": "continue_to_20",
    }
