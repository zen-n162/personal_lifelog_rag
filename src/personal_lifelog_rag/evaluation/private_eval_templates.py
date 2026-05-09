"""Generate local-only private eval templates from aggregate DB state."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


DEFAULT_FORBIDDEN_PHRASES = ["確実に", "断定", "デートしていた"]
DEFAULT_VLM_FORBIDDEN_TERMS = ["彼女", "恋人", "家族", "病気", "宗教", "政治"]


@dataclass(frozen=True)
class PrivateEvalTemplateSummary:
    path: Path
    date: str
    case_count: int
    event_count: int
    evidence_types: list[str]
    vlm_success_count: int
    vlm_engine: str | None
    completed_call_count: int


def build_private_eval_template_for_date(repository, *, date: str) -> tuple[str, PrivateEvalTemplateSummary]:
    """Build a compact YAML template for one date without raw LINE/photo content."""

    events = repository.list_events(start_date=date, end_date=date, include_hidden=False, limit=10_000)
    evidence_types = _event_evidence_types(repository, events)
    vlm_rows = repository.list_media_vlm(start_date=date, end_date=date, limit=100_000)
    vlm_success_rows = [row for row in vlm_rows if row.get("status") == "success"]
    vlm_engine = _most_common_text(row.get("vlm_engine") for row in vlm_success_rows)
    completed_calls = repository.list_line_call_events(
        start_date=date,
        end_date=date,
        statuses=["completed"],
        limit=100_000,
    )

    cases: list[dict[str, Any]] = [
        {
            "id": f"date_{_compact_date(date)}_summary",
            "type": "date_qa",
            "question": f"{date}は何していた？",
            "expected_dates": [date],
            "expected_min_events": _min_expected_events(len(events)),
            "expected_evidence_types": _expected_evidence_types(evidence_types),
            "should_not_include": DEFAULT_FORBIDDEN_PHRASES,
        },
        {
            "id": f"place_{_compact_date(date)}_shinjuku",
            "type": "routed_qa",
            "question": "新宿に行ったのはいつ？",
            "expected_intent": "place_visit",
            "expected_top_dates": [date],
            "expected_classification": {date: "actual_or_likely_action"},
            "should_not_include": ["確実に新宿に行った", "断定"],
        },
        {
            "id": f"image_food_{_compact_date(date)}",
            "type": "image_search",
            "query": "ご飯",
            "expected_top_dates": [date],
            "expected_evidence_types": ["vlm"],
            "expected_min_results": 1,
            "should_not_include": ["確実に食べた", "断定"],
        },
        {
            "id": f"mm_food_photo_{_compact_date(date)}",
            "type": "multimodal_search",
            "query": "ご飯を食べた写真",
            "expected_top_dates": [date],
            "expected_evidence_types_any": ["vlm", "event"],
            "expected_min_results": 1,
            "max_vlm_only_confidence": "中",
            "should_not_include": ["確実に食べた", "断定"],
        },
        {
            "id": f"vlm_quality_{_compact_date(date)}",
            "type": "vlm_quality",
            "date": date,
            "expected_min_vlm_success": _min_expected_vlm_success(len(vlm_success_rows)),
            "expected_engine": vlm_engine or "qwen3_vl_transformers",
            "forbidden_terms": DEFAULT_VLM_FORBIDDEN_TERMS,
            "allowed_safety_flags": ["people_present", "no_people_present"],
        },
        {
            "id": f"event_quality_{_compact_date(date)}",
            "type": "event_quality",
            "date": date,
            "expected_min_events": _min_expected_events(len(events)),
            "expected_max_events": max(_min_expected_events(len(events)) + 2, len(events) + 3),
            "expected_evidence_types": _expected_evidence_types(evidence_types),
            "max_vlm_only_high_confidence_events": 0,
            "no_orphan_evidence": True,
        },
        {
            "id": f"search_place_{_compact_date(date)}_shinjuku",
            "type": "keyword_search",
            "query": "新宿",
            "intent": "place_visit",
            "expected_top_dates": [date],
            "expected_classification": {date: "actual_or_likely_action"},
            "should_not_include": ["確実に新宿に行った", "断定"],
        },
    ]
    if completed_calls:
        cases.append(
            {
                "id": f"call_{_compact_date(date)}",
                "type": "call_search",
                "filters": {"date": date, "completed": True},
                "expected_min_results": 1,
                "expected_status": "completed",
            }
        )

    text = _render_yaml(cases, date=date)
    summary = PrivateEvalTemplateSummary(
        path=Path(),
        date=date,
        case_count=len(cases),
        event_count=len(events),
        evidence_types=_expected_evidence_types(evidence_types),
        vlm_success_count=len(vlm_success_rows),
        vlm_engine=vlm_engine,
        completed_call_count=len(completed_calls),
    )
    return text, summary


def write_private_eval_template_for_date(repository, *, date: str, output_path: str | Path) -> PrivateEvalTemplateSummary:
    """Write a one-date private eval YAML file and return compact stats."""

    text, summary = build_private_eval_template_for_date(repository, date=date)
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return PrivateEvalTemplateSummary(
        path=path,
        date=summary.date,
        case_count=summary.case_count,
        event_count=summary.event_count,
        evidence_types=summary.evidence_types,
        vlm_success_count=summary.vlm_success_count,
        vlm_engine=summary.vlm_engine,
        completed_call_count=summary.completed_call_count,
    )


def format_private_eval_template_summary(summary: PrivateEvalTemplateSummary) -> str:
    lines = [
        "Private eval template generated",
        f"- path: {summary.path}",
        f"- date: {summary.date}",
        f"- cases: {summary.case_count}",
        f"- events on date: {summary.event_count}",
        f"- evidence types: {', '.join(summary.evidence_types) if summary.evidence_types else '(none)'}",
        f"- VLM success rows: {summary.vlm_success_count}",
        f"- VLM engine: {summary.vlm_engine or '(none)'}",
        f"- completed calls on date: {summary.completed_call_count}",
        "",
        "Review and edit this local file before treating it as a stable regression suite.",
        "Do not commit private_eval/ files.",
    ]
    return "\n".join(lines)


def _event_evidence_types(repository, events: list[dict[str, Any]]) -> set[str]:
    evidence_types: set[str] = set()
    for event in events:
        for evidence in repository.list_event_evidence(str(event.get("id") or "")):
            evidence_type = str(evidence.get("evidence_type") or "").strip()
            if evidence_type:
                evidence_types.add(evidence_type)
    return evidence_types


def _expected_evidence_types(evidence_types: set[str]) -> list[str]:
    preferred = [item for item in ["line", "photo", "ocr", "vlm"] if item in evidence_types]
    return preferred or sorted(evidence_types)


def _min_expected_events(count: int) -> int:
    if count <= 0:
        return 1
    return min(3, count)


def _min_expected_vlm_success(count: int) -> int:
    if count <= 0:
        return 1
    if count <= 3:
        return count
    return max(1, int(count * 0.85))


def _most_common_text(values) -> str | None:
    counter = Counter(str(value) for value in values if str(value or "").strip())
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def _compact_date(value: str) -> str:
    return value.replace("-", "")


def _render_yaml(cases: list[dict[str, Any]], *, date: str) -> str:
    lines = [
        "# Generated local private eval template.",
        "# This file may encode private dates and local expectations.",
        "# Do not commit private_eval/ files.",
        f"# baseline_date: {_yaml_scalar(date)}",
        "",
        "cases:",
    ]
    for case in cases:
        lines.extend(_render_case(case, indent=2))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_case(case: dict[str, Any], *, indent: int) -> list[str]:
    prefix = " " * indent
    lines: list[str] = []
    first = True
    for key, value in case.items():
        marker = "-" if first else " "
        first = False
        rendered = _render_value(key, value, indent=indent)
        if rendered:
            head, *tail = rendered
            lines.append(f"{prefix}{marker} {head}")
            lines.extend(tail)
    return lines


def _render_value(key: str, value: Any, *, indent: int) -> list[str]:
    nested_prefix = " " * (indent + 4)
    if isinstance(value, dict):
        lines = [f"{key}:"]
        for child_key, child_value in value.items():
            lines.append(f"{nested_prefix}{_yaml_scalar(child_key)}: {_yaml_scalar(child_value)}")
        return lines
    if isinstance(value, list):
        lines = [f"{key}:"]
        for item in value:
            lines.append(f"{nested_prefix}- {_yaml_scalar(item)}")
        return lines
    return [f"{key}: {_yaml_scalar(value)}"]


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)
