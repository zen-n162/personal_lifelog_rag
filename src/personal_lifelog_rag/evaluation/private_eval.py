"""Private local evaluation for answer and event quality.

This module intentionally avoids external APIs and keeps saved reports compact:
it records counts, pass/fail checks, and short previews rather than raw LINE
history or full answers.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from personal_lifelog_rag.core.privacy import redact_text
from personal_lifelog_rag.embeddings.multimodal_search import multimodal_search
from personal_lifelog_rag.embeddings.schemas import MultimodalSearchOptions
from personal_lifelog_rag.retrieval.evidence_strength import confidence_at_most, strength_at_least
from personal_lifelog_rag.line.call_index import search_calls
from personal_lifelog_rag.retrieval.answer_builder import build_answer
from personal_lifelog_rag.retrieval.date_parser import parse_date_query
from personal_lifelog_rag.retrieval.local_search import LocalSearchOptions, local_text_search
from personal_lifelog_rag.retrieval.query_intent import classify_query_intent
from personal_lifelog_rag.retrieval.query_router import route_query
from personal_lifelog_rag.retrieval.temporal_search import (
    TimelineSearchResult,
    build_timeline_items,
    search_timeline,
)
from personal_lifelog_rag.vlm.schemas import ImageSearchOptions
from personal_lifelog_rag.vlm.prompts import get_vlm_prompt_template
from personal_lifelog_rag.vlm.review_service import apply_vlm_override_to_result
from personal_lifelog_rag.vlm.safety import safety_check_text
from personal_lifelog_rag.vlm.vlm_service import image_search


DEFAULT_QUESTIONS_PATH = Path("private_eval/questions.yaml")
DEFAULT_RUNS_DIR = Path("private_eval/runs")
ANSWER_REQUIRED_SECTIONS = ("根拠:", "信頼度:")
CLAIM_SUPPORT_KEYWORDS = {
    "food": ("食事", "カフェ", "ご飯", "ラーメン", "食べ", "おいしかった"),
    "meeting": ("待ち合わせ", "待って", "着く", "駅", "東口", "西口", "合流", "集合"),
    "call": ("通話", "不在着信", "電話"),
}


@dataclass(frozen=True)
class PrivateEvalQuestion:
    id: str
    question: str
    case_type: str = "date_qa"
    query: str | None = None
    expected_date: str | None = None
    expected_dates: list[str] = field(default_factory=list)
    require_events: bool = False
    min_events: int | None = None
    expected_keywords: list[str] = field(default_factory=list)
    expected_evidence_types: list[str] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    max_confidence_for_activity: str | None = None
    intent: str | None = None
    expected_top_dates: list[str] = field(default_factory=list)
    expected_top_dates_any: list[str] = field(default_factory=list)
    expected_classification: dict[str, str] = field(default_factory=dict)
    expected_evidence_keywords: list[str] = field(default_factory=list)
    should_downrank_phrases: list[str] = field(default_factory=list)
    expected_intent: str | None = None
    expected_entities: dict[str, str] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    expected_top_contains_any: list[str] = field(default_factory=list)
    expected_min_score: float | None = None
    expected_max_rank: int | None = None
    expected_min_results: int | None = None
    expected_status: str | None = None
    should_not_include_status: list[str] = field(default_factory=list)
    expected_date_any: list[str] = field(default_factory=list)
    expected_sender_any: list[str] = field(default_factory=list)
    date: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    expected_max_events: int | None = None
    max_line_only_low_value_events: int | None = None
    min_photo_and_line_events: int | None = None
    no_orphan_evidence: bool | None = None
    expected_any_location_name: list[str] = field(default_factory=list)
    forbidden_exact_gps_in_answer: bool = False
    event_id: str | None = None
    media_id: str | None = None
    expected_hidden: bool | None = None
    expected_pinned: bool | None = None
    expected_verified: bool | None = None
    expected_searchable: bool | None = None
    expected_event_usable: bool | None = None
    title_override_contains: str | None = None
    summary_override_contains: str | None = None
    expected_media_ids: list[str] = field(default_factory=list)
    should_exclude_media_ids: list[str] = field(default_factory=list)
    expected_min_ocr_success: int | None = None
    expected_min_ocr_processed: int | None = None
    allow_engine_unavailable: bool = False
    expected_min_vlm_success: int | None = None
    max_failed_vlm_rows: int | None = None
    expected_engine: str | None = None
    allowed_safety_flags: list[str] = field(default_factory=list)
    forbidden_terms: list[str] = field(default_factory=list)
    input_text: str | None = None
    template: str | None = None
    expected_flags: list[str] = field(default_factory=list)
    expected_contains: list[str] = field(default_factory=list)
    expected_no_overclaim: bool = False
    max_vlm_only_high_confidence_events: int | None = None
    expected_evidence_types_any: list[str] = field(default_factory=list)
    expected_classification_any: list[str] = field(default_factory=list)
    max_embedding_only_rank: int | None = None
    max_vlm_only_confidence: str | None = None
    expected_strength_at_least: str | None = None


def load_private_eval_questions(path: str | Path) -> list[PrivateEvalQuestion]:
    """Load private eval questions from a small YAML/JSON file."""

    source = Path(path).expanduser()
    raw_text = source.read_text(encoding="utf-8")
    payload = _load_json_or_simple_yaml(raw_text)
    if isinstance(payload, dict):
        rows = payload.get("cases", payload.get("questions", []))
    else:
        rows = payload if isinstance(payload, list) else []
    questions: list[PrivateEvalQuestion] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        case_type = str(row.get("type") or "date_qa")
        question = str(
            row.get("question")
            or row.get("query")
            or row.get("date")
            or row.get("event_id")
            or row.get("media_id")
            or row.get("input_text")
            or row.get("template")
            or row.get("id")
            or ""
        ).strip()
        if not question:
            continue
        expected_dates = _string_list(row.get("expected_dates"))
        expected_date = _none_or_str(row.get("expected_date")) or (expected_dates[0] if expected_dates else None)
        if expected_date and expected_date not in expected_dates:
            expected_dates.insert(0, expected_date)
        questions.append(
            PrivateEvalQuestion(
                id=str(row.get("id") or f"q{index}"),
                question=question,
                case_type=case_type,
                query=_none_or_str(row.get("query")),
                expected_date=expected_date,
                expected_dates=expected_dates,
                require_events=bool(row.get("require_events", False)),
                min_events=_none_or_int(row.get("min_events", row.get("expected_min_events"))),
                expected_max_events=_none_or_int(row.get("expected_max_events")),
                expected_keywords=_string_list(row.get("expected_keywords"))
                + _string_list(row.get("expected_event_keywords")),
                expected_evidence_types=_string_list(row.get("expected_evidence_types")),
                forbidden_claims=_string_list(row.get("forbidden_claims"))
                + _string_list(row.get("should_not_include")),
                max_confidence_for_activity=_none_or_str(
                    row.get("max_confidence_for_activity", row.get("expected_max_activity_confidence"))
                ),
                intent=_none_or_str(row.get("intent")),
                expected_top_dates=_string_list(row.get("expected_top_dates")),
                expected_top_dates_any=_string_list(row.get("expected_top_dates_any")),
                expected_classification=_string_dict(row.get("expected_classification")),
                expected_evidence_keywords=_string_list(row.get("expected_evidence_keywords")),
                should_downrank_phrases=_string_list(row.get("should_downrank_phrases")),
                expected_intent=_none_or_str(row.get("expected_intent")),
                expected_entities=_string_dict(row.get("expected_entities")),
                filters=_dict_or_empty(row.get("filters")),
                expected_top_contains_any=_string_list(row.get("expected_top_contains_any")),
                expected_min_score=_none_or_float(row.get("expected_min_score")),
                expected_max_rank=_none_or_int(row.get("expected_max_rank")),
                expected_min_results=_none_or_int(row.get("expected_min_results")),
                expected_status=_none_or_str(row.get("expected_status")),
                should_not_include_status=_string_list(row.get("should_not_include_status")),
                expected_date_any=_string_list(row.get("expected_date_any")),
                expected_sender_any=_string_list(row.get("expected_sender_any")),
                date=_none_or_str(row.get("date")),
                date_from=_none_or_str(row.get("date_from", row.get("from"))),
                date_to=_none_or_str(row.get("date_to", row.get("to"))),
                max_line_only_low_value_events=_none_or_int(row.get("max_line_only_low_value_events")),
                min_photo_and_line_events=_none_or_int(row.get("min_photo_and_line_events")),
                no_orphan_evidence=_none_or_bool(row.get("no_orphan_evidence")),
                expected_any_location_name=_string_list(row.get("expected_any_location_name")),
                forbidden_exact_gps_in_answer=bool(row.get("forbidden_exact_gps_in_answer", False)),
                event_id=_none_or_str(row.get("event_id")),
                media_id=_none_or_str(row.get("media_id")),
                expected_hidden=_none_or_bool(row.get("expected_hidden")),
                expected_pinned=_none_or_bool(row.get("expected_pinned")),
                expected_verified=_none_or_bool(row.get("expected_verified")),
                expected_searchable=_none_or_bool(row.get("expected_searchable")),
                expected_event_usable=_none_or_bool(row.get("expected_event_usable")),
                title_override_contains=_none_or_str(row.get("title_override_contains")),
                summary_override_contains=_none_or_str(row.get("summary_override_contains")),
                expected_media_ids=_string_list(row.get("expected_media_ids")),
                should_exclude_media_ids=_string_list(row.get("should_exclude_media_ids")),
                expected_min_ocr_success=_none_or_int(row.get("expected_min_ocr_success")),
                expected_min_ocr_processed=_none_or_int(row.get("expected_min_ocr_processed")),
                allow_engine_unavailable=bool(row.get("allow_engine_unavailable", False)),
                expected_min_vlm_success=_none_or_int(row.get("expected_min_vlm_success")),
                max_failed_vlm_rows=_none_or_int(row.get("max_failed_vlm_rows")),
                expected_engine=_none_or_str(row.get("expected_engine")),
                allowed_safety_flags=_string_list(row.get("allowed_safety_flags")),
                forbidden_terms=_string_list(row.get("forbidden_terms")),
                input_text=_none_or_str(row.get("input_text")),
                template=_none_or_str(row.get("template")),
                expected_flags=_string_list(row.get("expected_flags")),
                expected_contains=_string_list(row.get("expected_contains")),
                expected_no_overclaim=bool(row.get("expected_no_overclaim", False)),
                max_vlm_only_high_confidence_events=_none_or_int(row.get("max_vlm_only_high_confidence_events")),
                expected_evidence_types_any=_string_list(row.get("expected_evidence_types_any")),
                expected_classification_any=_string_list(row.get("expected_classification_any")),
                max_embedding_only_rank=_none_or_int(row.get("max_embedding_only_rank")),
                max_vlm_only_confidence=_none_or_str(row.get("max_vlm_only_confidence")),
                expected_strength_at_least=_none_or_str(row.get("expected_strength_at_least")),
            )
        )
    return questions


def evaluate_private_questions(
    repository,
    questions: list[PrivateEvalQuestion],
    *,
    keyword_search_available: bool = True,
) -> dict[str, Any]:
    run_id = "eval_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    cases = [
        _evaluate_one(
            repository,
            question,
            keyword_search_available=keyword_search_available,
        )
        for question in questions
    ]
    passed = sum(1 for case in cases if case["status"] == "pass")
    failed = sum(1 for case in cases if case["status"] == "fail")
    skipped = sum(1 for case in cases if case["status"] == "skip")
    by_type = _by_type_summary(cases)
    ranking_metrics = _ranking_metrics(cases)
    safety_metrics = _safety_metrics(cases)
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "cases": len(cases),
            "total": len(cases),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
        },
        "by_type": by_type,
        "ranking_metrics": ranking_metrics,
        "safety_metrics": safety_metrics,
        "cases": cases,
        "case_results": cases,
    }


def write_private_eval_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    path = Path(output_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / f"{report['run_id']}.json"
    suffix = 1
    while output_path.exists():
        output_path = path / f"{report['run_id']}_{suffix}.json"
        suffix += 1
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return output_path


def compare_private_eval_reports(before_path: str | Path, after_path: str | Path) -> dict[str, Any]:
    before = json.loads(Path(before_path).expanduser().read_text(encoding="utf-8"))
    after = json.loads(Path(after_path).expanduser().read_text(encoding="utf-8"))
    before_cases = {case["id"]: case for case in before.get("case_results", before.get("cases", []))}
    after_cases = {case["id"]: case for case in after.get("case_results", after.get("cases", []))}
    newly_failed = [
        case_id for case_id, case in after_cases.items()
        if case.get("status") == "fail" and before_cases.get(case_id, {}).get("status") != "fail"
    ]
    improved = [
        case_id for case_id, case in after_cases.items()
        if case.get("status") == "pass" and before_cases.get(case_id, {}).get("status") == "fail"
    ]
    return {
        "before": _compare_summary(before),
        "after": _compare_summary(after),
        "delta": {
            "passed": _summary_value(after, "passed") - _summary_value(before, "passed"),
            "failed": _summary_value(after, "failed") - _summary_value(before, "failed"),
            "skipped": _summary_value(after, "skipped") - _summary_value(before, "skipped"),
            "top1_accuracy": _metric_delta(before, after, "top1_accuracy"),
            "expected_date_recall_at_5": _metric_delta(before, after, "expected_date_recall_at_5"),
            "forbidden_phrase_violations": _safety_delta(before, after, "forbidden_phrase_violations"),
        },
        "newly_failed_cases": newly_failed,
        "improved_cases": improved,
    }


def format_private_eval_comparison(report: dict[str, Any]) -> str:
    lines = [
        "Private Eval Compare",
        "",
        "summary delta:",
        f"- passed: {report['delta']['passed']:+d}",
        f"- failed: {report['delta']['failed']:+d}",
        f"- skipped: {report['delta']['skipped']:+d}",
        f"- top1 accuracy: {report['delta']['top1_accuracy']}",
        f"- expected date recall@5: {report['delta']['expected_date_recall_at_5']}",
        f"- forbidden phrase violations: {report['delta']['forbidden_phrase_violations']:+d}",
        "",
        "newly failed cases:",
    ]
    lines.extend(f"- {case_id}" for case_id in report["newly_failed_cases"]) if report["newly_failed_cases"] else lines.append("- none")
    lines.append("")
    lines.append("improved cases:")
    lines.extend(f"- {case_id}" for case_id in report["improved_cases"]) if report["improved_cases"] else lines.append("- none")
    return "\n".join(lines)


def format_private_eval_report(report: dict[str, Any], output_path: str | Path | None = None) -> str:
    summary = report["summary"]
    lines = [
        "Private Eval",
        "",
        f"cases: {summary.get('cases', summary.get('total', 0))}",
        f"passed: {summary['passed']}",
        f"failed: {summary['failed']}",
        f"skipped: {summary['skipped']}",
    ]
    if output_path is not None:
        lines.append(f"run_file: {output_path}")
    if report.get("by_type"):
        lines.extend(["", "by type:"])
        for case_type, stats in sorted(report["by_type"].items()):
            lines.append(f"- {case_type}: {stats['passed']}/{stats['total']} passed, skipped={stats['skipped']}")
    if report.get("ranking_metrics"):
        ranking = report["ranking_metrics"]
        lines.extend(
            [
                "",
                "ranking:",
                f"- top1 accuracy: {_format_metric(ranking.get('top1_accuracy'))}",
                f"- expected date recall@5: {_format_metric(ranking.get('expected_date_recall_at_5'))}",
            ]
        )
    if report.get("safety_metrics"):
        safety = report["safety_metrics"]
        lines.extend(
            [
                "",
                "safety:",
                f"- forbidden phrase violations: {safety.get('forbidden_phrase_violations', 0)}",
                f"- overclaim violations: {safety.get('overclaim_violations', 0)}",
            ]
        )
    lines.append("")
    for case in report["cases"]:
        status = str(case["status"]).upper()
        lines.append(f"[{status}] {case['id']}")
        lines.append(f"- question: {case['question_preview']}")
        if case["status"] == "skip":
            lines.append(f"- reason: {case.get('skip_reason') or 'skipped'}")
            lines.append("")
            continue
        lines.append(f"- expected date matched: {_matched_date_label(case)}")
        lines.append(f"- expected keywords matched: {_matched_keyword_label(case)}")
        lines.append(f"- forbidden phrases: {_forbidden_label(case)}")
        lines.append(
            "- records: "
            f"events={case['events_count']}, "
            f"event_evidence={case['event_evidence_count']}, "
            f"line={case['line_message_count']}, "
            f"photos={case['photo_count']}, "
            f"gps_photos={case['gps_photo_count']}"
        )
        if case.get("activity_confidence"):
            lines.append(f"- activity confidence: {case['activity_confidence']}")
        for issue in case["issues"]:
            lines.append(f"  - {issue}")
        lines.append("")
    return "\n".join(lines)


def _matched_date_label(case: dict[str, Any]) -> str:
    if not case.get("expected_dates"):
        return "not specified"
    matched = [date for date in case["expected_dates"] if date in case.get("matched_dates", [])]
    if case.get("parsed_date") in case.get("expected_dates", []):
        matched = [str(case["parsed_date"]), *[date for date in matched if date != case["parsed_date"]]]
    return ", ".join(matched) if matched else "none"


def _matched_keyword_label(case: dict[str, Any]) -> str:
    matched = case.get("matched_expected_keywords") or []
    missing = case.get("missing_expected_keywords") or []
    if not matched and not missing:
        return "not specified"
    if missing:
        return "matched " + (", ".join(matched) or "none") + "; missing " + ", ".join(missing)
    return ", ".join(matched)


def _forbidden_label(case: dict[str, Any]) -> str:
    found = case.get("forbidden_claims_found") or []
    return ", ".join(found) if found else "none found"


def _format_metric(value: Any) -> str:
    return "n/a" if value is None else str(value)


def write_private_eval_template(path: str | Path) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path
    output_path.write_text(
        """cases:
  - id: date_001
    type: date_qa
    question: "2024年12月24日は何していた？"
    expected_dates:
      - "2024-12-24"
    expected_min_events: 1
    expected_event_keywords:
      - "根拠:"
      - "信頼度:"
    expected_evidence_types:
      - "line"
      - "photo"
    should_not_include:
      - "確実に"
      - "デートしていた"
    max_confidence_for_activity: "高"
""",
        encoding="utf-8",
    )
    return output_path


def _evaluate_one(
    repository,
    question: PrivateEvalQuestion,
    *,
    keyword_search_available: bool,
) -> dict[str, Any]:
    if question.case_type == "keyword_search" and not keyword_search_available:
        return _skip_case(question, "search command is not implemented yet")
    if question.case_type == "query_intent":
        return _evaluate_query_intent_case(question)
    if question.case_type == "routed_qa":
        return _evaluate_routed_qa_case(repository, question)
    if question.case_type in {"time_range_summary", "monthly_summary"}:
        return _evaluate_time_range_summary_case(repository, question)
    if question.case_type == "call_search":
        return _evaluate_call_search_case(repository, question)
    if question.case_type == "event_quality":
        return _evaluate_event_quality_case(repository, question)
    if question.case_type == "event_rebuild_quality":
        return _evaluate_event_rebuild_quality_case(repository, question)
    if question.case_type == "place_assignment":
        return _evaluate_place_assignment_case(repository, question)
    if question.case_type == "event_override":
        return _evaluate_event_override_case(repository, question)
    if question.case_type == "vlm_review":
        return _evaluate_vlm_review_case(repository, question)
    if question.case_type == "ocr_quality":
        return _evaluate_ocr_quality_case(repository, question)
    if question.case_type == "ocr_search":
        return _evaluate_ocr_search_case(repository, question)
    if question.case_type == "vlm_quality":
        return _evaluate_vlm_quality_case(repository, question)
    if question.case_type == "vlm_safety":
        return _evaluate_vlm_safety_case(question)
    if question.case_type == "vlm_prompt":
        return _evaluate_vlm_prompt_case(question)
    if question.case_type == "image_search":
        return _evaluate_image_search_case(repository, question)
    if question.case_type == "multimodal_search":
        return _evaluate_multimodal_search_case(repository, question)
    if question.case_type not in {"date_qa", "keyword_search"}:
        return _skip_case(question, f"unsupported case type: {question.case_type}")
    if question.case_type == "keyword_search":
        return _evaluate_keyword_search_case(repository, question)

    date_range = None if question.case_type == "keyword_search" else parse_date_query(question.question)
    result = search_timeline(
        repository,
        question.question,
        date_range=date_range,
        keyword=question.query,
    )
    if question.case_type == "keyword_search" and question.query:
        result = _filter_keyword_result(result, question.query)
    answer = build_answer(question.question, result)
    parsed_date = date_range.start_iso if date_range else None
    matched_dates = _result_dates(result, keyword=question.query or result.keyword)
    if parsed_date:
        matched_dates = sorted(set([parsed_date, *matched_dates]))
    expected_date_match = (
        not question.expected_dates
        or parsed_date in question.expected_dates
        or any(date in question.expected_dates for date in matched_dates)
        or any(date in answer for date in question.expected_dates)
    )
    event_evidence_count = sum(int(event.get("event_evidence_count") or 0) for event in result.events)
    matched_keywords = [keyword for keyword in question.expected_keywords if keyword in answer]
    missing_keywords = [keyword for keyword in question.expected_keywords if keyword not in answer]
    missing_evidence_types = _missing_evidence_types(question.expected_evidence_types, result)
    forbidden_found = _forbidden_claims_found(answer, question.forbidden_claims)
    unsupported_claims = _unsupported_claims(answer, result)
    activity_confidence = _activity_confidence(answer)
    confidence_too_high = _confidence_too_high(
        activity_confidence,
        question.max_confidence_for_activity,
    )
    issues: list[str] = []

    if not answer.strip():
        issues.append("ask answer is empty")
    if not expected_date_match:
        expected = ", ".join(question.expected_dates)
        issues.append(f"expected date mismatch: expected={expected}, parsed={parsed_date or 'none'}")
    if question.require_events and not result.events:
        issues.append("events are required but not found")
    if question.min_events is not None and len(result.events) < question.min_events:
        issues.append(f"events below min_events: {len(result.events)} < {question.min_events}")
    if missing_keywords:
        issues.append("missing expected keywords: " + ", ".join(missing_keywords))
    if missing_evidence_types:
        issues.append("missing expected evidence types: " + ", ".join(missing_evidence_types))
    if forbidden_found:
        issues.append("forbidden claims found: " + ", ".join(forbidden_found))
    if unsupported_claims:
        issues.append("unsupported claim categories: " + ", ".join(unsupported_claims))
    if confidence_too_high:
        issues.append(
            "activity confidence too high: "
            f"{activity_confidence or 'unknown'} > {question.max_confidence_for_activity}"
        )
    if len(result.events) + len(result.line_messages) + len(result.media_items) > 0:
        for section in ANSWER_REQUIRED_SECTIONS:
            if section not in answer:
                issues.append(f"missing answer section: {section}")

    return {
        "id": question.id,
        "type": question.case_type,
        "status": "pass" if not issues else "fail",
        "question_preview": redact_text(question.question, max_chars=80),
        "passed": not issues,
        "issues": issues,
        "parsed_date": parsed_date,
        "matched_dates": matched_dates,
        "expected_date": question.expected_date,
        "expected_dates": question.expected_dates,
        "expected_date_match": expected_date_match,
        "events_count": len(result.events),
        "event_evidence_count": event_evidence_count,
        "line_message_count": len(result.line_messages),
        "photo_count": len(result.media_items),
        "gps_photo_count": _gps_photo_count(result.media_items),
        "has_evidence_section": "根拠:" in answer,
        "has_confidence_section": "信頼度:" in answer,
        "matched_expected_keywords": matched_keywords,
        "missing_expected_keywords": missing_keywords,
        "missing_expected_evidence_types": missing_evidence_types,
        "forbidden_claims_found": forbidden_found,
        "unsupported_claims": unsupported_claims,
        "activity_confidence": activity_confidence,
        "max_confidence_for_activity": question.max_confidence_for_activity,
        "answer_preview": redact_text(answer, max_chars=240),
    }


def _evaluate_keyword_search_case(repository, question: PrivateEvalQuestion) -> dict[str, Any]:
    query = question.query or question.question
    report = local_text_search(
        repository,
        LocalSearchOptions(
            query=query,
            limit=10,
            intent=question.intent,  # type: ignore[arg-type]
        ),
    )
    results = list(report.get("results") or [])
    result_dates = [str(row.get("date")) for row in results if row.get("date")]
    matched_dates = [date for date in question.expected_dates if date in result_dates]
    top_dates = result_dates[: max(len(question.expected_top_dates), 1)]
    evidence_text = _search_report_text(report)
    issues: list[str] = []
    missing_evidence: list[str] = []

    if question.expected_dates and not matched_dates:
        issues.append("expected search dates missing: " + ", ".join(question.expected_dates))
    if question.expected_top_dates:
        missing_top = [date for date in question.expected_top_dates if date not in top_dates]
        if missing_top:
            issues.append("expected top dates not in top results: " + ", ".join(missing_top))
    if question.expected_max_rank is not None and question.expected_dates:
        ranked_dates = result_dates[: question.expected_max_rank]
        if not any(date in ranked_dates for date in question.expected_dates):
            issues.append(f"expected dates not within rank {question.expected_max_rank}: " + ", ".join(question.expected_dates))
    for date, expected in question.expected_classification.items():
        row = next((item for item in results if item.get("date") == date), None)
        if row is None:
            issues.append(f"expected classification date missing: {date}")
        elif row.get("classification") != expected:
            issues.append(f"classification mismatch for {date}: {row.get('classification')} != {expected}")
    if question.expected_evidence_types:
        evidence_types = {
            str(evidence_type)
            for row in results
            for evidence_type in row.get("evidence_types", [])
        }
        missing_evidence = [item for item in question.expected_evidence_types if item not in evidence_types]
        if missing_evidence:
            issues.append("missing expected evidence types: " + ", ".join(missing_evidence))
    missing_keywords = [keyword for keyword in question.expected_evidence_keywords if keyword not in evidence_text]
    if missing_keywords:
        issues.append("missing expected evidence keywords: " + ", ".join(missing_keywords))
    if question.expected_top_contains_any and results:
        first_text = _search_result_text(results[0])
        if not any(keyword in first_text for keyword in question.expected_top_contains_any):
            issues.append("top result missing any expected text: " + ", ".join(question.expected_top_contains_any))
    if question.expected_min_score is not None:
        top_score = float(results[0].get("ranking_score") or results[0].get("score") or 0.0) if results else 0.0
        if top_score < question.expected_min_score:
            issues.append(f"top score below expected_min_score: {top_score} < {question.expected_min_score}")
    if results:
        first_text = _search_result_text(results[0])
        downranked_found = [phrase for phrase in question.should_downrank_phrases if phrase in first_text]
        if downranked_found and results[0].get("classification") == "actual_or_likely_action":
            issues.append("downrank phrases appeared in top actual result: " + ", ".join(downranked_found))
    forbidden_found = _forbidden_claims_found(evidence_text, question.forbidden_claims)
    if forbidden_found:
        issues.append("forbidden claims found: " + ", ".join(forbidden_found))

    return {
        "id": question.id,
        "type": question.case_type,
        "status": "pass" if not issues else "fail",
        "question_preview": redact_text(question.question, max_chars=80),
        "passed": not issues,
        "issues": issues,
        "parsed_date": None,
        "matched_dates": matched_dates,
        "expected_date": question.expected_date,
        "expected_dates": question.expected_dates,
        "expected_date_match": not question.expected_dates or bool(matched_dates),
        "events_count": sum(int(row.get("event_count") or 0) for row in results),
        "event_evidence_count": sum(
            int(event.get("event_evidence_count") or 0)
            for row in results
            for event in row.get("events", [])
        ),
        "line_message_count": sum(int(row.get("line_match_count") or 0) for row in results),
        "photo_count": sum(int(row.get("same_day_photo_count", row.get("media_match_count")) or 0) for row in results),
        "gps_photo_count": sum(int(row.get("same_day_gps_photo_count") or 0) for row in results),
        "has_evidence_section": True,
        "has_confidence_section": True,
        "matched_expected_keywords": [
            keyword for keyword in question.expected_evidence_keywords if keyword in evidence_text
        ],
        "missing_expected_keywords": missing_keywords,
        "missing_expected_evidence_types": missing_evidence if question.expected_evidence_types else [],
        "forbidden_claims_found": forbidden_found,
        "unsupported_claims": [],
        "activity_confidence": None,
        "max_confidence_for_activity": question.max_confidence_for_activity,
        "answer_preview": redact_text(json.dumps(report.get("results", [])[:3], ensure_ascii=False), max_chars=240),
        "search_intent": report.get("intent"),
        "top_dates": top_dates,
        "classifications": {str(row.get("date")): row.get("classification") for row in results},
    }


def _evaluate_query_intent_case(question: PrivateEvalQuestion) -> dict[str, Any]:
    result = classify_query_intent(question.question)
    issues: list[str] = []
    if question.expected_intent and not _intent_matches(result.intent, question.expected_intent):
        issues.append(f"intent mismatch: {result.intent} != {question.expected_intent}")
    missing_entities = _missing_expected_entities(result.entities, question.expected_entities)
    if missing_entities:
        issues.append("entity mismatch: " + ", ".join(missing_entities))
    if result.confidence < 0.35:
        issues.append(f"intent confidence too low: {result.confidence}")

    return _generic_case(
        question,
        status="pass" if not issues else "fail",
        issues=issues,
        answer_preview=redact_text(json.dumps(result.to_dict(), ensure_ascii=False), max_chars=240),
        extra={
            "intent": result.intent,
            "intent_confidence": result.confidence,
            "entities": result.entities,
            "routing": result.routing_hint,
            "expected_intent": question.expected_intent,
        },
    )


def _evaluate_routed_qa_case(repository, question: PrivateEvalQuestion) -> dict[str, Any]:
    routed = route_query(repository, question.question, limit=10)
    results = routed.results
    result_dates = _dates_from_routed_results(results)
    evidence_text = routed.answer + "\n" + json.dumps(_compact_results(results), ensure_ascii=False)
    issues: list[str] = []
    if question.expected_intent and not _intent_matches(routed.intent, question.expected_intent):
        issues.append(f"intent mismatch: {routed.intent} != {question.expected_intent}")
    if question.expected_top_dates:
        top_dates = result_dates[: max(len(question.expected_top_dates), 1)]
        missing_top = [date for date in question.expected_top_dates if date not in top_dates]
        if missing_top:
            issues.append("expected top dates not in routed results: " + ", ".join(missing_top))
    for date, expected in question.expected_classification.items():
        row = next((item for item in results if item.get("date") == date), None)
        if row is None:
            issues.append(f"expected classification date missing: {date}")
        elif row.get("classification") != expected:
            issues.append(f"classification mismatch for {date}: {row.get('classification')} != {expected}")
    forbidden_found = _forbidden_claims_found(evidence_text, question.forbidden_claims)
    if forbidden_found:
        issues.append("forbidden claims found: " + ", ".join(forbidden_found))

    return _generic_case(
        question,
        status="pass" if not issues else "fail",
        issues=issues,
        matched_dates=[date for date in question.expected_top_dates if date in result_dates[:5]],
        forbidden_claims_found=forbidden_found,
        answer_preview=redact_text(routed.answer, max_chars=240),
        extra={
            "intent": routed.intent,
            "intent_confidence": routed.intent_confidence,
            "routing": routed.routing,
            "top_dates": result_dates[:5],
            "classifications": {str(row.get("date")): row.get("classification") for row in results if row.get("date")},
        },
    )


def _evaluate_time_range_summary_case(repository, question: PrivateEvalQuestion) -> dict[str, Any]:
    routed = route_query(repository, question.question, limit=10)
    report = routed.results[0] if routed.results and isinstance(routed.results[0], dict) else {}
    representative_days = list(report.get("representative_days") or [])
    result_dates = [str(row.get("date")) for row in representative_days if row.get("date")]
    evidence_types = _summary_evidence_types(report)
    issues: list[str] = []
    if routed.routing not in {"monthly-summary", "time-range-summary"}:
        issues.append(f"unexpected routing: {routed.routing}")
    if question.case_type == "monthly_summary" and routed.intent != "monthly_summary":
        issues.append(f"intent mismatch: {routed.intent} != monthly_summary")
    events_count = int(report.get("events_count") or 0)
    if question.min_events is not None and events_count < question.min_events:
        issues.append(f"events below expected_min_events: {events_count} < {question.min_events}")
    if question.expected_max_events is not None and events_count > question.expected_max_events:
        issues.append(f"events above expected_max_events: {events_count} > {question.expected_max_events}")
    missing_evidence = [item for item in question.expected_evidence_types if item not in evidence_types]
    if missing_evidence:
        issues.append("missing expected evidence types: " + ", ".join(missing_evidence))
    expected_dates = question.expected_top_dates or question.expected_dates
    if expected_dates:
        top_dates = result_dates[: max(len(expected_dates), 1)]
        missing_dates = [date for date in expected_dates if date not in top_dates]
        if missing_dates:
            issues.append("expected top dates not in summary results: " + ", ".join(missing_dates))
    forbidden_found = _forbidden_claims_found(routed.answer, question.forbidden_claims)
    if forbidden_found:
        issues.append("forbidden claims found: " + ", ".join(forbidden_found))

    media = report.get("media") or {}
    return _generic_case(
        question,
        status="pass" if not issues else "fail",
        issues=issues,
        matched_dates=[date for date in expected_dates if date in result_dates[:5]],
        events_count=events_count,
        event_evidence_count=int(report.get("event_evidence_count") or 0),
        line_message_count=int(report.get("line_messages_count") or 0),
        photo_count=int(media.get("photos") or 0),
        gps_photo_count=int(media.get("gps_photos") or 0),
        forbidden_claims_found=forbidden_found,
        answer_preview=redact_text(routed.answer, max_chars=240),
        extra={
            "intent": routed.intent,
            "intent_confidence": routed.intent_confidence,
            "routing": routed.routing,
            "top_dates": result_dates[:5],
            "evidence_types": sorted(evidence_types),
            "category_counts": report.get("category_counts") or {},
            "confidence_distribution": report.get("confidence_distribution") or {},
            "missing_expected_evidence_types": missing_evidence,
        },
    )


def _evaluate_call_search_case(repository, question: PrivateEvalQuestion) -> dict[str, Any]:
    filters = dict(question.filters)
    statuses = _statuses_from_filters(filters)
    min_duration = _none_or_int(filters.get("min_duration_sec"))
    exact_date = _none_or_str(filters.get("date"))
    report = search_calls(
        repository,
        statuses=statuses,
        min_duration_sec=min_duration,
        start_date=exact_date or _none_or_str(filters.get("from") or filters.get("date_from")),
        end_date=exact_date or _none_or_str(filters.get("to") or filters.get("date_to")),
        limit=100,
    )
    results = list(report.get("results") or [])
    issues: list[str] = []
    if question.expected_min_results is not None and len(results) < question.expected_min_results:
        issues.append(f"results below expected_min_results: {len(results)} < {question.expected_min_results}")
    if question.expected_status and any(row.get("call_status") != question.expected_status for row in results):
        issues.append(f"unexpected status found; expected only {question.expected_status}")
    forbidden_statuses = set(question.should_not_include_status)
    found_forbidden = sorted({str(row.get("call_status")) for row in results if row.get("call_status") in forbidden_statuses})
    if found_forbidden:
        issues.append("forbidden statuses found: " + ", ".join(found_forbidden))
    if question.expected_date_any and not any(row.get("date") in question.expected_date_any for row in results):
        issues.append("expected_date_any not found: " + ", ".join(question.expected_date_any))
    if question.expected_sender_any and not any(row.get("sender") in question.expected_sender_any for row in results):
        issues.append("expected_sender_any not found: " + ", ".join(question.expected_sender_any))

    return _generic_case(
        question,
        status="pass" if not issues else "fail",
        issues=issues,
        matched_dates=[str(row.get("date")) for row in results if row.get("date") in question.expected_date_any],
        answer_preview=redact_text(json.dumps(results[:5], ensure_ascii=False), max_chars=240),
        extra={
            "results_count": len(results),
            "statuses": [row.get("call_status") for row in results[:10]],
            "top_dates": [row.get("date") for row in results[:5]],
        },
    )


def _evaluate_event_quality_case(repository, question: PrivateEvalQuestion) -> dict[str, Any]:
    target_date = question.date or question.expected_date or (question.expected_dates[0] if question.expected_dates else None)
    if not target_date:
        return _skip_case(question, "event_quality requires date")
    events = repository.list_events(start_date=target_date, end_date=target_date, limit=10_000, include_hidden=False)
    issues: list[str] = []
    if question.min_events is not None and len(events) < question.min_events:
        issues.append(f"events below expected_min_events: {len(events)} < {question.min_events}")
    if question.expected_max_events is not None and len(events) > question.expected_max_events:
        issues.append(f"events above expected_max_events: {len(events)} > {question.expected_max_events}")
    evidence_types = _event_evidence_types(repository, events)
    missing_evidence = [item for item in question.expected_evidence_types if item not in evidence_types]
    if missing_evidence:
        issues.append("missing expected evidence types: " + ", ".join(missing_evidence))
    line_only_low = [
        event for event in events
        if int(event.get("line_evidence_count") or 0) > 0
        and int(event.get("photo_evidence_count") or 0) == 0
        and float(event.get("confidence") or 0.0) < 0.5
    ]
    if question.max_line_only_low_value_events is not None and len(line_only_low) > question.max_line_only_low_value_events:
        issues.append(
            f"line-only low-value events too many: {len(line_only_low)} > {question.max_line_only_low_value_events}"
        )
    photo_and_line = [
        event for event in events
        if int(event.get("line_evidence_count") or 0) > 0 and int(event.get("photo_evidence_count") or 0) > 0
    ]
    if question.min_photo_and_line_events is not None and len(photo_and_line) < question.min_photo_and_line_events:
        issues.append(f"photo+line events below minimum: {len(photo_and_line)} < {question.min_photo_and_line_events}")
    orphan_count = _orphan_evidence_count(repository, events)
    if question.no_orphan_evidence and orphan_count:
        issues.append(f"orphan evidence found: {orphan_count}")
    vlm_only_high = 0
    for event in events:
        evidence = repository.list_event_evidence(str(event["id"]))
        types = {str(row.get("evidence_type") or "") for row in evidence}
        if "vlm" in types and "line" not in types and "ocr" not in types and float(event.get("confidence") or 0.0) >= 0.8:
            vlm_only_high += 1
    if question.max_vlm_only_high_confidence_events is not None and vlm_only_high > question.max_vlm_only_high_confidence_events:
        issues.append(
            f"VLM-only high confidence events too many: {vlm_only_high} > {question.max_vlm_only_high_confidence_events}"
        )

    return _generic_case(
        question,
        status="pass" if not issues else "fail",
        issues=issues,
        matched_dates=[target_date],
        events_count=len(events),
        event_evidence_count=sum(len(repository.list_event_evidence(str(event["id"]))) for event in events),
        answer_preview=redact_text(json.dumps(_compact_results(events[:5]), ensure_ascii=False), max_chars=240),
        extra={
            "date": target_date,
            "evidence_types": sorted(evidence_types),
            "line_only_low_value_events": len(line_only_low),
            "photo_and_line_events": len(photo_and_line),
            "orphan_evidence_count": orphan_count,
            "vlm_only_high_confidence_events": vlm_only_high,
        },
    )


def _evaluate_event_rebuild_quality_case(repository, question: PrivateEvalQuestion) -> dict[str, Any]:
    target_date = question.date or question.expected_date or (question.expected_dates[0] if question.expected_dates else None)
    if not target_date:
        return _skip_case(question, "event_rebuild_quality requires date")
    events = repository.list_events(start_date=target_date, end_date=target_date, limit=10_000, include_hidden=True)
    evidence_types = _event_evidence_types(repository, events)
    issues: list[str] = []
    if question.min_events is not None and len(events) < question.min_events:
        issues.append(f"events below expected_min_events: {len(events)} < {question.min_events}")
    expected_any = question.expected_evidence_types_any or question.expected_evidence_types
    if expected_any and not any(item in evidence_types for item in expected_any):
        issues.append("none of expected evidence types found: " + ", ".join(expected_any))
    vlm_only_high = 0
    overclaim_phrases = ("確実に食事した", "確実に新宿に行った", "確実に行った")
    text_blob = "\n".join(str(event.get("title") or "") + "\n" + str(event.get("summary") or "") for event in events)
    if question.expected_no_overclaim and any(phrase in text_blob for phrase in overclaim_phrases):
        issues.append("overclaim phrase found")
    for event in events:
        evidence = repository.list_event_evidence(str(event["id"]))
        types = {str(row.get("evidence_type") or "") for row in evidence}
        if "vlm" in types and "line" not in types and "ocr" not in types and float(event.get("confidence") or 0.0) >= 0.8:
            vlm_only_high += 1
    if question.max_vlm_only_high_confidence_events is not None and vlm_only_high > question.max_vlm_only_high_confidence_events:
        issues.append(
            f"VLM-only high confidence events too many: {vlm_only_high} > {question.max_vlm_only_high_confidence_events}"
        )
    forbidden_found = _forbidden_claims_found(text_blob, question.forbidden_claims)
    if forbidden_found:
        issues.append("forbidden claims found: " + ", ".join(forbidden_found))
    return _generic_case(
        question,
        status="pass" if not issues else "fail",
        issues=issues,
        matched_dates=[target_date],
        events_count=len(events),
        event_evidence_count=sum(len(repository.list_event_evidence(str(event["id"]))) for event in events),
        forbidden_claims_found=forbidden_found,
        answer_preview=redact_text(text_blob, max_chars=240),
        extra={
            "date": target_date,
            "evidence_types": sorted(evidence_types),
            "vlm_only_high_confidence_events": vlm_only_high,
        },
    )


def _evaluate_place_assignment_case(repository, question: PrivateEvalQuestion) -> dict[str, Any]:
    target_date = question.date or question.expected_date or (question.expected_dates[0] if question.expected_dates else None)
    if not target_date:
        return _skip_case(question, "place_assignment requires date")
    events = repository.list_events(start_date=target_date, end_date=target_date, limit=10_000, include_hidden=True)
    location_names = [str(event.get("location_name") or "") for event in events if event.get("location_name")]
    issues: list[str] = []
    if question.expected_any_location_name and not any(
        expected in location for expected in question.expected_any_location_name for location in location_names
    ):
        issues.append("expected location name not found: " + ", ".join(question.expected_any_location_name))
    answer_text = "\n".join(location_names)
    if question.forbidden_exact_gps_in_answer and _contains_precise_gps(answer_text):
        issues.append("precise GPS-like coordinate found")

    return _generic_case(
        question,
        status="pass" if not issues else "fail",
        issues=issues,
        matched_dates=[target_date],
        events_count=len(events),
        answer_preview=redact_text(answer_text, max_chars=240),
        extra={"date": target_date, "location_names": location_names[:20]},
    )


def _evaluate_event_override_case(repository, question: PrivateEvalQuestion) -> dict[str, Any]:
    event_id = question.event_id or question.question
    override = repository.get_event_override(event_id)
    issues: list[str] = []
    if override is None:
        has_expectations = any(
            value is not None
            for value in (
                question.expected_hidden,
                question.expected_pinned,
                question.expected_verified,
                question.title_override_contains,
                question.summary_override_contains,
            )
        )
        if not has_expectations:
            return _skip_case(question, f"event override not found: {event_id}")
        issues.append(f"event override not found: {event_id}")
        override = {}
    for key, expected in (
        ("is_hidden", question.expected_hidden),
        ("is_pinned", question.expected_pinned),
        ("is_verified", question.expected_verified),
    ):
        if expected is not None and int(bool(override.get(key))) != int(bool(expected)):
            issues.append(f"{key} mismatch: {override.get(key)} != {int(bool(expected))}")
    if question.title_override_contains and question.title_override_contains not in str(override.get("title_override") or ""):
        issues.append("title_override does not contain expected text")
    if question.summary_override_contains and question.summary_override_contains not in str(override.get("summary_override") or ""):
        issues.append("summary_override does not contain expected text")

    return _generic_case(
        question,
        status="pass" if not issues else "fail",
        issues=issues,
        answer_preview=redact_text(json.dumps(override, ensure_ascii=False), max_chars=240),
        extra={"event_id": event_id, "override": override},
    )


def _evaluate_vlm_review_case(repository, question: PrivateEvalQuestion) -> dict[str, Any]:
    media_id = question.media_id or question.question
    base = repository.get_media_vlm(media_id) or {"media_id": media_id}
    override = repository.get_media_vlm_override(media_id)
    effective = apply_vlm_override_to_result(base)
    issues: list[str] = []
    if override is None:
        has_expectations = any(
            value is not None
            for value in (
                question.expected_hidden,
                question.expected_verified,
                question.expected_searchable,
                question.expected_event_usable,
            )
        )
        if not has_expectations:
            return _skip_case(question, f"VLM override not found: {media_id}")
        issues.append(f"VLM override not found: {media_id}")
    for key, expected in (
        ("is_hidden", question.expected_hidden),
        ("is_verified", question.expected_verified),
        ("is_searchable", question.expected_searchable),
        ("is_event_usable", question.expected_event_usable),
    ):
        if expected is not None and int(bool(effective.get(key))) != int(bool(expected)):
            issues.append(f"{key} mismatch: {effective.get(key)} != {int(bool(expected))}")

    return _generic_case(
        question,
        status="pass" if not issues else "fail",
        issues=issues,
        answer_preview=redact_text(json.dumps(effective, ensure_ascii=False), max_chars=240),
        extra={"media_id": media_id, "review_status": effective.get("review_status")},
    )


def _evaluate_ocr_quality_case(repository, question: PrivateEvalQuestion) -> dict[str, Any]:
    target_date = question.date or question.expected_date or (question.expected_dates[0] if question.expected_dates else None)
    if not target_date:
        return _skip_case(question, "ocr_quality requires date")
    rows = repository.list_media_ocr(start_date=target_date, end_date=target_date, limit=100_000)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    success_count = status_counts.get("success", 0)
    processed_count = len(rows)
    engine_unavailable_count = status_counts.get("engine_unavailable", 0)
    issues: list[str] = []
    if question.expected_min_ocr_success is not None and success_count < question.expected_min_ocr_success:
        issues.append(
            f"OCR success below expected_min_ocr_success: {success_count} < {question.expected_min_ocr_success}"
        )
    if question.expected_min_ocr_processed is not None and processed_count < question.expected_min_ocr_processed:
        issues.append(
            f"OCR processed below expected_min_ocr_processed: {processed_count} < {question.expected_min_ocr_processed}"
        )
    missing_evidence = []
    if "ocr" in {item.lower() for item in question.expected_evidence_types} and success_count == 0:
        missing_evidence.append("ocr")
        issues.append("missing expected evidence types: ocr")

    return _generic_case(
        question,
        status="pass" if not issues else "fail",
        issues=issues,
        matched_dates=[target_date],
        photo_count=len(rows),
        answer_preview=redact_text(json.dumps(status_counts, ensure_ascii=False), max_chars=240),
        extra={
            "date": target_date,
            "ocr_status_counts": dict(sorted(status_counts.items())),
            "ocr_processed_count": processed_count,
            "ocr_success_count": success_count,
            "expected_min_ocr_success": question.expected_min_ocr_success,
            "expected_min_ocr_processed": question.expected_min_ocr_processed,
            "allow_engine_unavailable": question.allow_engine_unavailable,
            "missing_expected_evidence_types": missing_evidence,
        },
    )


def _evaluate_ocr_search_case(repository, question: PrivateEvalQuestion) -> dict[str, Any]:
    query = question.query or question.question
    rows = repository.list_media_ocr(
        start_date=question.date_from,
        end_date=question.date_to or question.date_from,
        statuses=["success"],
        keyword=query,
        limit=100,
    )
    issues: list[str] = []
    if question.expected_min_results is not None and len(rows) < question.expected_min_results:
        issues.append(f"OCR search results below expected_min_results: {len(rows)} < {question.expected_min_results}")
    matched_dates = sorted(
        {
            str(row.get("captured_at") or row.get("fallback_captured_at") or "")[:10]
            for row in rows
            if str(row.get("captured_at") or row.get("fallback_captured_at") or "")[:10]
        }
    )
    for expected_date in question.expected_top_dates:
        if expected_date not in matched_dates[:5]:
            issues.append(f"expected date not in OCR search top dates: {expected_date}")
    preview = [
        {
            "media_id": row.get("media_id"),
            "date": str(row.get("captured_at") or row.get("fallback_captured_at") or "")[:10],
            "text": redact_text(row.get("ocr_text_redacted") or row.get("ocr_text"), max_chars=80),
        }
        for row in rows[:5]
    ]
    return _generic_case(
        question,
        status="pass" if not issues else "fail",
        issues=issues,
        matched_dates=matched_dates[:5],
        answer_preview=redact_text(json.dumps(preview, ensure_ascii=False), max_chars=240),
        extra={"query": query, "result_count": len(rows), "preview": preview},
    )


def _evaluate_vlm_quality_case(repository, question: PrivateEvalQuestion) -> dict[str, Any]:
    target_date = question.date or question.expected_date or (question.expected_dates[0] if question.expected_dates else None)
    start_date = target_date or question.date_from
    end_date = target_date or question.date_to or start_date
    if not start_date:
        return _skip_case(question, "vlm_quality requires date or date_from/date_to")
    rows = repository.list_media_vlm(start_date=start_date, end_date=end_date, limit=100_000)
    status_counts: dict[str, int] = {}
    searchable_text_parts: list[str] = []
    safety_flag_counts: dict[str, int] = {}
    success_engines: dict[str, int] = {}
    empty_success_caption_count = 0
    for row in rows:
        status = str(row.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        searchable_text_parts.extend(
            [
                str(row.get("caption") or ""),
                str(row.get("short_caption") or ""),
                str(row.get("scene_tags_json") or ""),
                str(row.get("object_tags_json") or ""),
                str(row.get("activity_tags_json") or ""),
                str(row.get("food_cues_json") or ""),
                str(row.get("location_cues_json") or ""),
                str(row.get("text_cues_json") or ""),
            ]
        )
        if status == "success":
            engine = str(row.get("vlm_engine") or "unknown")
            success_engines[engine] = success_engines.get(engine, 0) + 1
            if not str(row.get("caption") or "").strip():
                empty_success_caption_count += 1
        for flag in _json_string_list(row.get("safety_flags_json")):
            safety_flag_counts[flag] = safety_flag_counts.get(flag, 0) + 1
    caption_text = "\n".join(searchable_text_parts)
    success_count = status_counts.get("success", 0)
    issues: list[str] = []
    if question.expected_min_vlm_success is not None and success_count < question.expected_min_vlm_success:
        issues.append(
            f"VLM success below expected_min_vlm_success: {success_count} < {question.expected_min_vlm_success}"
        )
    if question.expected_engine:
        if success_engines and question.expected_engine not in success_engines:
            issues.append(f"expected VLM engine not found: {question.expected_engine}")
        unexpected_engines = sorted(engine for engine in success_engines if engine != question.expected_engine)
        if unexpected_engines:
            issues.append("unexpected success VLM engines found: " + ", ".join(unexpected_engines))
    if empty_success_caption_count:
        issues.append(f"success VLM rows with empty caption: {empty_success_caption_count}")
    failed_count = status_counts.get("failed", 0)
    engine_unavailable_count = status_counts.get("engine_unavailable", 0)
    if question.max_failed_vlm_rows is not None:
        if failed_count > question.max_failed_vlm_rows:
            issues.append(f"failed VLM rows above max_failed_vlm_rows: {failed_count} > {question.max_failed_vlm_rows}")
    elif failed_count:
        issues.append(f"failed VLM rows found: {failed_count}")
    if engine_unavailable_count:
        issues.append(f"engine_unavailable VLM rows found: {engine_unavailable_count}")
    forbidden_found = [term for term in question.forbidden_terms if term in caption_text]
    if forbidden_found:
        issues.append("forbidden VLM terms found: " + ", ".join(forbidden_found))
    disallowed_flags: list[str] = []
    if question.allowed_safety_flags:
        allowed = set(question.allowed_safety_flags)
        disallowed_flags = sorted(flag for flag in safety_flag_counts if flag not in allowed)
        if disallowed_flags:
            issues.append("disallowed VLM safety flags found: " + ", ".join(disallowed_flags))

    return _generic_case(
        question,
        status="pass" if not issues else "fail",
        issues=issues,
        matched_dates=[value for value in (target_date, start_date) if value],
        photo_count=len(rows),
        forbidden_claims_found=forbidden_found,
        answer_preview=redact_text(json.dumps(status_counts, ensure_ascii=False), max_chars=240),
        extra={
            "date": target_date,
            "date_from": start_date,
            "date_to": end_date,
            "vlm_status_counts": dict(sorted(status_counts.items())),
            "vlm_success_count": success_count,
            "expected_min_vlm_success": question.expected_min_vlm_success,
            "max_failed_vlm_rows": question.max_failed_vlm_rows,
            "failed_vlm_rows": failed_count,
            "success_engines": dict(sorted(success_engines.items())),
            "expected_engine": question.expected_engine,
            "safety_flag_counts": dict(sorted(safety_flag_counts.items())),
            "allowed_safety_flags": question.allowed_safety_flags,
            "disallowed_safety_flags": disallowed_flags,
            "empty_success_caption_count": empty_success_caption_count,
        },
    )


def _evaluate_vlm_safety_case(question: PrivateEvalQuestion) -> dict[str, Any]:
    input_text = question.input_text or question.question
    report = safety_check_text(input_text)
    sanitized = str(report.get("sanitized") or "")
    flags = set(str(flag) for flag in report.get("safety_flags", []))
    issues: list[str] = []
    missing_flags = [flag for flag in question.expected_flags if flag not in flags]
    if missing_flags:
        issues.append("missing expected safety flags: " + ", ".join(missing_flags))
    forbidden_found = [term for term in question.forbidden_claims if term in sanitized]
    if forbidden_found:
        issues.append("forbidden phrases still present: " + ", ".join(forbidden_found))
    return _generic_case(
        question,
        status="pass" if not issues else "fail",
        issues=issues,
        forbidden_claims_found=forbidden_found,
        answer_preview=redact_text(sanitized, max_chars=240),
        extra={
            "safety_flags": sorted(flags),
            "violations": report.get("violations", []),
            "sanitized": sanitized,
        },
    )


def _evaluate_vlm_prompt_case(question: PrivateEvalQuestion) -> dict[str, Any]:
    template_name = question.template or question.question
    try:
        template = get_vlm_prompt_template(template_name)
    except ValueError as exc:
        return _generic_case(
            question,
            status="fail",
            issues=[str(exc)],
            answer_preview="",
            extra={"template": template_name},
        )
    prompt = template.prompt
    issues = [text for text in question.expected_contains if text not in prompt]
    return _generic_case(
        question,
        status="pass" if not issues else "fail",
        issues=[f"missing expected prompt text: {text}" for text in issues],
        answer_preview=redact_text(prompt, max_chars=240),
        extra={"template": template.name},
    )


def _evaluate_image_search_case(repository, question: PrivateEvalQuestion) -> dict[str, Any]:
    query = question.query or question.question
    report = image_search(repository, ImageSearchOptions(query=query, limit=10))
    results = list(report.get("results") or [])
    result_dates = [str(row.get("date")) for row in results if row.get("date")]
    result_media_ids = [str(row.get("media_id")) for row in results if row.get("media_id")]
    issues: list[str] = []
    if question.expected_min_results is not None and len(results) < question.expected_min_results:
        issues.append(f"results below expected_min_results: {len(results)} < {question.expected_min_results}")
    if question.expected_top_dates_any and not any(date in result_dates[:5] for date in question.expected_top_dates_any):
        issues.append("none of expected_top_dates_any found in top5: " + ", ".join(question.expected_top_dates_any))
    expected_dates = question.expected_top_dates or question.expected_dates
    if expected_dates and not any(date in result_dates[:5] for date in expected_dates):
        issues.append("expected image search dates missing: " + ", ".join(expected_dates))
    expected_types = question.expected_evidence_types_any or question.expected_evidence_types
    if expected_types:
        evidence_types = {str(item) for row in results for item in row.get("evidence_types", [])}
        if question.expected_evidence_types_any:
            if not any(item in evidence_types for item in expected_types):
                issues.append("none of expected evidence types found: " + ", ".join(expected_types))
        else:
            missing = [item for item in expected_types if item not in evidence_types]
            if missing:
                issues.append("missing expected evidence types: " + ", ".join(missing))
    missing_media_ids = [media_id for media_id in question.expected_media_ids if media_id not in result_media_ids]
    if missing_media_ids:
        issues.append("expected media ids missing: " + ", ".join(missing_media_ids))
    excluded_found = [media_id for media_id in question.should_exclude_media_ids if media_id in result_media_ids]
    if excluded_found:
        issues.append("excluded media ids appeared: " + ", ".join(excluded_found))
    answer_text = json.dumps(results[:5], ensure_ascii=False)
    forbidden = _forbidden_claims_found(answer_text, question.forbidden_claims)
    if forbidden:
        issues.append("forbidden phrase found: " + ", ".join(forbidden))
    return _generic_case(
        question,
        status="pass" if not issues else "fail",
        issues=issues,
        matched_dates=[date for date in [*expected_dates, *question.expected_top_dates_any] if date in result_dates[:5]],
        photo_count=len(results),
        forbidden_claims_found=forbidden,
        answer_preview=redact_text(answer_text, max_chars=240),
        extra={
            "top_dates": result_dates[:5],
            "expected_top_dates_any": question.expected_top_dates_any,
            "results_count": len(results),
            "top_media_ids": result_media_ids[:5],
        },
    )


def _evaluate_multimodal_search_case(repository, question: PrivateEvalQuestion) -> dict[str, Any]:
    query = question.query or question.question
    report = multimodal_search(
        repository,
        MultimodalSearchOptions(
            query=query,
            limit=10,
            backend="hybrid",
        ),
    )
    results = list(report.get("results") or [])
    result_dates = [str(row.get("date")) for row in results if row.get("date")]
    issues: list[str] = []
    if question.expected_min_results is not None and len(results) < question.expected_min_results:
        issues.append(f"results below expected_min_results: {len(results)} < {question.expected_min_results}")
    if question.expected_top_dates_any and not any(date in result_dates[:5] for date in question.expected_top_dates_any):
        issues.append("none of expected_top_dates_any found in top5: " + ", ".join(question.expected_top_dates_any))
    expected_dates = question.expected_top_dates or question.expected_dates
    if expected_dates and not any(date in result_dates[:5] for date in expected_dates):
        issues.append("expected multimodal dates missing: " + ", ".join(expected_dates))
    expected_types = question.expected_evidence_types_any or question.expected_evidence_types
    if expected_types:
        evidence_types = {str(item) for row in results for item in row.get("evidence_types", [])}
        if not any(item in evidence_types for item in expected_types):
            issues.append("none of expected evidence types found: " + ", ".join(expected_types))
    if question.expected_classification_any:
        strengths = {str(row.get("evidence_strength") or "") for row in results}
        labels = {str(item) for row in results for item in row.get("evidence_types", [])}
        combined = strengths | labels | {str(row.get("classification") or "") for row in results}
        if not any(item in combined for item in question.expected_classification_any):
            issues.append("expected classification/evidence label missing: " + ", ".join(question.expected_classification_any))
    if question.max_embedding_only_rank is not None:
        for index, row in enumerate(results, start=1):
            types = set(row.get("evidence_types") or [])
            if types.issubset({"photo", "embedding"}) and index <= question.max_embedding_only_rank:
                issues.append(f"embedding-only result ranked too high: rank {index}")
                break
    if question.max_vlm_only_confidence:
        for index, row in enumerate(results, start=1):
            types = set(row.get("evidence_types") or [])
            if types.issubset({"photo", "vlm"}) and not confidence_at_most(str(row.get("confidence_label") or ""), question.max_vlm_only_confidence):
                issues.append(
                    f"VLM-only confidence too high at rank {index}: "
                    f"{row.get('confidence_label')} > {question.max_vlm_only_confidence}"
                )
                break
    if question.expected_strength_at_least and results:
        top_strength = str(results[0].get("evidence_strength") or "")
        if not strength_at_least(top_strength, question.expected_strength_at_least):
            issues.append(
                f"top evidence_strength below expected: {top_strength} < {question.expected_strength_at_least}"
            )
    answer_text = json.dumps(results[:5], ensure_ascii=False)
    forbidden = _forbidden_claims_found(answer_text, question.forbidden_claims)
    if forbidden:
        issues.append("forbidden phrase found: " + ", ".join(forbidden))
    return _generic_case(
        question,
        status="pass" if not issues else "fail",
        issues=issues,
        matched_dates=[date for date in [*expected_dates, *question.expected_top_dates_any] if date in result_dates[:5]],
        photo_count=len(results),
        forbidden_claims_found=forbidden,
        answer_preview=redact_text(answer_text, max_chars=240),
        extra={
            "top_dates": result_dates[:5],
            "expected_top_dates_any": question.expected_top_dates_any,
            "results_count": len(results),
            "evidence_strengths": [row.get("evidence_strength") for row in results[:5]],
        },
    )


def _generic_case(
    question: PrivateEvalQuestion,
    *,
    status: str,
    issues: list[str],
    matched_dates: list[str] | None = None,
    events_count: int = 0,
    event_evidence_count: int = 0,
    line_message_count: int = 0,
    photo_count: int = 0,
    gps_photo_count: int = 0,
    forbidden_claims_found: list[str] | None = None,
    answer_preview: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "id": question.id,
        "type": question.case_type,
        "status": status,
        "question_preview": redact_text(question.question, max_chars=80),
        "passed": status == "pass",
        "issues": issues,
        "parsed_date": None,
        "matched_dates": matched_dates or [],
        "expected_date": question.expected_date,
        "expected_dates": question.expected_dates,
        "expected_top_dates_any": question.expected_top_dates_any,
        "expected_date_match": not question.expected_dates or bool(matched_dates),
        "events_count": events_count,
        "event_evidence_count": event_evidence_count,
        "line_message_count": line_message_count,
        "photo_count": photo_count,
        "gps_photo_count": gps_photo_count,
        "has_evidence_section": True,
        "has_confidence_section": True,
        "matched_expected_keywords": [],
        "missing_expected_keywords": [],
        "missing_expected_evidence_types": [],
        "forbidden_claims_found": forbidden_claims_found or [],
        "unsupported_claims": [],
        "activity_confidence": None,
        "max_confidence_for_activity": question.max_confidence_for_activity,
        "answer_preview": answer_preview,
    }
    if extra:
        row.update(extra)
    return row


def _missing_expected_entities(actual: dict[str, Any], expected: dict[str, str]) -> list[str]:
    missing: list[str] = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(actual_value, list):
            ok = expected_value in {str(item) for item in actual_value}
        else:
            ok = str(actual_value) == expected_value
        if not ok:
            missing.append(f"{key}={expected_value} (actual={actual_value})")
    return missing


def _intent_matches(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    return {actual, expected} == {"monthly_summary", "time_range_summary"}


def _dates_from_routed_results(results: list[dict[str, Any]]) -> list[str]:
    dates: list[str] = []
    for row in results:
        value = row.get("date") or row.get("start_date")
        if value and str(value) not in dates:
            dates.append(str(value)[:10])
    return dates


def _compact_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rows:
        compact.append(
            {
                key: row.get(key)
                for key in (
                    "id",
                    "date",
                    "start_time",
                    "end_time",
                    "title",
                    "location_name",
                    "classification",
                    "confidence",
                    "call_status",
                    "duration_sec",
                )
                if key in row
            }
        )
    return compact


def _statuses_from_filters(filters: dict[str, Any]) -> list[str] | None:
    statuses: list[str] = []
    for key, status in (
        ("completed", "completed"),
        ("missed", "missed"),
        ("unanswered", "unanswered"),
        ("canceled", "canceled"),
    ):
        if filters.get(key):
            statuses.append(status)
    explicit_status = _none_or_str(filters.get("status"))
    if explicit_status:
        statuses.append(explicit_status)
    return statuses or None


def _event_evidence_types(repository, events: list[dict[str, Any]]) -> set[str]:
    evidence_types: set[str] = set()
    for event in events:
        for evidence in repository.list_event_evidence(str(event["id"])):
            evidence_types.add(str(evidence.get("evidence_type")))
    return evidence_types


def _summary_evidence_types(report: dict[str, Any]) -> set[str]:
    evidence_types = {str(key) for key, value in (report.get("evidence_type_counts") or {}).items() if int(value or 0) > 0}
    media = report.get("media") or {}
    if int(media.get("photos") or 0) > 0:
        evidence_types.add("photo")
    if int(media.get("vlm_success_photos") or 0) > 0:
        evidence_types.add("vlm")
    if int(media.get("ocr_success_photos") or 0) > 0:
        evidence_types.add("ocr")
    if int(report.get("line_messages_count") or 0) > 0:
        evidence_types.add("line")
    if int(report.get("call_events_count") or 0) > 0:
        evidence_types.add("call")
    return evidence_types


def _orphan_evidence_count(repository, events: list[dict[str, Any]]) -> int:
    count = 0
    for event in events:
        for evidence in repository.list_event_evidence(str(event["id"])):
            evidence_type = str(evidence.get("evidence_type") or "")
            evidence_id = str(evidence.get("evidence_id") or "")
            if evidence_type == "line":
                count += 0 if repository.get_embedding_record("line_message", evidence_id) else 1
            elif evidence_type == "photo":
                count += 0 if repository.get_embedding_record("media_item", evidence_id) else 1
            elif evidence_type == "ocr":
                count += 0 if repository.get_media_ocr(evidence_id) else 1
            elif evidence_type == "vlm":
                count += 0 if repository.get_media_vlm(evidence_id) else 1
    return count


def _contains_precise_gps(text: str) -> bool:
    import re

    return bool(re.search(r"\d{2,3}\.\d{5,}", text))


def _skip_case(question: PrivateEvalQuestion, reason: str) -> dict[str, Any]:
    return {
        "id": question.id,
        "type": question.case_type,
        "status": "skip",
        "passed": False,
        "skipped": True,
        "skip_reason": reason,
        "question_preview": redact_text(question.question, max_chars=80),
        "issues": [],
        "parsed_date": None,
        "matched_dates": [],
        "expected_date": question.expected_date,
        "expected_dates": question.expected_dates,
        "expected_date_match": False,
        "events_count": 0,
        "event_evidence_count": 0,
        "line_message_count": 0,
        "photo_count": 0,
        "gps_photo_count": 0,
        "has_evidence_section": False,
        "has_confidence_section": False,
        "matched_expected_keywords": [],
        "missing_expected_keywords": question.expected_keywords,
        "missing_expected_evidence_types": question.expected_evidence_types,
        "forbidden_claims_found": [],
        "unsupported_claims": [],
        "activity_confidence": None,
        "max_confidence_for_activity": question.max_confidence_for_activity,
        "answer_preview": "",
    }


def _by_type_summary(cases: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for case in cases:
        case_type = str(case.get("type") or "unknown")
        bucket = summary.setdefault(case_type, {"total": 0, "passed": 0, "failed": 0, "skipped": 0})
        bucket["total"] += 1
        if case.get("status") == "pass":
            bucket["passed"] += 1
        elif case.get("status") == "fail":
            bucket["failed"] += 1
        elif case.get("status") == "skip":
            bucket["skipped"] += 1
    return summary


def _ranking_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    ranked_cases = [
        case for case in cases
        if case.get("expected_dates") or case.get("expected_top_dates") or case.get("expected_top_dates_any")
    ]
    if not ranked_cases:
        return {"top1_accuracy": None, "expected_date_recall_at_5": None, "evaluated_cases": 0}
    top1_hits = 0
    recall_hits = 0
    for case in ranked_cases:
        expected = list(case.get("expected_top_dates") or case.get("expected_dates") or case.get("expected_top_dates_any") or [])
        top_dates = list(case.get("top_dates") or case.get("matched_dates") or [])
        if expected and top_dates and top_dates[0] in expected:
            top1_hits += 1
        if expected and any(date in top_dates[:5] for date in expected):
            recall_hits += 1
    total = len(ranked_cases)
    return {
        "top1_accuracy": round(top1_hits / total, 3),
        "expected_date_recall_at_5": round(recall_hits / total, 3),
        "evaluated_cases": total,
    }


def _safety_metrics(cases: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "forbidden_phrase_violations": sum(len(case.get("forbidden_claims_found") or []) for case in cases),
        "overclaim_violations": sum(len(case.get("unsupported_claims") or []) for case in cases),
    }


def _compare_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": report.get("summary", {}),
        "ranking_metrics": report.get("ranking_metrics", {}),
        "safety_metrics": report.get("safety_metrics", {}),
    }


def _summary_value(report: dict[str, Any], key: str) -> int:
    return int(report.get("summary", {}).get(key) or 0)


def _metric_delta(before: dict[str, Any], after: dict[str, Any], key: str) -> float | None:
    before_value = before.get("ranking_metrics", {}).get(key)
    after_value = after.get("ranking_metrics", {}).get(key)
    if before_value is None or after_value is None:
        return None
    return round(float(after_value) - float(before_value), 3)


def _safety_delta(before: dict[str, Any], after: dict[str, Any], key: str) -> int:
    return int(after.get("safety_metrics", {}).get(key) or 0) - int(before.get("safety_metrics", {}).get(key) or 0)


def _unsupported_claims(answer: str, result) -> list[str]:
    evidence_text = _evidence_text(result)
    unsupported: list[str] = []
    for category, keywords in CLAIM_SUPPORT_KEYWORDS.items():
        answer_has_claim = any(keyword in answer for keyword in keywords)
        evidence_has_support = any(keyword in evidence_text for keyword in keywords)
        if answer_has_claim and not evidence_has_support:
            unsupported.append(category)
    return unsupported


def _evidence_text(result) -> str:
    parts: list[str] = []
    for event in result.events:
        parts.append(event.get("title") or "")
        parts.append(event.get("summary") or "")
        parts.append(event.get("location_name") or "")
    for message in result.line_messages:
        parts.append(message.get("text") or message.get("message_text") or "")
        parts.append(message.get("message_type") or "")
    for item in result.media_items:
        parts.append(item.get("caption") or "")
        parts.append(item.get("ocr_text") or "")
        parts.append(item.get("file_name") or "")
    return "\n".join(parts)


def _gps_photo_count(media_items: list[dict[str, Any]]) -> int:
    return sum(1 for item in media_items if item.get("gps_lat") is not None and item.get("gps_lon") is not None)


def _activity_confidence(answer: str) -> str | None:
    marker = "- 何をしていたか:"
    for line in answer.splitlines():
        if marker in line:
            return line.split(marker, 1)[1].strip() or None
    return None


def _confidence_too_high(actual: str | None, maximum: str | None) -> bool:
    if maximum is None or actual is None:
        return False
    order = {"不明": 0, "低": 1, "中": 2, "高": 3}
    return order.get(actual, -1) > order.get(maximum, 3)


def _result_dates(result, *, keyword: str | None = None) -> list[str]:
    dates: set[str] = set()
    for event in result.events:
        value = event.get("date")
        if value:
            dates.add(str(value)[:10])
    for message in result.line_messages:
        value = message.get("sent_at")
        if value:
            dates.add(str(value)[:10])
    for item in result.media_items:
        if keyword and keyword not in _media_search_text(item):
            continue
        value = item.get("captured_at") or item.get("fallback_captured_at")
        if value:
            dates.add(str(value)[:10])
    return sorted(dates)


def _filter_keyword_result(result: TimelineSearchResult, keyword: str) -> TimelineSearchResult:
    media_items = [item for item in result.media_items if keyword in _media_search_text(item)]
    return TimelineSearchResult(
        question=result.question,
        date_range=result.date_range,
        keyword=result.keyword,
        events=result.events,
        media_items=media_items,
        line_messages=result.line_messages,
        timeline_items=build_timeline_items(
            line_messages=result.line_messages,
            media_items=media_items,
        ),
    )


def _media_search_text(item: dict[str, Any]) -> str:
    return "\n".join(
        str(item.get(key) or "")
        for key in ("file_name", "caption", "ocr_text", "analysis_json", "location_name")
    )


def _search_report_text(report: dict[str, Any]) -> str:
    return "\n".join(_search_result_text(row) for row in report.get("results", []))


def _search_result_text(row: dict[str, Any]) -> str:
    parts = [str(row.get("reason") or ""), str(row.get("classification") or "")]
    for event in row.get("events", []):
        parts.extend(str(event.get(key) or "") for key in ("title", "summary_preview", "location_name"))
    for sample in row.get("line_samples", []):
        parts.extend(str(sample.get(key) or "") for key in ("time", "sender", "text"))
    for sample in row.get("ocr_samples", []):
        parts.extend(str(sample.get(key) or "") for key in ("captured_at", "file_name", "text"))
    for sample in row.get("vlm_samples", []):
        parts.extend(str(sample.get(key) or "") for key in ("captured_at", "file_name", "caption"))
    return "\n".join(parts)


def _missing_evidence_types(expected_types: list[str], result) -> list[str]:
    missing: list[str] = []
    for evidence_type in expected_types:
        normalized = evidence_type.strip().lower()
        if normalized in {"line", "line_message", "line_messages"} and not result.line_messages:
            missing.append(evidence_type)
        if normalized in {"photo", "media", "media_item", "media_items"} and not result.media_items:
            missing.append(evidence_type)
        if normalized == "event" and not result.events:
            missing.append(evidence_type)
        if normalized == "ocr":
            missing.append(evidence_type)
    return missing


def _forbidden_claims_found(answer: str, claims: list[str]) -> list[str]:
    return [claim for claim in claims if _has_non_negated_claim(answer, claim)]


def _has_non_negated_claim(answer: str, claim: str) -> bool:
    if not claim:
        return False
    start = 0
    while True:
        index = answer.find(claim, start)
        if index < 0:
            return False
        context = answer[index : index + max(len(claim) + 24, 32)]
        if not _is_negated_or_cautioned_context(context):
            return True
        start = index + len(claim)


def _is_negated_or_cautioned_context(context: str) -> bool:
    cautious_markers = (
        "断定しません",
        "断定できません",
        "断定はできません",
        "断定しない",
        "断定できない",
        "断定を避け",
        "とは言えません",
        "とはいえません",
        "とは限りません",
        "ではありません",
    )
    return any(marker in context for marker in cautious_markers)


def _load_json_or_simple_yaml(raw_text: str) -> Any:
    stripped = raw_text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return json.loads(raw_text)
    return _parse_simple_questions_yaml(raw_text)


def _parse_simple_questions_yaml(raw_text: str) -> dict[str, Any]:
    questions: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_list_key: str | None = None
    current_container_indent: int | None = None
    in_questions = False
    question_indent: int | None = None

    for raw_line in raw_text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0 and line.rstrip() in {"questions:", "cases:"}:
            in_questions = True
            continue
        if not in_questions:
            if not line.strip().startswith("- "):
                continue
            in_questions = True
        stripped = line.strip()
        if stripped.startswith("- "):
            remainder = stripped[2:].strip()
            if (
                current is not None
                and question_indent is not None
                and current_list_key is not None
                and current_container_indent is not None
                and indent > current_container_indent
            ):
                if not isinstance(current.get(current_list_key), list):
                    current[current_list_key] = []
                current.setdefault(current_list_key, []).append(_parse_yaml_scalar(remainder))
                continue
            if current is not None:
                questions.append(current)
            current = {}
            current_list_key = None
            current_container_indent = None
            question_indent = indent
            if remainder:
                key, value = _split_key_value(remainder)
                current[key] = _parse_yaml_scalar(value)
            continue
        if current is not None and ":" in stripped:
            key, value = _split_key_value(stripped)
            if (
                current_list_key is not None
                and current_container_indent is not None
                and indent > current_container_indent
            ):
                if not isinstance(current.get(current_list_key), dict):
                    current[current_list_key] = {}
                current[current_list_key][key] = _parse_yaml_scalar(value)
                continue
            if value == "":
                current[key] = []
                current_list_key = key
                current_container_indent = indent
            else:
                current[key] = _parse_yaml_scalar(value)
                current_list_key = None
                current_container_indent = None

    if current is not None:
        questions.append(current)
    return {"questions": questions}


def _split_key_value(value: str) -> tuple[str, str]:
    key, raw = value.split(":", 1)
    key = key.strip()
    if (key.startswith('"') and key.endswith('"')) or (key.startswith("'") and key.endswith("'")):
        try:
            parsed_key = ast.literal_eval(key)
        except (ValueError, SyntaxError):
            parsed_key = key.strip("\"'")
        key = str(parsed_key)
    return key, raw.strip()


def _parse_yaml_scalar(value: str) -> Any:
    if value == "":
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = [part.strip().strip("\"'") for part in value[1:-1].split(",") if part.strip()]
        return parsed if isinstance(parsed, list) else []
    if value.startswith("{") and value.endswith("}"):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = {}
        return parsed if isinstance(parsed, dict) else {}
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value.strip("\"'")
    try:
        return int(value)
    except ValueError:
        return value


def _none_or_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _none_or_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _none_or_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _none_or_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _json_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    return [str(parsed)] if str(parsed).strip() else []


def _string_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}
