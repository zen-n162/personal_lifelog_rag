"""Local SQLite keyword search for lifelog records."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Any

from personal_lifelog_rag.core.privacy import redact_text
from personal_lifelog_rag.line.call_index import format_duration
from personal_lifelog_rag.retrieval.query_intent import QueryIntent, infer_query_intent
from personal_lifelog_rag.retrieval.search_ranking import SECTION_TITLES, SearchMode, rank_search_results


STOPWORDS = {
    "いつ",
    "どこ",
    "だれ",
    "誰",
    "何",
    "なに",
    "行った",
    "行く",
    "した",
    "していた",
    "してた",
    "まとめて",
}


@dataclass(frozen=True)
class LocalSearchOptions:
    query: str
    date_from: str | None = None
    date_to: str | None = None
    limit: int = 5
    mode: SearchMode = "all"
    intent: QueryIntent | None = None
    include_hidden: bool = False


def local_text_search(repository, options: LocalSearchOptions) -> dict[str, Any]:
    terms = extract_search_terms(options.query)
    intent = infer_query_intent(options.query, override=options.intent)
    records = repository.search_text_records(
        terms=terms,
        start_date=options.date_from,
        end_date=options.date_to,
        limit=20_000,
        include_hidden=options.include_hidden,
    )
    grouped = _group_by_date(records)
    results = [_build_day_result(date, rows) for date, rows in grouped.items()]
    ranked = rank_search_results(
        repository,
        query=terms[0] if terms else options.query,
        intent=intent,
        results=results,
        mode=options.mode,
    )
    limited = ranked[: max(options.limit, 0)]
    return {
        "query": options.query,
        "terms": terms,
        "backend": "sqlite_like",
        "intent": intent,
        "mode": options.mode,
        "date_from": options.date_from,
        "date_to": options.date_to,
        "include_hidden": options.include_hidden,
        "total_dates": len(ranked),
        "results": limited,
    }


def format_local_search_report(report: dict[str, Any]) -> str:
    if not report["results"]:
        return f"検索結果は見つかりませんでした: {report['query']}"

    lines = [
        f"Search Results: {report['query']}",
        f"- backend: {report['backend']}",
        f"- intent: {report.get('intent') or 'generic'}",
        f"- mode: {report.get('mode') or 'all'}",
        f"- terms: {', '.join(report['terms']) if report['terms'] else 'none'}",
        f"- dates: {report['total_dates']}",
    ]
    for classification in (
        "actual_or_likely_action",
        "plan_or_candidate",
        "mention_only",
        "unknown",
    ):
        section_rows = [row for row in report["results"] if row.get("classification") == classification]
        if not section_rows:
            continue
        lines.append("")
        lines.append(f"{SECTION_TITLES[classification]}:")
        for index, result in enumerate(section_rows, start=1):
            lines.append(
                f"{index}. {result['date']} "
                f"confidence={result['confidence_label']} "
                f"score={result['ranking_score']:.2f} "
                f"classification={result['classification']}"
            )
            lines.append(f"   reason: {result.get('reason') or 'none'}")
            lines.append(
                "   counts: "
                f"events={result['event_count']}, "
                f"line={result['line_match_count']}, "
                f"ocr={result.get('ocr_match_count', 0)}, "
                f"vlm={result.get('vlm_match_count', 0)}, "
                f"photos={result.get('same_day_photo_count', result['media_match_count'])}, "
                f"gps_photos={result.get('same_day_gps_photo_count', 0)}"
            )
            if result.get("call_summary"):
                summary = result["call_summary"]
                lines.append(
                    "   call summary: "
                    f"completed={summary.get('completed', 0)}, "
                    f"missed={summary.get('missed', 0)}, "
                    f"unanswered={summary.get('unanswered', 0)}, "
                    f"canceled={summary.get('canceled', 0)}, "
                    f"total_duration={format_duration(summary.get('total_duration_sec'))}"
                )
            if result["events"]:
                lines.append("   related events:")
                for event in result["events"][:3]:
                    time_range = _event_time_range(event)
                    flags = _event_flags(event)
                    flag_text = f" [{', '.join(flags)}]" if flags else ""
                    lines.append(
                        f"   - {time_range} {event['title']} "
                        f"confidence={_format_float(event.get('confidence'))}{flag_text} "
                        f"evidence={event.get('event_evidence_count', 0)}"
                    )
                    place_label = event.get("location_name") or event.get("place_display_name") or event.get("place_public_name")
                    if place_label:
                        lines.append(f"     place: {place_label}")
                    if event.get("summary_preview"):
                        lines.append(f"     {event['summary_preview']}")
            if result["line_samples"]:
                lines.append("   LINE samples:")
                for sample in result["line_samples"][:5]:
                    lines.append(f"   - {sample['time']} {sample['sender']}: {sample['text']}")
            if result.get("ocr_samples"):
                lines.append("   OCR evidence:")
                for sample in result["ocr_samples"][:5]:
                    lines.append(f"   - {sample['time']} {sample['file_name']}: {sample['text']}")
            if result.get("vlm_samples"):
                lines.append("   VLM evidence:")
                lines.append("   - 画像解析は自動推定です。必要に応じて写真を確認してください。")
                for sample in result["vlm_samples"][:5]:
                    lines.append(f"   - {sample['time']} {sample['file_name']}: {sample['caption']}")
            lines.append("   evidence: " + ", ".join(result["evidence_types"]))
            lines.append("")
    return "\n".join(lines).rstrip()


def extract_search_terms(query: str) -> list[str]:
    text = query.strip()
    if not text:
        return []
    cleaned = re.sub(r"[?？。!！]", " ", text)
    cleaned = re.sub(r"(のはいつ|はいつ|いつ|ですか|でしたか)", " ", cleaned)
    rough_parts = re.split(r"[\s,、/]+|(?:で|に|を|は|が|と)", cleaned)
    terms: list[str] = []
    for value in [text, *rough_parts]:
        term = _normalize_search_term(value)
        if not term or term in STOPWORDS:
            continue
        if term not in terms:
            terms.append(term)
    if len(terms) > 1 and terms[0] in {text, _normalize_search_term(text)}:
        # For natural questions, exact full-sentence LIKE is usually too narrow.
        terms = terms[1:]
    return terms or [text]


def _normalize_search_term(value: str) -> str:
    term = re.sub(r"[?？。!！]", "", value).strip(" 　")
    term = re.sub(
        r"(した日は|した日|したのは|したの|した|する日は|する日|するのは|するの|する|日は|日)$",
        "",
        term,
    )
    return term.strip(" 　")


def _group_by_date(records: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"events": [], "line_messages": [], "media_items": [], "media_ocr": [], "media_vlm": []}
    )
    for event in records["events"]:
        date = _record_date(event, "event")
        if date:
            grouped[date]["events"].append(event)
    for message in records["line_messages"]:
        date = _record_date(message, "line")
        if date:
            grouped[date]["line_messages"].append(message)
    for media in records["media_items"]:
        date = _record_date(media, "media")
        if date:
            grouped[date]["media_items"].append(media)
    for row in records.get("media_ocr", []):
        date = _record_date(row, "media_ocr")
        if date:
            grouped[date]["media_ocr"].append(row)
    for row in records.get("media_vlm", []):
        date = _record_date(row, "media_vlm")
        if date:
            grouped[date]["media_vlm"].append(row)
    return dict(grouped)


def _build_day_result(date: str, rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    events = [_event_preview(event) for event in rows["events"]]
    line_samples = [_line_preview(message) for message in rows["line_messages"][:5]]
    ocr_samples = [_ocr_preview(row) for row in rows.get("media_ocr", [])[:5]]
    vlm_samples = [_vlm_preview(row) for row in rows.get("media_vlm", [])[:5]]
    media_count = len(rows["media_items"])
    ocr_count = len(rows.get("media_ocr", []))
    vlm_count = len(rows.get("media_vlm", []))
    event_count = len(rows["events"])
    line_count = len(rows["line_messages"])
    total_matches = event_count + line_count + media_count + ocr_count + vlm_count
    score = event_count * 3.0 + line_count * 2.0 + media_count + ocr_count * 2.0 + vlm_count * 2.5
    confidence = _search_confidence(
        event_count=event_count,
        line_count=line_count,
        media_count=media_count,
        max_event_confidence=max((_float_or_none(event.get("confidence")) or 0.0 for event in rows["events"]), default=0.0),
    )
    evidence_types = []
    if event_count:
        evidence_types.append("events")
        if any(event.get("location_name") or event.get("place_display_name") or event.get("place_public_name") for event in rows["events"]):
            evidence_types.append("place")
    if line_count:
        evidence_types.append("line")
    if media_count:
        evidence_types.append("photos")
    if ocr_count:
        evidence_types.append("ocr")
    if vlm_count:
        evidence_types.append("vlm")
    return {
        "date": date,
        "score": round(score, 3),
        "confidence": confidence,
        "confidence_label": _confidence_label(confidence),
        "total_matches": total_matches,
        "event_count": event_count,
        "line_match_count": line_count,
        "media_match_count": media_count,
        "ocr_match_count": ocr_count,
        "vlm_match_count": vlm_count,
        "events": events,
        "line_samples": line_samples,
        "ocr_samples": ocr_samples,
        "vlm_samples": vlm_samples,
        "evidence_types": evidence_types or ["none"],
    }


def _search_confidence(
    *,
    event_count: int,
    line_count: int,
    media_count: int,
    max_event_confidence: float,
) -> float:
    source_types = sum(1 for count in (event_count, line_count, media_count) if count)
    total = event_count + line_count + media_count
    confidence = 0.15 + source_types * 0.15 + min(total, 8) * 0.04 + min(max_event_confidence, 0.95) * 0.2
    return round(min(confidence, 0.95), 2)


def _confidence_label(value: float) -> str:
    if value >= 0.75:
        return "高"
    if value >= 0.45:
        return "中"
    return "低"


def _event_preview(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": event.get("id"),
        "date": event.get("date"),
        "start_time": event.get("start_time"),
        "end_time": event.get("end_time"),
        "title": redact_text(event.get("title"), max_chars=80),
        "summary_preview": redact_text(event.get("summary"), max_chars=100),
        "location_name": redact_text(event.get("location_name"), max_chars=40),
        "place_display_name": redact_text(event.get("place_display_name"), max_chars=40),
        "place_public_name": redact_text(event.get("place_public_name"), max_chars=40),
        "place_category": event.get("place_category"),
        "place_manual_verified": int(event.get("place_manual_verified") or 0),
        "confidence": event.get("confidence"),
        "event_evidence_count": int(event.get("event_evidence_count") or 0),
        "line_evidence_count": int(event.get("line_evidence_count") or 0),
        "photo_evidence_count": int(event.get("photo_evidence_count") or 0),
        "is_verified": int(event.get("is_verified") or 0),
        "is_hidden": int(event.get("is_hidden") or 0),
        "is_pinned": int(event.get("is_pinned") or 0),
        "tags_json": event.get("tags_json"),
    }


def _line_preview(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": message.get("id"),
        "time": _time_label(message.get("sent_at")),
        "sender": redact_text(message.get("sender"), max_chars=24),
        "text": redact_text(message.get("text"), max_chars=60),
    }


def _ocr_preview(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "media_id": row.get("media_id"),
        "time": _time_label(row.get("captured_at") or row.get("fallback_captured_at")),
        "file_name": redact_text(row.get("file_name"), max_chars=60),
        "text": redact_text(row.get("ocr_text_redacted") or row.get("ocr_text"), max_chars=80),
        "status": row.get("status"),
    }


def _vlm_preview(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "media_id": row.get("media_id"),
        "time": _time_label(row.get("captured_at") or row.get("fallback_captured_at")),
        "file_name": redact_text(row.get("file_name"), max_chars=60),
        "caption": redact_text(row.get("short_caption") or row.get("caption"), max_chars=90),
        "scene_tags": row.get("scene_tags_json"),
        "object_tags": row.get("object_tags_json"),
        "activity_tags": row.get("activity_tags_json"),
        "food_cues": row.get("food_cues_json"),
        "status": row.get("status"),
    }


def _record_date(record: dict[str, Any], source_type: str) -> str | None:
    if source_type == "event":
        value = record.get("date")
    elif source_type == "line":
        value = record.get("sent_at")
    else:
        value = record.get("captured_at") or record.get("fallback_captured_at")
    return str(value)[:10] if value else None


def _event_time_range(event: dict[str, Any]) -> str:
    start = event.get("start_time")
    end = event.get("end_time")
    if start and end and start != end:
        return f"{str(start)[:5]}-{str(end)[:5]}"
    if start:
        return str(start)[:5]
    return "time unknown"


def _event_flags(event: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if event.get("is_verified"):
        flags.append("手動確認済み")
    if event.get("is_pinned"):
        flags.append("pinned")
    if event.get("is_hidden"):
        flags.append("hidden")
    return flags


def _time_label(value: Any) -> str:
    if not value:
        return "time unknown"
    text = str(value)
    return text[11:16] if len(text) >= 16 and "T" in text else text[:16]


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "none"


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
