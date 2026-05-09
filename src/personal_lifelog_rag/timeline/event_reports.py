"""Privacy-conscious event statistics and listings."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from typing import Any

from personal_lifelog_rag.core.privacy import redact_text


def event_stats(
    repository,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    events = repository.list_events(start_date=start_date, end_date=end_date, limit=1_000_000)
    event_ids = {event["id"] for event in events}
    evidence = [
        row
        for row in repository.list_event_evidence()
        if row.get("event_id") in event_ids
    ]
    evidence_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence:
        evidence_by_event[str(row["event_id"])].append(row)

    confidence_values = [
        float(event["confidence"])
        for event in events
        if event.get("confidence") is not None
    ]
    modality_counts = _modality_counts(events, evidence_by_event)
    return {
        "range": {"from": start_date, "to": end_date},
        "total_events": len(events),
        "total_event_evidence": len(evidence),
        "monthly_event_counts": dict(sorted(Counter((event.get("date") or "")[:7] for event in events if event.get("date")).items())),
        "daily_event_counts_top20": _top_counts(Counter((event.get("date") or "")[:10] for event in events if event.get("date")), limit=20),
        "title_counts": _top_counts(Counter(str(event.get("title") or "untitled") for event in events), limit=50),
        "confidence": _confidence_stats(confidence_values),
        "confidence_buckets": _confidence_buckets(confidence_values),
        "evidence_type_counts": dict(sorted(Counter(str(row.get("evidence_type") or "unknown") for row in evidence).items())),
        "modality_counts": modality_counts,
        "low_confidence_events": _low_confidence_events(events, limit=20),
    }


def format_event_stats(report: dict[str, Any]) -> str:
    confidence = report["confidence"]
    lines = [
        "Event Stats",
        f"- total_events: {report['total_events']}",
        f"- total_event_evidence: {report['total_event_evidence']}",
        "- confidence min/max/avg: "
        f"{_format_float(confidence['min'])} / {_format_float(confidence['max'])} / {_format_float(confidence['avg'])}",
        "",
        "Monthly event counts:",
    ]
    lines.extend(_format_counts(report["monthly_event_counts"]))
    lines.append("")
    lines.append("Daily event counts top20:")
    lines.extend(_format_counts(report["daily_event_counts_top20"]))
    lines.append("")
    lines.append("Title counts:")
    lines.extend(_format_counts(report["title_counts"]))
    lines.append("")
    lines.append("Confidence buckets:")
    lines.extend(_format_counts(report["confidence_buckets"]))
    lines.append("")
    lines.append("Evidence type counts:")
    lines.extend(_format_counts(report["evidence_type_counts"]))
    lines.append("")
    lines.append("Event modality counts:")
    lines.extend(_format_counts(report["modality_counts"]))
    lines.append("")
    lines.append("Low confidence events:")
    if not report["low_confidence_events"]:
        lines.append("- none")
    for event in report["low_confidence_events"]:
        lines.append(
            f"- {event['date']} {event.get('start_time') or '--:--'} "
            f"{event['title']} confidence={_format_float(event.get('confidence'))} id={event['id']}"
        )
    return "\n".join(lines)


def list_events_report(
    repository,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    with_evidence: bool = False,
    include_hidden: bool = False,
) -> list[dict[str, Any]]:
    events = repository.list_events(
        start_date=start_date,
        end_date=end_date,
        limit=1_000_000,
        include_hidden=include_hidden,
    )
    rows: list[dict[str, Any]] = []
    for event in events:
        evidence = repository.list_event_evidence(str(event["id"]))
        line_count = sum(1 for row in evidence if row.get("evidence_type") == "line")
        photo_count = sum(1 for row in evidence if row.get("evidence_type") == "photo")
        ocr_count = sum(1 for row in evidence if row.get("evidence_type") == "ocr")
        vlm_count = sum(1 for row in evidence if row.get("evidence_type") == "vlm")
        row = {
            "id": event.get("id"),
            "date": event.get("date"),
            "start_time": event.get("start_time"),
            "end_time": event.get("end_time"),
            "title": event.get("title"),
            "summary": event.get("summary"),
            "summary_preview": redact_text(event.get("summary"), max_chars=120),
            "confidence": event.get("confidence"),
            "event_evidence_count": len(evidence),
            "line_evidence_count": line_count,
            "photo_evidence_count": photo_count,
            "ocr_evidence_count": ocr_count,
            "vlm_evidence_count": vlm_count,
            "location_name": event.get("location_name"),
            "source": event.get("source"),
            "generation_method": event.get("generation_method"),
            "is_user_edited": int(event.get("is_user_edited") or 0),
            "is_verified": int(event.get("is_verified") or 0),
            "is_hidden": int(event.get("is_hidden") or 0),
            "is_pinned": int(event.get("is_pinned") or 0),
            "tags_json": event.get("tags_json"),
            "title_override": event.get("title_override"),
            "summary_override": event.get("summary_override"),
            "location_name_override": event.get("location_name_override"),
        }
        if with_evidence:
            row["evidence"] = _evidence_preview(repository, evidence)
        rows.append(row)
    return rows


def format_event_list(rows: list[dict[str, Any]], *, with_evidence: bool = False) -> str:
    if not rows:
        return "Events: none"
    lines = [f"Events: {len(rows)}"]
    for event in rows:
        lines.append("")
        header = (
            f"- {event['date']} {event.get('start_time') or '--:--'}"
            f"〜{event.get('end_time') or '--:--'} {event.get('title') or 'untitled'}"
            f" confidence={_format_float(event.get('confidence'))}"
        )
        lines.append(header)
        lines.append(f"  id: {event['id']}")
        if event.get("location_name"):
            lines.append(f"  location: {event['location_name']}")
        flags = _event_flags(event)
        if flags:
            lines.append(f"  flags: {', '.join(flags)}")
        if event.get("tags_json"):
            lines.append(f"  tags: {_tags_preview(event.get('tags_json'))}")
        if event.get("summary_preview"):
            lines.append(f"  summary: {event['summary_preview']}")
        lines.append(
            "  evidence: "
            f"total={event['event_evidence_count']}, "
            f"line={event['line_evidence_count']}, "
            f"photo={event['photo_evidence_count']}, "
            f"ocr={event.get('ocr_evidence_count', 0)}, "
            f"vlm={event.get('vlm_evidence_count', 0)}"
        )
        if with_evidence:
            evidence = event.get("evidence") or {}
            if evidence.get("line"):
                lines.append("  line evidence:")
                for item in evidence["line"]:
                    lines.append(f"    - {item['sent_at']} {item['sender']}: {item['text']}")
            if evidence.get("photo"):
                lines.append("  photo evidence:")
                for item in evidence["photo"]:
                    lines.append(f"    - {item['captured_at']} {item['file_name']}")
            if evidence.get("ocr"):
                lines.append("  OCR evidence:")
                for item in evidence["ocr"]:
                    lines.append(f"    - {item['captured_at']} {item['text']}")
            if evidence.get("vlm"):
                lines.append("  VLM evidence:")
                for item in evidence["vlm"]:
                    lines.append(f"    - {item['captured_at']} {item['caption']}")
    return "\n".join(lines)


def _evidence_preview(repository, evidence: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    line_rows: list[dict[str, Any]] = []
    photo_rows: list[dict[str, Any]] = []
    ocr_rows: list[dict[str, Any]] = []
    vlm_rows: list[dict[str, Any]] = []
    for row in evidence:
        evidence_type = row.get("evidence_type")
        if evidence_type == "line" and len(line_rows) < 5:
            record = repository.get_embedding_record("line_message", str(row.get("evidence_id"))) or {}
            line_rows.append(
                {
                    "id": row.get("evidence_id"),
                    "sent_at": _time_preview(record.get("sent_at")),
                    "sender": redact_text(record.get("sender"), max_chars=24),
                    "text": redact_text(record.get("text"), max_chars=60),
                }
            )
        elif evidence_type == "photo" and len(photo_rows) < 5:
            record = repository.get_embedding_record("media_item", str(row.get("evidence_id"))) or {}
            photo_rows.append(
                {
                    "id": row.get("evidence_id"),
                    "captured_at": _time_preview(record.get("captured_at") or record.get("fallback_captured_at")),
                    "file_name": redact_text(record.get("file_name"), max_chars=80),
                }
            )
        elif evidence_type == "ocr" and len(ocr_rows) < 5:
            record = repository.get_media_ocr(str(row.get("evidence_id"))) or {}
            ocr_rows.append(
                {
                    "id": row.get("evidence_id"),
                    "captured_at": _time_preview(record.get("captured_at") or record.get("fallback_captured_at")),
                    "text": redact_text(record.get("ocr_text_redacted") or record.get("ocr_text"), max_chars=60),
                }
            )
        elif evidence_type == "vlm" and len(vlm_rows) < 5:
            record = repository.get_media_vlm(str(row.get("evidence_id"))) or {}
            vlm_rows.append(
                {
                    "id": row.get("evidence_id"),
                    "captured_at": _time_preview(record.get("captured_at") or record.get("fallback_captured_at")),
                    "caption": redact_text(record.get("short_caption") or record.get("caption"), max_chars=80),
                }
            )
    return {"line": line_rows, "photo": photo_rows, "ocr": ocr_rows, "vlm": vlm_rows}


def _modality_counts(events: list[dict[str, Any]], evidence_by_event: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    counts = {"photo_only": 0, "line_only": 0, "photo_and_line": 0, "no_evidence": 0}
    for event in events:
        types = {row.get("evidence_type") for row in evidence_by_event.get(str(event["id"]), [])}
        has_photo = "photo" in types
        has_line = "line" in types
        if has_photo and has_line:
            counts["photo_and_line"] += 1
        elif has_photo:
            counts["photo_only"] += 1
        elif has_line:
            counts["line_only"] += 1
        else:
            counts["no_evidence"] += 1
    return counts


def _confidence_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "avg": None}
    return {
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
    }


def _confidence_buckets(values: list[float]) -> dict[str, int]:
    buckets = {
        "0.0-0.3": 0,
        "0.3-0.6": 0,
        "0.6-0.8": 0,
        "0.8-1.0": 0,
    }
    for value in values:
        if value < 0.3:
            buckets["0.0-0.3"] += 1
        elif value < 0.6:
            buckets["0.3-0.6"] += 1
        elif value < 0.8:
            buckets["0.6-0.8"] += 1
        else:
            buckets["0.8-1.0"] += 1
    return buckets


def _low_confidence_events(events: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    low = [
        event
        for event in events
        if event.get("confidence") is not None and float(event["confidence"]) < 0.5
    ]
    low.sort(key=lambda event: (float(event.get("confidence") or 0.0), event.get("date") or "", event.get("start_time") or ""))
    return [
        {
            "id": event.get("id"),
            "date": event.get("date"),
            "start_time": event.get("start_time"),
            "title": event.get("title"),
            "confidence": event.get("confidence"),
        }
        for event in low[:limit]
    ]


def _top_counts(counter: Counter[str], *, limit: int) -> dict[str, int]:
    return dict(counter.most_common(limit))


def _format_counts(counts: dict[str, int]) -> list[str]:
    if not counts:
        return ["- none"]
    return [f"- {key}: {value}" for key, value in counts.items()]


def _format_float(value: Any) -> str:
    if value is None:
        return "none"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _time_preview(value: Any) -> str:
    if not value:
        return "time unknown"
    text = str(value)
    return text[11:16] if len(text) >= 16 and "T" in text else text[:16]


def _event_flags(event: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if event.get("is_verified"):
        flags.append("manual verified")
    if event.get("is_pinned"):
        flags.append("pinned")
    if event.get("is_hidden"):
        flags.append("hidden")
    if event.get("title_override") or event.get("summary_override") or event.get("location_name_override"):
        flags.append("overridden")
    return flags


def _tags_preview(value: Any) -> str:
    if not value:
        return ""
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return redact_text(str(value), max_chars=80)
    if isinstance(parsed, list):
        return ", ".join(redact_text(str(item), max_chars=20) for item in parsed[:10])
    return redact_text(str(value), max_chars=80)


def report_to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
