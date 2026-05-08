"""Service helpers for operating the event review queue safely."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from personal_lifelog_rag.core.privacy import redact_text


REVIEW_QUEUE_LIMIT = 1_000


@dataclass(frozen=True)
class ReviewQueueFilters:
    date: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    confidence_lte: float | None = None
    title_contains: str | None = None
    location_contains: str | None = None
    modality: str | None = None
    verified: str = "all"
    hidden: str = "exclude"
    pinned: str = "all"
    evidence_count_min: int | None = None
    evidence_count_max: int | None = None
    title_category: str | None = None
    limit: int = REVIEW_QUEUE_LIMIT


def review_queue(repository, filters: ReviewQueueFilters) -> dict[str, Any]:
    """Return privacy-conscious review queue rows for UI/CLI use."""

    start_date = filters.date or filters.date_from
    end_date = filters.date or filters.date_to or filters.date_from
    include_hidden = filters.hidden in {"include", "only"}
    events = repository.list_events(
        start_date=start_date,
        end_date=end_date,
        include_hidden=include_hidden,
        limit=1_000_000,
    )
    rows = [_review_row(event) for event in events]
    rows = [row for row in rows if _matches_filters(row, filters)]
    rows.sort(
        key=lambda row: (
            not bool(row["pinned"]),
            row["date"],
            row["start_time"],
            row["event_id"],
        )
    )
    limited = rows[: max(filters.limit, 0)]
    return {
        "filters": filters.__dict__,
        "total": len(rows),
        "rows": limited,
    }


def format_review_queue(report: dict[str, Any]) -> str:
    rows = report["rows"]
    lines = [
        "Review Queue",
        f"- total: {report['total']}",
        f"- shown: {len(rows)}",
    ]
    if not rows:
        lines.append("- none")
        return "\n".join(lines)
    for row in rows:
        flags = _flags(row)
        flag_text = f" flags={','.join(flags)}" if flags else ""
        tags = f" tags={row['tags']}" if row.get("tags") else ""
        lines.append(
            f"- {row['date']} {row['time_range']} {row['title']} "
            f"confidence={_format_float(row['confidence'])} "
            f"modality={row['modality']} evidence={row['evidence_count']}"
            f"{flag_text}{tags} id={row['event_id']}"
        )
        if row.get("location_name"):
            lines.append(f"  location: {row['location_name']}")
    return "\n".join(lines)


def review_rows_for_dataframe(report: dict[str, Any]) -> list[list[Any]]:
    return [
        [
            row["event_id"],
            row["date"],
            row["time_range"],
            row["title"],
            row["location_name"],
            row["confidence"],
            row["modality"],
            row["evidence_count"],
            row["line_evidence_count"],
            row["photo_evidence_count"],
            row["verified"],
            row["hidden"],
            row["pinned"],
            row["tags"],
        ]
        for row in report["rows"]
    ]


def bulk_update_events(
    repository,
    event_ids: list[str],
    *,
    verified: bool | None = None,
    hidden: bool | None = None,
    pinned: bool | None = None,
    add_tags: list[str] | None = None,
    clear_overrides: bool = False,
) -> dict[str, Any]:
    updated: list[str] = []
    missing: list[str] = []
    for event_id in _clean_ids(event_ids):
        if repository.get_event(event_id, include_hidden=True) is None:
            missing.append(event_id)
            continue
        if clear_overrides:
            repository.delete_event_override(event_id)
        else:
            tags = _merged_tags(repository.get_event_override(event_id), add_tags)
            repository.upsert_event_override(
                event_id,
                tags=tags if add_tags is not None else None,
                is_verified=verified,
                is_hidden=hidden,
                is_pinned=pinned,
            )
        updated.append(event_id)
    return {
        "updated": updated,
        "missing": missing,
        "updated_count": len(updated),
        "missing_count": len(missing),
    }


def hide_low_confidence_line_only(
    repository,
    filters: ReviewQueueFilters,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    effective = ReviewQueueFilters(
        date=filters.date,
        date_from=filters.date_from,
        date_to=filters.date_to,
        confidence_lte=threshold,
        modality="line_only",
        verified=filters.verified,
        hidden="exclude",
        pinned=filters.pinned,
        evidence_count_min=filters.evidence_count_min,
        evidence_count_max=filters.evidence_count_max,
        title_category=filters.title_category,
        limit=filters.limit,
    )
    report = review_queue(repository, effective)
    return bulk_update_events(
        repository,
        [row["event_id"] for row in report["rows"]],
        hidden=True,
    )


def make_eval_case_yaml(
    repository,
    *,
    event_id: str | None = None,
    case_type: str = "routed_qa",
    query: str | None = None,
    expected_date: str | None = None,
) -> str:
    """Create a compact YAML fragment for private eval without raw evidence."""

    event = repository.get_event(event_id, include_hidden=True) if event_id else None
    date_value = expected_date or (str(event.get("date")) if event else None)
    title = str(event.get("title") or "event") if event else str(query or "query")
    slug = _slugify("_".join(part for part in [date_value, title] if part)) or "manual_case"
    question = query or (f"{date_value}は何していた？" if date_value else "質問を入力してください")
    if case_type == "date_qa":
        lines = [
            f"- id: manual_case_{slug}",
            "  type: date_qa",
            f"  question: {_yaml_quote(question)}",
        ]
        if date_value:
            lines.extend(["  expected_dates:", f"    - {_yaml_quote(date_value)}"])
    else:
        lines = [
            f"- id: manual_case_{slug}",
            "  type: routed_qa",
            f"  question: {_yaml_quote(question)}",
        ]
        if date_value:
            lines.extend(["  expected_top_dates:", f"    - {_yaml_quote(date_value)}"])
            lines.extend(
                [
                    "  expected_classification:",
                    f"    {_yaml_quote(date_value)}: actual_or_likely_action",
                ]
            )
    lines.extend(["  should_not_include:", "    - \"確実に\""])
    return "\n".join(lines)


def _review_row(event: dict[str, Any]) -> dict[str, Any]:
    line_count = int(event.get("line_evidence_count") or 0)
    photo_count = int(event.get("photo_evidence_count") or 0)
    evidence_count = int(event.get("event_evidence_count") or (line_count + photo_count))
    return {
        "event_id": str(event.get("id") or ""),
        "date": str(event.get("date") or ""),
        "start_time": str(event.get("start_time") or ""),
        "end_time": str(event.get("end_time") or ""),
        "time_range": _time_range(event),
        "title": redact_text(event.get("title"), max_chars=80),
        "summary_preview": redact_text(event.get("summary"), max_chars=120),
        "location_name": redact_text(event.get("location_name"), max_chars=60),
        "confidence": _float_or_none(event.get("confidence")),
        "modality": _modality(line_count, photo_count),
        "evidence_count": evidence_count,
        "line_evidence_count": line_count,
        "photo_evidence_count": photo_count,
        "verified": bool(event.get("is_verified")),
        "hidden": bool(event.get("is_hidden")),
        "pinned": bool(event.get("is_pinned")),
        "tags": ", ".join(_tags(event.get("tags_json"))),
    }


def _matches_filters(row: dict[str, Any], filters: ReviewQueueFilters) -> bool:
    if filters.confidence_lte is not None:
        confidence = row["confidence"]
        if confidence is None or confidence > filters.confidence_lte:
            return False
    if filters.title_contains and filters.title_contains not in row["title"]:
        return False
    if filters.location_contains and filters.location_contains not in row["location_name"]:
        return False
    if filters.modality and filters.modality != "all" and row["modality"] != filters.modality:
        return False
    if filters.verified == "verified" and not row["verified"]:
        return False
    if filters.verified == "unverified" and row["verified"]:
        return False
    if filters.hidden == "only" and not row["hidden"]:
        return False
    if filters.hidden == "exclude" and row["hidden"]:
        return False
    if filters.pinned == "pinned" and not row["pinned"]:
        return False
    if filters.evidence_count_min is not None and row["evidence_count"] < filters.evidence_count_min:
        return False
    if filters.evidence_count_max is not None and row["evidence_count"] > filters.evidence_count_max:
        return False
    if filters.title_category and filters.title_category != "all" and filters.title_category not in row["title"]:
        return False
    return True


def _modality(line_count: int, photo_count: int) -> str:
    if line_count and photo_count:
        return "photo_and_line"
    if line_count:
        return "line_only"
    if photo_count:
        return "photo_only"
    return "no_evidence"


def _time_range(event: dict[str, Any]) -> str:
    start = str(event.get("start_time") or "--:--")
    end = str(event.get("end_time") or "--:--")
    return f"{start[:5]}〜{end[:5]}"


def _tags(value: Any) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        parsed = str(value).replace("、", ",").split(",")
    if isinstance(parsed, list):
        return [redact_text(str(item).strip(), max_chars=24) for item in parsed if str(item).strip()]
    text = str(parsed).strip()
    return [redact_text(text, max_chars=24)] if text else []


def _merged_tags(existing: dict[str, Any] | None, new_tags: list[str] | None) -> list[str] | None:
    if new_tags is None:
        return None
    tags = _tags((existing or {}).get("tags_json"))
    for tag in new_tags:
        clean = str(tag).strip()
        if clean and clean not in tags:
            tags.append(clean)
    return tags


def _clean_ids(event_ids: list[str]) -> list[str]:
    result: list[str] = []
    for value in event_ids:
        for item in str(value).replace(",", "\n").splitlines():
            event_id = item.strip()
            if event_id and event_id not in result:
                result.append(event_id)
    return result


def _flags(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if row["verified"]:
        flags.append("verified")
    if row["hidden"]:
        flags.append("hidden")
    if row["pinned"]:
        flags.append("pinned")
    return flags


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value: Any) -> str:
    if value is None:
        return "none"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _slugify(value: str) -> str:
    text = re.sub(r"\s+", "_", value.strip())
    text = re.sub(r"[^0-9A-Za-z_\-\u3040-\u30ff\u3400-\u9fff]", "", text)
    return text[:48].strip("_")


def _yaml_quote(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)
