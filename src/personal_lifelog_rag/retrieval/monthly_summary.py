"""Period/month summaries for natural-language QA."""

from __future__ import annotations

from collections import Counter, defaultdict
import calendar
from typing import Any

from personal_lifelog_rag.core.privacy import redact_text


FOOD_TERMS = ("食事", "カフェ", "ご飯", "料理", "meal", "food", "rice", "cafe", "restaurant", "ramen")
PERFORMANCE_TERMS = ("パフォーマンス", "ステージ", "performance", "stage", "theater", "dancing", "dance")
PHOTO_TERMS = ("写真", "photo", "image", "位置情報付き写真", "写真撮影")
LINE_TERMS = ("LINE", "line", "やりとり")
CALL_TERMS = ("通話", "電話", "call")


def build_monthly_summary_report(
    repository,
    *,
    start_date: str,
    end_date: str,
    include_hidden: bool = False,
    top_days_limit: int = 5,
    events_per_day: int = 5,
) -> dict[str, Any]:
    """Aggregate local evidence into a compact range summary."""

    events = repository.list_events(
        start_date=start_date,
        end_date=end_date,
        limit=1_000_000,
        include_hidden=include_hidden,
    )
    event_ids = {str(event.get("id")) for event in events}
    evidence_rows = [row for row in repository.list_event_evidence() if str(row.get("event_id")) in event_ids]
    evidence_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence_rows:
        evidence_by_event[str(row.get("event_id"))].append(row)

    media_items = repository.list_media_items(start_date=start_date, end_date=end_date, limit=1_000_000)
    vlm_rows = _usable_vlm_rows(
        repository.list_media_vlm(start_date=start_date, end_date=end_date, statuses=["success"], limit=1_000_000)
    )
    ocr_rows = repository.list_media_ocr(start_date=start_date, end_date=end_date, statuses=["success"], limit=1_000_000)
    line_rows = repository.list_line_messages(start_date=start_date, end_date=end_date, limit=1_000_000)
    call_rows = repository.list_line_call_events(start_date=start_date, end_date=end_date, limit=1_000_000)

    title_counts = Counter(str(event.get("title") or "untitled") for event in events)
    confidence_values = [_float_or_none(event.get("confidence")) for event in events]
    confidence_values = [value for value in confidence_values if value is not None]
    evidence_type_counts = Counter(str(row.get("evidence_type") or "unknown") for row in evidence_rows)
    category_counts = _category_counts(events, vlm_rows, call_rows)
    daily = _daily_rollup(
        events=events,
        media_items=media_items,
        vlm_rows=vlm_rows,
        ocr_rows=ocr_rows,
        line_rows=line_rows,
        call_rows=call_rows,
    )
    representative_days = _representative_days(
        daily,
        events=events,
        evidence_by_event=evidence_by_event,
        limit=top_days_limit,
        events_per_day=events_per_day,
    )
    return {
        "range": {"from": start_date, "to": end_date, "label": _range_label(start_date, end_date)},
        "events_count": len(events),
        "event_evidence_count": len(evidence_rows),
        "title_distribution": dict(title_counts.most_common(10)),
        "media": {
            "photos": len(media_items),
            "gps_photos": sum(1 for item in media_items if item.get("gps_lat") is not None and item.get("gps_lon") is not None),
            "vlm_success_photos": len(vlm_rows),
            "ocr_success_photos": len(ocr_rows),
        },
        "line_messages_count": len(line_rows),
        "call_events_count": len(call_rows),
        "call_status_counts": dict(sorted(Counter(str(row.get("call_status") or "unknown") for row in call_rows).items())),
        "category_counts": category_counts,
        "confidence_distribution": _confidence_distribution(confidence_values),
        "evidence_type_counts": dict(sorted(evidence_type_counts.items())),
        "representative_days": representative_days,
        "notes": [
            "VLM-only evidence is treated as image-analysis candidate information.",
            "Hidden events are excluded unless include_hidden is enabled.",
        ],
    }


def format_monthly_summary(report: dict[str, Any]) -> str:
    """Format a compact Japanese monthly/range summary."""

    range_label = report["range"]["label"]
    media = report["media"]
    categories = report["category_counts"]
    top_titles = list(report["title_distribution"].items())[:5]
    main_titles = "、".join(f"{title}({count})" for title, count in top_titles) if top_titles else "目立つタイトルなし"
    active_categories = _active_category_text(categories)
    lines = [
        f"{range_label}の月次要約",
        "",
        (
            f"この期間はイベント{report['events_count']}件、写真{media['photos']}枚"
            f"（GPS付き{media['gps_photos']}枚）、LINE記録{report['line_messages_count']}件、"
            f"通話ログ{report['call_events_count']}件がありました。"
        ),
        (
            f"画像解析済み写真は{media['vlm_success_photos']}枚、OCR成功写真は{media['ocr_success_photos']}枚です。"
            "画像解析のみの手がかりは「候補」として扱っています。"
        ),
        f"全体の傾向: {active_categories}",
        f"イベントタイトル分布の上位: {main_titles}",
        f"confidence分布: {_format_counts_inline(report['confidence_distribution'])}",
        "",
        "代表日 top5:",
    ]
    for day in report["representative_days"]:
        lines.append(
            f"- {day['date']}: events={day['events_count']}, photos={day['photos']}, "
            f"gps={day['gps_photos']}, line={day['line_messages']}, calls={day['call_events']}"
        )
        for event in day.get("events", [])[:5]:
            flags = []
            if event.get("vlm_evidence_count"):
                flags.append("画像解析による候補")
            if event.get("line_evidence_count"):
                flags.append("LINE")
            if event.get("photo_evidence_count"):
                flags.append("写真")
            flag_text = f" [{', '.join(flags)}]" if flags else ""
            lines.append(
                f"  - {event.get('start_time') or '--:--'} {event.get('title') or 'untitled'}"
                f" confidence={_format_float(event.get('confidence'))}{flag_text}"
            )
    if not report["representative_days"]:
        lines.append("- none")
    return "\n".join(lines)


def _daily_rollup(
    *,
    events: list[dict[str, Any]],
    media_items: list[dict[str, Any]],
    vlm_rows: list[dict[str, Any]],
    ocr_rows: list[dict[str, Any]],
    line_rows: list[dict[str, Any]],
    call_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    daily: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "events_count": 0,
        "photos": 0,
        "gps_photos": 0,
        "vlm_success_photos": 0,
        "ocr_success_photos": 0,
        "line_messages": 0,
        "call_events": 0,
    })
    for event in events:
        if event.get("date"):
            daily[str(event["date"])]["events_count"] += 1
    for item in media_items:
        key = _media_date(item)
        if key:
            daily[key]["photos"] += 1
            if item.get("gps_lat") is not None and item.get("gps_lon") is not None:
                daily[key]["gps_photos"] += 1
    for row in vlm_rows:
        key = _media_date(row)
        if key:
            daily[key]["vlm_success_photos"] += 1
    for row in ocr_rows:
        key = _media_date(row)
        if key:
            daily[key]["ocr_success_photos"] += 1
    for row in line_rows:
        key = str(row.get("sent_at") or "")[:10]
        if key:
            daily[key]["line_messages"] += 1
    for row in call_rows:
        key = str(row.get("sent_at") or "")[:10]
        if key:
            daily[key]["call_events"] += 1
    return daily


def _representative_days(
    daily: dict[str, dict[str, Any]],
    *,
    events: list[dict[str, Any]],
    evidence_by_event: dict[str, list[dict[str, Any]]],
    limit: int,
    events_per_day: int,
) -> list[dict[str, Any]]:
    events_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("date"):
            events_by_date[str(event["date"])].append(event)

    def score(item: tuple[str, dict[str, Any]]) -> tuple[float, str]:
        date_value, row = item
        value = (
            row["events_count"] * 5
            + min(row["photos"], 80) * 0.08
            + min(row["gps_photos"], 80) * 0.06
            + min(row["line_messages"], 80) * 0.04
            + row["call_events"] * 1.5
            + row["vlm_success_photos"] * 0.05
        )
        return (value, date_value)

    rows: list[dict[str, Any]] = []
    for date_value, counts in sorted(daily.items(), key=score, reverse=True)[:limit]:
        event_rows = sorted(
            events_by_date.get(date_value, []),
            key=lambda event: (
                -float(event.get("confidence") or 0.0),
                str(event.get("start_time") or ""),
                str(event.get("id") or ""),
            ),
        )
        rows.append(
            {
                "date": date_value,
                **counts,
                "events": [_event_summary(event, evidence_by_event.get(str(event.get("id")), [])) for event in event_rows[:events_per_day]],
            }
        )
    return rows


def _event_summary(event: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("evidence_type") or "unknown") for row in evidence)
    return {
        "id": event.get("id"),
        "date": event.get("date"),
        "start_time": event.get("start_time"),
        "end_time": event.get("end_time"),
        "title": event.get("title"),
        "summary_preview": redact_text(event.get("summary"), max_chars=120),
        "confidence": event.get("confidence"),
        "line_evidence_count": counts.get("line", 0),
        "photo_evidence_count": counts.get("photo", 0),
        "ocr_evidence_count": counts.get("ocr", 0),
        "vlm_evidence_count": counts.get("vlm", 0),
    }


def _category_counts(events: list[dict[str, Any]], vlm_rows: list[dict[str, Any]], call_rows: list[dict[str, Any]]) -> dict[str, int]:
    event_texts = [" ".join(str(event.get(key) or "") for key in ("title", "summary", "location_name")) for event in events]
    vlm_texts = [
        " ".join(
            str(row.get(key) or "")
            for key in (
                "caption",
                "short_caption",
                "scene_tags_json",
                "object_tags_json",
                "activity_tags_json",
                "food_cues_json",
                "location_cues_json",
            )
        )
        for row in vlm_rows
    ]
    return {
        "food_cafe": _count_texts(event_texts + vlm_texts, FOOD_TERMS),
        "performance_stage": _count_texts(event_texts + vlm_texts, PERFORMANCE_TERMS),
        "photo": _count_texts(event_texts, PHOTO_TERMS),
        "line": _count_texts(event_texts, LINE_TERMS),
        "call": _count_texts(event_texts, CALL_TERMS) + len(call_rows),
    }


def _usable_vlm_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usable: list[dict[str, Any]] = []
    for row in rows:
        engine = str(row.get("vlm_engine") or "").lower()
        model = str(row.get("model_name") or "").lower()
        if "fake" in engine or "fake" in model:
            continue
        if int(row.get("vlm_is_hidden") or 0) or int(row.get("vlm_is_wrong") or 0):
            continue
        if int(row.get("vlm_is_event_usable") or 1) == 0:
            continue
        usable.append(row)
    return usable


def _count_texts(texts: list[str], terms: tuple[str, ...]) -> int:
    lowered_terms = tuple(term.lower() for term in terms)
    return sum(1 for text in texts if any(term in text.lower() for term in lowered_terms))


def _active_category_text(categories: dict[str, int]) -> str:
    labels = {
        "food_cafe": "食事・カフェ",
        "performance_stage": "パフォーマンス・ステージ",
        "photo": "写真記録",
        "line": "LINEのやりとり",
        "call": "通話・連絡",
    }
    active = [f"{labels.get(key, key)}({value})" for key, value in sorted(categories.items(), key=lambda item: -item[1]) if value]
    return "、".join(active[:5]) if active else "明確な偏りは少なめです"


def _confidence_distribution(values: list[float]) -> dict[str, int]:
    buckets = {"高": 0, "中": 0, "低": 0, "unknown": 0}
    for value in values:
        if value >= 0.75:
            buckets["高"] += 1
        elif value >= 0.45:
            buckets["中"] += 1
        else:
            buckets["低"] += 1
    return buckets


def _format_counts_inline(counts: dict[str, Any]) -> str:
    return "、".join(f"{key}={value}" for key, value in counts.items()) if counts else "none"


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _media_date(row: dict[str, Any]) -> str:
    return str(row.get("captured_at") or row.get("fallback_captured_at") or "")[:10]


def _range_label(start_date: str, end_date: str) -> str:
    if _is_calendar_month(start_date, end_date):
        return start_date[:7]
    return f"{start_date}..{end_date}"


def _is_calendar_month(start_date: str, end_date: str) -> bool:
    try:
        year = int(start_date[:4])
        month = int(start_date[5:7])
    except ValueError:
        return False
    return start_date == f"{year:04d}-{month:02d}-01" and end_date == f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"
