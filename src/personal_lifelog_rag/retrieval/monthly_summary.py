"""Period/month summaries for natural-language QA."""

from __future__ import annotations

from collections import Counter, defaultdict
import calendar
from typing import Any

from personal_lifelog_rag.core.privacy import redact_text
from personal_lifelog_rag.db.repository import connect
from personal_lifelog_rag.db.schema import initialize_schema
from personal_lifelog_rag.faces.person_service import public_person_name
from personal_lifelog_rag.places.location_store import public_place_label


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
    public: bool = False,
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
    place_summary = _place_summary(repository, start_date=start_date, end_date=end_date, public=public)
    person_summary = _person_summary(repository, start_date=start_date, end_date=end_date, public=public)
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
        "place_summary": place_summary,
        "person_summary": person_summary,
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
    place_summary = report.get("place_summary") or {}
    person_summary = report.get("person_summary") or {}
    place_category_text = _format_counts_inline(place_summary.get("category_counts") or {})
    place_label_text = _format_counts_inline(place_summary.get("label_counts") or {})
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
        f"場所カテゴリ候補: {place_category_text}",
        f"代表的な場所ラベル候補: {place_label_text}",
        f"イベントタイトル分布の上位: {main_titles}",
        f"confidence分布: {_format_counts_inline(report['confidence_distribution'])}",
    ]
    if person_summary.get("enabled") and person_summary.get("rows"):
        lines.extend(
            [
                "手動リンク済み人物の関連候補:",
                (
                    f"- event_people={person_summary.get('event_people_count', 0)}, "
                    f"media_people={person_summary.get('media_people_count', 0)}, "
                    f"LINE={person_summary.get('line_message_count', 0)}, "
                    f"calls={person_summary.get('call_count', 0)}"
                ),
            ]
        )
        for row in person_summary["rows"][:5]:
            lines.append(
                f"  - {row.get('label')}: events={row.get('event_count', 0)} "
                f"media={row.get('media_count', 0)} line={row.get('line_count', 0)} calls={row.get('call_count', 0)}"
            )
        lines.append("  - 人物関連は手動リンク済みの範囲での候補です。関係性は推定していません。")
    elif person_summary.get("public_hidden"):
        lines.append("人物関連情報: public modeでは非表示です。")
    lines.extend(["", "代表日 top5:"])
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
        "location_name": redact_text(event.get("location_name"), max_chars=80),
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


def _place_summary(repository, *, start_date: str, end_date: str, public: bool = False) -> dict[str, Any]:
    try:
        with connect(repository.db_path) as connection:
            initialize_schema(connection)
            rows = connection.execute(
                """
                SELECT places.id, places.display_name, places.public_name, places.category,
                       places.privacy_level, COUNT(DISTINCT event_places.event_id) AS event_count
                FROM event_places
                JOIN places ON places.id = event_places.place_id
                JOIN events ON events.id = event_places.event_id
                WHERE substr(events.date, 1, 10) >= ?
                  AND substr(events.date, 1, 10) <= ?
                GROUP BY places.id
                ORDER BY event_count DESC, places.category ASC, places.id ASC
                LIMIT 20
                """,
                (start_date, end_date),
            ).fetchall()
    except Exception:
        return {"category_counts": {}, "label_counts": {}}
    category_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    for row in rows:
        count = int(row["event_count"] or 0)
        category = str(row["category"] or "other")
        category_counts[category] += count
        place = dict(row)
        label = public_place_label(place) if public else str(row["display_name"] or row["public_name"] or category)
        label_counts[label] += count
    return {
        "category_counts": dict(category_counts.most_common(8)),
        "label_counts": dict(label_counts.most_common(8)),
    }


def _person_summary(repository, *, start_date: str, end_date: str, public: bool = False) -> dict[str, Any]:
    if public:
        return {"enabled": False, "public_hidden": True, "rows": []}
    try:
        with connect(repository.db_path) as connection:
            initialize_schema(connection)
            event_rows = connection.execute(
                """
                SELECT persons.*,
                       COUNT(DISTINCT event_people.event_id) AS event_count
                FROM event_people
                JOIN persons ON persons.id = event_people.person_id
                JOIN events ON events.id = event_people.event_id
                WHERE substr(events.date, 1, 10) >= ?
                  AND substr(events.date, 1, 10) <= ?
                  AND event_people.hidden = 0
                  AND persons.manual_verified = 1
                  AND persons.hidden = 0
                  AND persons.deleted_at IS NULL
                GROUP BY persons.id
                ORDER BY event_count DESC, persons.display_name ASC
                LIMIT 20
                """,
                (start_date, end_date),
            ).fetchall()
            media_rows = connection.execute(
                """
                SELECT media_people.person_id, COUNT(DISTINCT media_people.media_id) AS media_count
                FROM media_people
                JOIN persons ON persons.id = media_people.person_id
                JOIN media_items ON media_items.id = media_people.media_id
                WHERE substr(COALESCE(media_items.captured_at, media_items.fallback_captured_at), 1, 10) >= ?
                  AND substr(COALESCE(media_items.captured_at, media_items.fallback_captured_at), 1, 10) <= ?
                  AND media_people.hidden = 0
                  AND persons.manual_verified = 1
                  AND persons.hidden = 0
                  AND persons.deleted_at IS NULL
                GROUP BY media_people.person_id
                """,
                (start_date, end_date),
            ).fetchall()
            line_rows = connection.execute(
                """
                SELECT line_speaker_links.person_id,
                       COUNT(*) AS line_count
                FROM line_speaker_links
                JOIN persons ON persons.id = line_speaker_links.person_id
                JOIN line_messages
                  ON line_messages.chat_id = line_speaker_links.chat_id
                 AND line_messages.sender = line_speaker_links.speaker_name
                WHERE substr(line_messages.sent_at, 1, 10) >= ?
                  AND substr(line_messages.sent_at, 1, 10) <= ?
                  AND line_speaker_links.verified_by_user = 1
                  AND persons.manual_verified = 1
                  AND persons.hidden = 0
                  AND persons.deleted_at IS NULL
                GROUP BY line_speaker_links.person_id
                """,
                (start_date, end_date),
            ).fetchall()
            call_rows = connection.execute(
                """
                SELECT line_speaker_links.person_id,
                       COUNT(*) AS call_count
                FROM line_speaker_links
                JOIN persons ON persons.id = line_speaker_links.person_id
                JOIN line_call_events
                  ON line_call_events.chat_id = line_speaker_links.chat_id
                 AND line_call_events.sender = line_speaker_links.speaker_name
                WHERE substr(line_call_events.sent_at, 1, 10) >= ?
                  AND substr(line_call_events.sent_at, 1, 10) <= ?
                  AND line_speaker_links.verified_by_user = 1
                  AND persons.manual_verified = 1
                  AND persons.hidden = 0
                  AND persons.deleted_at IS NULL
                GROUP BY line_speaker_links.person_id
                """,
                (start_date, end_date),
            ).fetchall()
            person_rows = connection.execute(
                """
                SELECT id, display_name, public_name
                FROM persons
                WHERE manual_verified = 1
                  AND hidden = 0
                  AND deleted_at IS NULL
                """
            ).fetchall()
    except Exception:
        return {"enabled": True, "rows": [], "event_people_count": 0, "media_people_count": 0, "line_message_count": 0, "call_count": 0}
    media_counts = {str(row["person_id"]): int(row["media_count"] or 0) for row in media_rows}
    line_counts = {str(row["person_id"]): int(row["line_count"] or 0) for row in line_rows}
    call_counts = {str(row["person_id"]): int(row["call_count"] or 0) for row in call_rows}
    labels = {str(row["id"]): str(row["display_name"] or row["public_name"] or row["id"]) for row in person_rows}
    rows = []
    seen_ids: set[str] = set()
    for index, row in enumerate(event_rows, start=1):
        person = dict(row)
        person_id = str(person["id"])
        seen_ids.add(person_id)
        rows.append(
            {
                "person_id": person_id,
                "label": str(person.get("display_name") or person.get("public_name") or f"人物{index}"),
                "event_count": int(person.get("event_count") or 0),
                "media_count": media_counts.get(person_id, 0),
                "line_count": line_counts.get(person_id, 0),
                "call_count": call_counts.get(person_id, 0),
            }
        )
    for person_id in sorted((set(media_counts) | set(line_counts) | set(call_counts)) - seen_ids):
        rows.append(
            {
                "person_id": person_id,
                "label": labels.get(person_id, person_id),
                "event_count": 0,
                "media_count": media_counts.get(person_id, 0),
                "line_count": line_counts.get(person_id, 0),
                "call_count": call_counts.get(person_id, 0),
            }
        )
    rows.sort(key=lambda row: (-(row["event_count"] + row["media_count"] + row["line_count"] + row["call_count"]), row["label"]))
    return {
        "enabled": True,
        "rows": rows[:10],
        "event_people_count": sum(row["event_count"] for row in rows),
        "media_people_count": sum(row["media_count"] for row in rows),
        "line_message_count": sum(row["line_count"] for row in rows),
        "call_count": sum(row["call_count"] for row in rows),
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
