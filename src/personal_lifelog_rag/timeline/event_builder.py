"""Build practical timeline events from local LINE, photo, and GPS evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import hashlib
import json
import math
import os
import re
from typing import Any, Iterable
from collections.abc import Callable

from personal_lifelog_rag.line.call_parser import parse_line_call_text
from personal_lifelog_rag.vlm.review_service import apply_vlm_override_to_result, should_use_vlm_for_events


GENERATION_METHOD = "rule_based_daily_v2"
PHOTO_GAP_MINUTES_ENV_VAR = "PERSONAL_LIFELOG_RAG_EVENT_PHOTO_GAP_MINUTES"
LINE_GAP_MINUTES_ENV_VAR = "PERSONAL_LIFELOG_RAG_EVENT_LINE_GAP_MINUTES"
MERGE_WINDOW_MINUTES_ENV_VAR = "PERSONAL_LIFELOG_RAG_EVENT_MERGE_WINDOW_MINUTES"
GPS_DISTANCE_METERS_ENV_VAR = "PERSONAL_LIFELOG_RAG_EVENT_GPS_DISTANCE_METERS"

LOCATION_WORDS = {
    "新宿",
    "渋谷",
    "池袋",
    "東京",
    "横浜",
    "大阪",
    "京都",
    "名古屋",
    "東口",
    "西口",
    "南口",
    "北口",
}
LOCATION_SUFFIX_RE = re.compile(
    r"([一-龥ァ-ヶA-Za-z0-9ー]{1,16}(?:駅|口|公園|空港|ホテル|店|カフェ|レストラン|神社|寺|港|海|山|温泉|美術館|博物館|大学|会場))"
)
WAITING_WORDS = {"駅", "着く", "待って", "待ってる", "待ち合わせ", "合流", "集合", "東口", "西口", "南口", "北口"}
FOOD_WORDS = {
    "ご飯",
    "食べ",
    "食事",
    "ランチ",
    "ラーメン",
    "カフェ",
    "店",
    "おいしかった",
    "飲み",
    "ramen_possible",
    "meal_possible",
    "cafe_possible",
}
ACTIVITY_WORDS = WAITING_WORDS | FOOD_WORDS | {
    "予約",
    "予定",
    "行く",
    "会う",
    "散歩",
    "旅行",
    "買い物",
    "仕事",
    "映画",
    "ライブ",
}
SPECIAL_MESSAGE_TYPES = {"image", "video", "sticker", "file", "system", "unknown"}


@dataclass(frozen=True)
class EventBuildConfig:
    photo_gap_minutes: int = 90
    line_gap_minutes: int = 120
    merge_window_minutes: int = 90
    gps_distance_meters: float = 500.0

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "EventBuildConfig":
        return cls(
            photo_gap_minutes=_coerce_int(values.get("photo_gap_minutes"), cls.photo_gap_minutes),
            line_gap_minutes=_coerce_int(values.get("line_gap_minutes"), cls.line_gap_minutes),
            merge_window_minutes=_coerce_int(values.get("merge_window_minutes"), cls.merge_window_minutes),
            gps_distance_meters=_coerce_float(values.get("gps_distance_meters"), cls.gps_distance_meters),
        )

    @classmethod
    def from_env(cls, base: "EventBuildConfig | None" = None) -> "EventBuildConfig":
        resolved = base or cls()
        return cls(
            photo_gap_minutes=_env_int(PHOTO_GAP_MINUTES_ENV_VAR, resolved.photo_gap_minutes),
            line_gap_minutes=_env_int(LINE_GAP_MINUTES_ENV_VAR, resolved.line_gap_minutes),
            merge_window_minutes=_env_int(MERGE_WINDOW_MINUTES_ENV_VAR, resolved.merge_window_minutes),
            gps_distance_meters=_env_float(GPS_DISTANCE_METERS_ENV_VAR, resolved.gps_distance_meters),
        )


@dataclass
class TimelineEventDraft:
    date: str
    title: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    start_time: str | None = None
    end_time: str | None = None
    summary: str | None = None
    location_name: str | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    participants: list[str] = field(default_factory=list)
    confidence: float = 0.0
    event_id: str | None = None


@dataclass
class BuildEventsReport:
    start_date: str
    end_date: str
    days_scanned: int = 0
    days_skipped: int = 0
    dry_run: bool = False
    events_planned: int = 0
    evidence_planned: int = 0
    events_created: int = 0
    evidence_saved: int = 0
    events_deleted: int = 0
    day_reports: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _EvidenceItem:
    kind: str
    record_id: str
    at: datetime
    record: dict[str, Any]
    text: str = ""
    message_type: str | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    locations: set[str] = field(default_factory=set)
    activities: set[str] = field(default_factory=set)


@dataclass
class _EventCluster:
    items: list[_EvidenceItem]

    @property
    def start(self) -> datetime:
        return min(item.at for item in self.items)

    @property
    def end(self) -> datetime:
        return max(item.at for item in self.items)

    @property
    def kinds(self) -> set[str]:
        return {item.kind for item in self.items}

    def extend(self, other: "_EventCluster") -> None:
        self.items.extend(other.items)
        self.items.sort(key=lambda item: (item.at, item.kind, item.record_id))


def build_events(
    repository,
    *,
    start_date: str,
    end_date: str | None = None,
    config: EventBuildConfig | None = None,
    dry_run: bool = False,
    skip_existing: bool = False,
    force: bool = True,
    limit_days: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> BuildEventsReport:
    """Build and persist generated event candidates for the inclusive date range."""

    resolved_end = end_date or start_date
    dates = [target.isoformat() for target in _date_range(start_date, resolved_end)]
    return _build_events_for_dates(
        repository,
        dates,
        start_date=start_date,
        end_date=resolved_end,
        config=config or EventBuildConfig.from_env(),
        dry_run=dry_run,
        skip_existing=skip_existing,
        force=force,
        limit_days=limit_days,
        progress_callback=progress_callback,
    )


def build_all_events(
    repository,
    *,
    config: EventBuildConfig | None = None,
    dry_run: bool = False,
    skip_existing: bool = False,
    force: bool = True,
    limit_days: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> BuildEventsReport:
    """Build generated event candidates for every date that has local records."""

    dates = repository.list_record_dates()
    if not dates:
        return BuildEventsReport(start_date="", end_date="", dry_run=dry_run)
    return _build_events_for_dates(
        repository,
        dates,
        start_date=dates[0],
        end_date=dates[-1],
        config=config or EventBuildConfig.from_env(),
        dry_run=dry_run,
        skip_existing=skip_existing,
        force=force,
        limit_days=limit_days,
        progress_callback=progress_callback,
    )


def build_event_drafts_for_date(
    repository,
    target_date: str,
    *,
    config: EventBuildConfig | None = None,
) -> list[TimelineEventDraft]:
    """Create event drafts from records on one local date."""

    resolved_config = config or EventBuildConfig.from_env()
    line_messages = repository.list_line_messages(
        start_date=target_date,
        end_date=target_date,
        limit=100_000,
    )
    media_items = repository.list_media_items(
        start_date=target_date,
        end_date=target_date,
        limit=100_000,
    )
    media_items = _apply_vlm_review_for_events(repository, media_items, target_date)
    photo_items = _photo_items(media_items)
    line_items = _line_items(line_messages)
    photo_clusters = _cluster_photos(photo_items, resolved_config)
    line_clusters = _cluster_lines(line_items, resolved_config)
    event_clusters = _merge_photo_and_line_clusters(photo_clusters, line_clusters, resolved_config)
    return [
        _draft_from_cluster(target_date, cluster, index)
        for index, cluster in enumerate(event_clusters)
        if cluster.items
    ]


def _apply_vlm_review_for_events(repository, media_items: list[dict[str, Any]], target_date: str) -> list[dict[str, Any]]:
    vlm_rows = repository.list_media_vlm(start_date=target_date, end_date=target_date, limit=100_000)
    vlm_by_media = {str(row.get("media_id")): apply_vlm_override_to_result(row) for row in vlm_rows}
    output: list[dict[str, Any]] = []
    for media in media_items:
        row = dict(media)
        vlm = vlm_by_media.get(str(row.get("id")))
        if vlm:
            if should_use_vlm_for_events(vlm):
                row["caption"] = vlm.get("short_caption") or vlm.get("caption") or row.get("caption")
                row["analysis_json"] = _json_or_none(
                    {
                        "short_caption": vlm.get("short_caption"),
                        "caption": vlm.get("caption"),
                        "scene_tags": _list_from_any_json(vlm.get("scene_tags_json")),
                        "object_tags": _list_from_any_json(vlm.get("object_tags_json")),
                        "activity_tags": _list_from_any_json(vlm.get("activity_tags_json")),
                        "location_cues": _list_from_any_json(vlm.get("location_cues_json")),
                        "food_cues": _list_from_any_json(vlm.get("food_cues_json")),
                        "safety_flags": _list_from_any_json(vlm.get("safety_flags_json")),
                        "review_status": vlm.get("review_status"),
                        "is_verified": bool(vlm.get("is_verified")),
                    }
                )
            else:
                row["caption"] = None
                row["analysis_json"] = None
        output.append(row)
    return output


def build_daily_event_drafts(
    *,
    media_items: list[dict[str, Any]],
    line_messages: list[dict[str, Any]],
) -> list[TimelineEventDraft]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in media_items:
        captured_at = item.get("captured_at") or item.get("fallback_captured_at")
        if captured_at:
            grouped[captured_at[:10]].append({"type": "photo", **item})
    for message in line_messages:
        sent_at = message.get("sent_at")
        if sent_at:
            grouped[sent_at[:10]].append({"type": "line", **message})

    drafts: list[TimelineEventDraft] = []
    for event_date, evidence in sorted(grouped.items()):
        photo_count = sum(1 for item in evidence if item.get("type") == "photo")
        line_count = sum(1 for item in evidence if item.get("type") == "line")
        title = f"{event_date}: photos={photo_count}, line_messages={line_count}"
        drafts.append(TimelineEventDraft(date=event_date, title=title, evidence=evidence))
    return drafts


def _build_events_for_dates(
    repository,
    dates: list[str],
    *,
    start_date: str,
    end_date: str,
    config: EventBuildConfig,
    dry_run: bool = False,
    skip_existing: bool = False,
    force: bool = True,
    limit_days: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> BuildEventsReport:
    normalized_dates = sorted({_normalize_date(value) for value in dates})
    if limit_days is not None:
        normalized_dates = normalized_dates[: max(limit_days, 0)]
    report = BuildEventsReport(
        start_date=start_date,
        end_date=end_date,
        days_scanned=len(normalized_dates),
        dry_run=dry_run,
    )
    if not normalized_dates:
        return report

    for index, target_date in enumerate(normalized_dates, start=1):
        if progress_callback is not None:
            progress_callback(f"Processing {target_date} ({index}/{len(normalized_dates)})")
        existing_events = repository.count_events(start_date=target_date, end_date=target_date)
        if skip_existing and existing_events:
            report.days_skipped += 1
            report.day_reports.append(
                {
                    "date": target_date,
                    "action": "skipped_existing",
                    "existing_events": existing_events,
                    "events": 0,
                    "evidence": 0,
                }
            )
            continue

        drafts = build_event_drafts_for_date(repository, target_date, config=config)
        planned_evidence = sum(len(draft.evidence) for draft in drafts)
        report.events_planned += len(drafts)
        report.evidence_planned += planned_evidence
        day_report = {
            "date": target_date,
            "action": "dry_run" if dry_run else "built",
            "existing_events": existing_events,
            "events": len(drafts),
            "evidence": planned_evidence,
            "titles": _title_counts(draft.title for draft in drafts),
        }
        report.day_reports.append(day_report)
        if dry_run:
            continue

        if force:
            report.events_deleted += repository.delete_generated_events(
                start_date=target_date,
                end_date=target_date,
                generation_method=GENERATION_METHOD,
                include_legacy_null=True,
            )
        for draft in drafts:
            event_id = repository.add_event(
                id=draft.event_id,
                date=draft.date,
                start_time=draft.start_time,
                end_time=draft.end_time,
                title=draft.title,
                summary=draft.summary,
                location_name=draft.location_name,
                gps_lat=draft.gps_lat,
                gps_lon=draft.gps_lon,
                participants=draft.participants,
                confidence=draft.confidence,
                source="generated",
                generation_method=GENERATION_METHOD,
                is_user_edited=False,
            )
            evidence_rows = [
                {
                    "evidence_type": item["evidence_type"],
                    "evidence_id": item["evidence_id"],
                    "weight": item["weight"],
                }
                for item in draft.evidence
            ]
            repository.replace_event_evidence(event_id, evidence_rows)
            report.events_created += 1
            report.evidence_saved += len(evidence_rows)
    return report


def _title_counts(titles: Iterable[str]) -> dict[str, int]:
    return dict(Counter(titles))


def _photo_items(media_items: list[dict[str, Any]]) -> list[_EvidenceItem]:
    items: list[_EvidenceItem] = []
    for media in media_items:
        captured_at = _parse_datetime(media.get("captured_at") or media.get("fallback_captured_at"))
        if captured_at is None:
            continue
        text = " ".join(
            str(part)
            for part in (
                media.get("caption"),
                media.get("ocr_text"),
                media.get("analysis_json"),
                media.get("file_name"),
                media.get("camera_model"),
            )
            if part
        )
        items.append(
            _EvidenceItem(
                kind="photo",
                record_id=str(media["id"]),
                at=captured_at,
                record=media,
                text=text,
                gps_lat=_float_or_none(media.get("gps_lat")),
                gps_lon=_float_or_none(media.get("gps_lon")),
                locations=_extract_locations(text),
                activities=_extract_activities(text),
            )
        )
    return sorted(items, key=lambda item: (item.at, item.record_id))


def _line_items(line_messages: list[dict[str, Any]]) -> list[_EvidenceItem]:
    items: list[_EvidenceItem] = []
    for message in line_messages:
        sent_at = _parse_datetime(message.get("sent_at"))
        if sent_at is None:
            continue
        text = message.get("text") or message.get("message_text") or ""
        message_type = message.get("message_type") or "text"
        parsed_call = parse_line_call_text(text)
        record = dict(message)
        if parsed_call is not None:
            record["call_status"] = parsed_call.call_status
            record["duration_sec"] = parsed_call.duration_sec
        items.append(
            _EvidenceItem(
                kind="line",
                record_id=str(message["id"]),
                at=sent_at,
                record=record,
                text=text,
                message_type=message_type,
                locations=_extract_locations(text),
                activities=_extract_activities(text),
            )
        )
    return sorted(items, key=lambda item: (item.at, item.record_id))


def _cluster_photos(items: list[_EvidenceItem], config: EventBuildConfig) -> list[_EventCluster]:
    clusters: list[_EventCluster] = []
    current: list[_EvidenceItem] = []
    for item in items:
        if not current:
            current = [item]
            continue

        minutes = _minutes_between(current[-1].at, item.at)
        gps_far = _has_gps(item) and _cluster_has_gps(current) and _distance_to_cluster(item, current) > config.gps_distance_meters
        if minutes >= config.photo_gap_minutes or gps_far:
            clusters.append(_EventCluster(current))
            current = [item]
        else:
            current.append(item)

    if current:
        clusters.append(_EventCluster(current))
    return clusters


def _cluster_lines(items: list[_EvidenceItem], config: EventBuildConfig) -> list[_EventCluster]:
    clusters: list[_EventCluster] = []
    current: list[_EvidenceItem] = []
    for item in items:
        if not current:
            current = [item]
            continue

        minutes = _minutes_between(current[-1].at, item.at)
        if minutes >= config.line_gap_minutes:
            clusters.append(_EventCluster(current))
            current = [item]
        else:
            current.append(item)

    if current:
        clusters.append(_EventCluster(current))
    return clusters


def _merge_photo_and_line_clusters(
    photo_clusters: list[_EventCluster],
    line_clusters: list[_EventCluster],
    config: EventBuildConfig,
) -> list[_EventCluster]:
    clusters = sorted(photo_clusters + line_clusters, key=lambda cluster: (cluster.start, cluster.end))
    merged: list[_EventCluster] = []
    for cluster in clusters:
        if merged and _should_merge(merged[-1], cluster, config):
            merged[-1].extend(cluster)
        else:
            merged.append(_EventCluster(list(cluster.items)))
    return merged


def _should_merge(left: _EventCluster, right: _EventCluster, config: EventBuildConfig) -> bool:
    if not _has_cross_modal_match(left, right):
        return False
    return _cluster_time_gap_minutes(left, right) <= config.merge_window_minutes


def _has_cross_modal_match(left: _EventCluster, right: _EventCluster) -> bool:
    return ("photo" in left.kinds and "line" in right.kinds) or ("line" in left.kinds and "photo" in right.kinds)


def _draft_from_cluster(target_date: str, cluster: _EventCluster, index: int) -> TimelineEventDraft:
    items = sorted(cluster.items, key=lambda item: (item.at, item.kind, item.record_id))
    line_items = [item for item in items if item.kind == "line"]
    photo_items = [item for item in items if item.kind == "photo"]
    locations = _ordered_values(item.locations for item in items)
    activities = _ordered_values(item.activities for item in items)
    participants = _ordered_values(
        {str(item.record.get("sender") or item.record.get("sender_name"))}
        for item in line_items
        if item.record.get("sender") or item.record.get("sender_name")
    )
    gps_lat, gps_lon = _cluster_centroid(photo_items)
    title = _event_title(line_items=line_items, photo_items=photo_items)
    summary = _event_summary(locations, activities, line_items, photo_items)
    confidence = _confidence(line_items=line_items, photo_items=photo_items, locations=locations, activities=activities)

    return TimelineEventDraft(
        date=target_date,
        start_time=cluster.start.strftime("%H:%M:%S"),
        end_time=cluster.end.strftime("%H:%M:%S"),
        title=title,
        summary=summary,
        location_name=locations[0] if locations else None,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
        participants=participants,
        confidence=confidence,
        event_id=_event_id(target_date, items, index),
        evidence=[payload for item in items for payload in _evidence_payloads(item)],
    )


def _event_title(*, line_items: list[_EvidenceItem], photo_items: list[_EvidenceItem]) -> str:
    text = "\n".join(item.text for item in [*line_items, *photo_items])
    line_count = len(line_items)
    photo_count = len(photo_items)
    gps_count = sum(1 for item in photo_items if _has_gps(item))
    call_count = _completed_call_count(line_items)
    incomplete_call_count = _incomplete_call_count(line_items)

    if _has_waiting_terms(text):
        return "移動・待ち合わせの可能性"
    if _has_food_terms(text):
        return "食事・カフェの可能性"
    if call_count >= 2 or (line_count >= 3 and call_count >= max(1, line_count // 2)):
        return "通話・連絡"
    if incomplete_call_count and not call_count:
        return "不在着信・通話未成立"
    if photo_count >= 3 and gps_count:
        return "位置情報付き写真の記録"
    if photo_count >= 3:
        return "写真撮影の記録"
    if photo_count and line_count:
        return "写真とLINEが残る出来事"
    if photo_count:
        return "写真撮影の記録"
    if line_count:
        return "LINEのやりとり"
    return "記録された出来事"


def _event_summary(
    locations: list[str],
    activities: list[str],
    line_items: list[_EvidenceItem],
    photo_items: list[_EvidenceItem],
) -> str:
    gps_count = sum(1 for item in photo_items if _has_gps(item))
    type_counts = Counter(item.message_type or "text" for item in line_items)
    completed_call_count = _completed_call_count(line_items)
    incomplete_call_count = _incomplete_call_count(line_items)
    total_call_duration = sum(int(item.record.get("duration_sec") or 0) for item in line_items if item.record.get("call_status") == "completed")
    parts = [f"LINE {len(line_items)}件、写真 {len(photo_items)}枚から作成したイベント候補です。"]
    if gps_count:
        parts.append(f"GPS付き写真が{gps_count}枚あります。")
    if completed_call_count:
        parts.append(f"completed callが{completed_call_count}件あります。")
        if total_call_duration:
            minutes = max(1, round(total_call_duration / 60))
            parts.append(f"通話時間合計は約{minutes}分です。")
    elif incomplete_call_count:
        parts.append(f"不在着信・未応答・キャンセルなどの通話未成立ログが{incomplete_call_count}件あります。")
    if locations:
        parts.append("場所候補: " + "、".join(locations[:3]) + "。")
    if activities:
        parts.append("活動候補: " + "、".join(activities[:3]) + "。")
    special_types = [key for key in ("image", "video", "sticker", "file", "system", "unknown") if type_counts.get(key)]
    if special_types:
        parts.append("特殊メッセージ: " + "、".join(f"{key}={type_counts[key]}" for key in special_types) + "。")
    if any((item.record.get("caption") or item.record.get("ocr_text") or item.record.get("analysis_json")) for item in photo_items):
        parts.append("写真のcaption/OCR/VLMテキストも弱い根拠に含めています。")
    ocr_values = _photo_ocr_cues(photo_items)
    vlm_values = _photo_vlm_cues(photo_items)
    if ocr_values:
        parts.append("OCR候補: " + "、".join(ocr_values[:3]) + "。")
    if vlm_values:
        parts.append("画像解析による推定: " + "、".join(vlm_values[:3]) + "。")
        parts.append("VLMのみの推定は弱い補助根拠として扱います。")
    return " ".join(parts)


def _confidence(
    *,
    line_items: list[_EvidenceItem],
    photo_items: list[_EvidenceItem],
    locations: list[str],
    activities: list[str],
) -> float:
    evidence_count = len(line_items) + len(photo_items)
    gps_count = sum(1 for item in photo_items if _has_gps(item))
    text_rich_photo_count = sum(
        1 for item in photo_items if item.record.get("caption") or item.record.get("ocr_text") or item.record.get("analysis_json")
    )
    ocr_count = sum(1 for item in photo_items if _has_ocr_text(item))
    vlm_count = sum(1 for item in photo_items if _has_vlm_text(item))

    score = 0.35
    if evidence_count >= 2:
        score += 0.12
    if evidence_count >= 5:
        score += 0.08
    if line_items and photo_items:
        score += 0.15
    if gps_count:
        score += 0.12
    if gps_count >= 3:
        score += 0.04
    if locations:
        score += 0.06
    if activities:
        score += 0.06
    if text_rich_photo_count:
        score += 0.04
    if ocr_count and vlm_count and line_items:
        score += 0.04
    if vlm_count and not ocr_count and not line_items:
        score = min(score, 0.74)
    return round(min(score, 0.95), 2)


def _evidence_payloads(item: _EvidenceItem) -> list[dict[str, Any]]:
    rows = [
        {
            "evidence_type": item.kind,
            "evidence_id": item.record_id,
            "weight": _evidence_weight(item),
        }
    ]
    if item.kind == "photo" and _has_ocr_text(item):
        rows.append(
            {
                "evidence_type": "ocr",
                "evidence_id": item.record_id,
                "weight": 0.72,
            }
        )
    if item.kind == "photo" and _has_vlm_text(item):
        rows.append(
            {
                "evidence_type": "vlm",
                "evidence_id": item.record_id,
                "weight": 0.45,
            }
        )
    return rows


def _evidence_weight(item: _EvidenceItem) -> float:
    weight = 0.65 if item.kind == "line" else 0.6
    if item.kind == "photo" and _has_gps(item):
        weight += 0.15
    if item.locations or item.activities:
        weight += 0.1
    if item.kind == "photo" and (item.record.get("caption") or item.record.get("ocr_text") or item.record.get("analysis_json")):
        weight += 0.1
    if item.message_type in SPECIAL_MESSAGE_TYPES:
        weight += 0.03
    return round(min(weight, 1.0), 2)


def _has_ocr_text(item: _EvidenceItem) -> bool:
    return bool(str(item.record.get("ocr_text") or "").strip())


def _has_vlm_text(item: _EvidenceItem) -> bool:
    if str(item.record.get("caption") or "").strip():
        return True
    analysis = _analysis_dict(item.record.get("analysis_json"))
    return any(
        analysis.get(key)
        for key in (
            "short_caption",
            "scene_tags",
            "object_tags",
            "activity_tags",
            "location_cues",
            "food_cues",
            "text_cues",
        )
    )


def _photo_ocr_cues(photo_items: list[_EvidenceItem]) -> list[str]:
    values: list[str] = []
    for item in photo_items:
        text = str(item.record.get("ocr_text") or "").strip()
        for value in sorted(_extract_locations(text) | _extract_activities(text)):
            if value not in values:
                values.append(value)
    return values


def _photo_vlm_cues(photo_items: list[_EvidenceItem]) -> list[str]:
    values: list[str] = []
    for item in photo_items:
        analysis = _analysis_dict(item.record.get("analysis_json"))
        raw_values: list[Any] = []
        raw_values.extend(_list_from_any(analysis.get("food_cues")))
        raw_values.extend(_list_from_any(analysis.get("location_cues")))
        raw_values.extend(_list_from_any(analysis.get("activity_tags")))
        raw_values.extend(_list_from_any(analysis.get("scene_tags")))
        for raw in raw_values:
            value = _humanize_vlm_cue(str(raw))
            if value and value not in values:
                values.append(value)
    return values


def _analysis_dict(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_or_none(value: Any | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _list_from_any_json(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return [part.strip() for part in str(value).split(",") if part.strip()]
    return parsed if isinstance(parsed, list) else []


def _list_from_any(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _humanize_vlm_cue(value: str) -> str:
    mapping = {
        "ramen_possible": "ラーメンの可能性",
        "meal_possible": "料理・食事の可能性",
        "cafe_possible": "カフェの可能性",
        "restaurant": "飲食店のような場所の可能性",
        "indoor": "屋内の可能性",
        "outdoor": "屋外の可能性",
        "station_possible": "駅周辺の可能性",
        "shop_possible": "店内または店舗周辺の可能性",
    }
    return mapping.get(value, value if value.endswith("可能性") else value.replace("_", " "))


def _extract_locations(text: str) -> set[str]:
    found = {word for word in LOCATION_WORDS if word in text}
    found.update(match.group(1) for match in LOCATION_SUFFIX_RE.finditer(text))
    return {value.strip() for value in found if value.strip()}


def _extract_activities(text: str) -> set[str]:
    found = {word for word in ACTIVITY_WORDS if word in text}
    if "待って" in text or "待つ" in text:
        found.add("待ち合わせ")
    if "食べ" in text:
        found.add("食事")
    return found


def _has_waiting_terms(text: str) -> bool:
    return any(word in text for word in WAITING_WORDS)


def _has_food_terms(text: str) -> bool:
    return any(word in text for word in FOOD_WORDS)


def _completed_call_count(items: list[_EvidenceItem]) -> int:
    return sum(
        1
        for item in items
        if item.record.get("call_status") == "completed"
        or (item.message_type == "call" and "通話時間" in item.text)
        or "通話時間" in item.text
    )


def _incomplete_call_count(items: list[_EvidenceItem]) -> int:
    return sum(
        1
        for item in items
        if item.record.get("call_status") in {"missed", "unanswered", "canceled"}
        or any(word in item.text for word in ("不在着信", "応答がありませんでした", "キャンセル"))
    )


def _cluster_centroid(items: list[_EvidenceItem]) -> tuple[float | None, float | None]:
    gps_items = [item for item in items if _has_gps(item)]
    if not gps_items:
        return None, None
    lat = sum(item.gps_lat or 0.0 for item in gps_items) / len(gps_items)
    lon = sum(item.gps_lon or 0.0 for item in gps_items) / len(gps_items)
    return round(lat, 7), round(lon, 7)


def _distance_to_cluster(item: _EvidenceItem, cluster: list[_EvidenceItem]) -> float:
    distances = [_distance_meters(item, other) for other in cluster if _has_gps(other)]
    return min(distances) if distances else math.inf


def _distance_meters(left: _EvidenceItem, right: _EvidenceItem) -> float:
    if not _has_gps(left) or not _has_gps(right):
        return math.inf
    radius = 6_371_000.0
    lat1 = math.radians(left.gps_lat or 0.0)
    lat2 = math.radians(right.gps_lat or 0.0)
    delta_lat = math.radians((right.gps_lat or 0.0) - (left.gps_lat or 0.0))
    delta_lon = math.radians((right.gps_lon or 0.0) - (left.gps_lon or 0.0))
    haversine = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(haversine), math.sqrt(1 - haversine))


def _has_gps(item: _EvidenceItem) -> bool:
    return item.gps_lat is not None and item.gps_lon is not None


def _cluster_has_gps(items: list[_EvidenceItem]) -> bool:
    return any(_has_gps(item) for item in items)


def _minutes_between(left: datetime, right: datetime) -> float:
    return abs((right - left).total_seconds()) / 60.0


def _cluster_time_gap_minutes(left: _EventCluster, right: _EventCluster) -> float:
    if left.end >= right.start and right.end >= left.start:
        return 0.0
    if left.end < right.start:
        return _minutes_between(left.end, right.start)
    return _minutes_between(right.end, left.start)


def _ordered_values(values: Iterable[set[str]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value_set in values:
        for value in sorted(value_set):
            if value and value not in seen:
                ordered.append(value)
                seen.add(value)
    return ordered


def _event_id(target_date: str, items: list[_EvidenceItem], index: int) -> str:
    first_at = items[0].at.strftime("%H:%M") if items else "00:00"
    last_at = items[-1].at.strftime("%H:%M") if items else "00:00"
    evidence_signature = ",".join(item.record_id for item in items[:5])
    signature = "|".join([GENERATION_METHOD, target_date, first_at, last_at, str(index), evidence_signature])
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    return f"event_{digest}"


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).replace(tzinfo=None)
    except ValueError:
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_date(value: str) -> str:
    return date.fromisoformat(value).isoformat()


def _date_range(start_date: str, end_date: str) -> list[date]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < start:
        raise ValueError("--to must be on or after --from")
    days = (end - start).days
    return [start + timedelta(days=offset) for offset in range(days + 1)]


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return _coerce_int(value, default)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return _coerce_float(value, default)


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
