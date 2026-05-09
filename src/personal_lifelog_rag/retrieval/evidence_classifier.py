"""Classify local search evidence as action, plan, mention, or unknown."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from personal_lifelog_rag.retrieval.query_intent import QueryIntent


EvidenceClassification = Literal[
    "actual_or_likely_action",
    "plan_or_candidate",
    "mention_only",
    "unknown",
]

ACTUAL_PLACE_TERMS = (
    "着いた",
    "着く",
    "到着",
    "いる",
    "来た",
    "向かってる",
    "駅着いた",
    "改札",
    "待ってる",
    "待って",
    "合流",
)
ACTUAL_FOOD_TERMS = (
    "食べた",
    "食べる",
    "食べます",
    "おいしかった",
    "ご飯食べ",
    "夜ご飯",
    "昼ご飯",
    "カフェ",
    "店",
)
ACTUAL_CALL_TERMS = ("通話時間", "LINE通話", "電話した")
PLAN_TERMS = (
    "行く？",
    "行こう",
    "行きたい",
    "行くかも",
    "行けたら",
    "行くなら",
    "着いたら",
    "着くなら",
    "候補",
    "どっか",
    "じゃないかな",
    "予定",
    "予約",
    "どうする",
    "どうしよ",
    "見とく",
    "探す",
    "あるとしたら",
    "かも",
    "かもしれない",
    "かな",
    "かなー",
    "何食べたい",
    "ご飯どうしよ",
    "どこ行く",
)
CONDITIONAL_ACTUAL_PATTERNS = (
    "着いたら",
    "着くなら",
    "行けたら",
    "行くなら",
    "あるとしたら",
    "かも",
    "かな",
    "どっか",
    "候補",
    "予定",
)
MISSED_CALL_TERMS = ("応答がありませんでした", "不在着信", "不在", "キャンセル")

ACTUAL_EVENT_TITLES = (
    "移動・待ち合わせの可能性",
    "食事・カフェの可能性",
    "通話・連絡",
    "写真とLINEが残る出来事",
    "位置情報付き写真の記録",
)


@dataclass(frozen=True)
class ClassifiedEvidence:
    classification: EvidenceClassification
    reason_parts: list[str]
    score_components: dict[str, float] = field(default_factory=dict)


def classify_day_evidence(
    *,
    query: str,
    intent: QueryIntent,
    day_result: dict[str, Any],
    same_day_media_items: list[dict[str, Any]] | None = None,
) -> ClassifiedEvidence:
    """Classify one day of search evidence with conservative local rules."""

    same_day_media_items = same_day_media_items or []
    text_blob = _day_text(day_result)
    events = list(day_result.get("events") or [])
    line_samples = list(day_result.get("line_samples") or [])
    ocr_samples = list(day_result.get("ocr_samples") or [])
    vlm_samples = list(day_result.get("vlm_samples") or [])
    matching_media_count = int(day_result.get("media_match_count") or 0)
    ocr_match_count = int(day_result.get("ocr_match_count") or 0)
    vlm_match_count = int(day_result.get("vlm_match_count") or 0)
    same_day_photo_count = len(same_day_media_items)
    same_day_gps_count = sum(1 for item in same_day_media_items if _has_gps(item))
    call_summary = dict(day_result.get("call_summary") or {})
    completed_calls = int(call_summary.get("completed") or 0)
    incomplete_calls = sum(
        int(call_summary.get(key) or 0)
        for key in ("missed", "unanswered", "canceled")
    )
    max_call_duration = int(call_summary.get("max_duration_sec") or 0)

    actual_hits = _filter_conditional_actual_hits(
        _matched_terms(text_blob, _actual_terms_for_intent(intent)),
        text_blob,
    )
    plan_hits = _matched_terms(text_blob, PLAN_TERMS)
    missed_call_hits = _matched_terms(text_blob, MISSED_CALL_TERMS)
    event_actual = [
        event
        for event in events
        if _has_actual_event_signal(event=event, intent=intent, query=query)
    ]

    components = {
        "base": 0.12,
        "event_match": min(len(events), 3) * 0.10,
        "event_confidence": min(
            max((_float_or_none(event.get("confidence")) or 0.0 for event in events), default=0.0),
            0.95,
        ) * 0.20,
        "actual_terms": min(len(actual_hits), 4) * 0.15,
        "plan_terms_penalty": -min(len(plan_hits), 4) * 0.12,
        "missed_call_penalty": -min(len(missed_call_hits), 3) * 0.20,
        "same_day_photos": min(same_day_photo_count, 5) * 0.035,
        "same_day_gps": min(same_day_gps_count, 5) * 0.045,
        "matching_media": min(matching_media_count, 3) * 0.05,
        "ocr_match": min(ocr_match_count, 5) * 0.08,
        "vlm_match": min(vlm_match_count, 5) * 0.10,
        "multi_evidence": _multi_evidence_bonus(day_result, same_day_photo_count),
        "single_line_penalty": -0.10 if int(day_result.get("line_match_count") or 0) == 1 and not events and not same_day_photo_count else 0.0,
        "mention_penalty": 0.0,
        "completed_calls": 0.0,
        "long_call_duration": 0.0,
        "incomplete_calls_penalty": 0.0,
    }

    if intent == "call_activity":
        if completed_calls:
            components["completed_calls"] = min(completed_calls, 4) * 0.18
            components["long_call_duration"] = min(max_call_duration / 3600.0, 1.0) * 0.16
        if incomplete_calls and not completed_calls:
            components["incomplete_calls_penalty"] = -min(incomplete_calls, 3) * 0.16
        if actual_hits and not missed_call_hits:
            components["completed_call"] = 0.20
        elif missed_call_hits and not actual_hits and not completed_calls:
            components["missed_only_penalty"] = -0.18

    if event_actual:
        components["actual_event_title"] = 0.16

    raw_score = sum(components.values())
    classification = _classification(
        actual_hits=actual_hits,
        plan_hits=plan_hits,
        missed_call_hits=missed_call_hits,
        event_actual=bool(event_actual),
        same_day_photo_count=same_day_photo_count,
        same_day_gps_count=same_day_gps_count,
        line_samples=line_samples,
        ocr_samples=ocr_samples,
        vlm_samples=vlm_samples,
        intent=intent,
        completed_calls=completed_calls,
        incomplete_calls=incomplete_calls,
    )
    if classification == "mention_only":
        components["mention_penalty"] = -0.15
        raw_score -= 0.15
    elif classification == "plan_or_candidate":
        raw_score -= 0.05
    elif classification == "actual_or_likely_action":
        raw_score += 0.10
        components["classification_bonus"] = 0.10

    score = round(max(0.0, min(raw_score, 0.98)), 3)
    components["final_score"] = score
    return ClassifiedEvidence(
        classification=classification,
        reason_parts=_reason_parts(
            actual_hits=actual_hits,
            plan_hits=plan_hits,
            missed_call_hits=missed_call_hits,
            events=events,
            event_actual=bool(event_actual),
            same_day_photo_count=same_day_photo_count,
            same_day_gps_count=same_day_gps_count,
            ocr_match_count=ocr_match_count,
            vlm_match_count=vlm_match_count,
            classification=classification,
            call_summary=call_summary,
        ),
        score_components={key: round(value, 3) for key, value in components.items()},
    )


def _actual_terms_for_intent(intent: QueryIntent) -> tuple[str, ...]:
    if intent == "food_activity":
        return ACTUAL_FOOD_TERMS
    if intent == "call_activity":
        return ACTUAL_CALL_TERMS
    if intent == "place_visit":
        return ACTUAL_PLACE_TERMS
    return ACTUAL_PLACE_TERMS + ACTUAL_FOOD_TERMS + ACTUAL_CALL_TERMS


def _actual_event_titles_for_intent(intent: QueryIntent) -> tuple[str, ...]:
    if intent == "food_activity":
        return ("食事・カフェの可能性",)
    if intent == "call_activity":
        return ("通話・連絡",)
    if intent == "place_visit":
        return (
            "移動・待ち合わせの可能性",
            "外出・写真撮影",
            "位置情報付き写真の記録",
            "写真とLINEが残る出来事",
        )
    return ACTUAL_EVENT_TITLES


def _has_actual_event_signal(*, event: dict[str, Any], intent: QueryIntent, query: str) -> bool:
    title = str(event.get("title") or "")
    if any(term in title for term in _actual_event_titles_for_intent(intent)):
        return True
    location_name = str(event.get("location_name") or "")
    return bool(query and location_name and query in location_name)


def _classification(
    *,
    actual_hits: list[str],
    plan_hits: list[str],
    missed_call_hits: list[str],
    event_actual: bool,
    same_day_photo_count: int,
    same_day_gps_count: int,
    line_samples: list[dict[str, Any]],
    ocr_samples: list[dict[str, Any]],
    vlm_samples: list[dict[str, Any]],
    intent: QueryIntent,
    completed_calls: int = 0,
    incomplete_calls: int = 0,
) -> EvidenceClassification:
    if intent == "call_activity" and completed_calls:
        return "actual_or_likely_action"
    if intent == "call_activity" and incomplete_calls and not actual_hits and not event_actual:
        return "mention_only"
    if intent == "call_activity" and missed_call_hits and not actual_hits and not event_actual:
        return "mention_only"
    if actual_hits or event_actual:
        if plan_hits and len(plan_hits) > len(actual_hits) + (1 if event_actual else 0):
            return "plan_or_candidate"
        return "actual_or_likely_action"
    if intent in {"place_visit", "food_activity"} and (same_day_gps_count or same_day_photo_count >= 2 or ocr_samples or vlm_samples) and not plan_hits:
        return "actual_or_likely_action"
    if plan_hits:
        return "plan_or_candidate"
    if line_samples:
        return "mention_only"
    return "unknown"


def _reason_parts(
    *,
    actual_hits: list[str],
    plan_hits: list[str],
    missed_call_hits: list[str],
    events: list[dict[str, Any]],
    event_actual: bool,
    same_day_photo_count: int,
    same_day_gps_count: int,
    ocr_match_count: int,
    vlm_match_count: int,
    classification: EvidenceClassification,
    call_summary: dict[str, Any] | None = None,
) -> list[str]:
    reasons: list[str] = []
    call_summary = call_summary or {}
    if event_actual:
        reasons.append("関連eventあり")
    elif events:
        reasons.append("event一致")
    if actual_hits:
        reasons.append("actual語: " + "、".join(actual_hits[:3]))
    if plan_hits:
        reasons.append("候補・予定語: " + "、".join(plan_hits[:3]))
    if missed_call_hits:
        reasons.append("未完了通話語: " + "、".join(missed_call_hits[:2]))
    if same_day_photo_count:
        reasons.append(f"同日写真{same_day_photo_count}枚")
    if same_day_gps_count:
        reasons.append(f"GPS付き写真{same_day_gps_count}枚")
    if ocr_match_count:
        reasons.append(f"OCR一致{ocr_match_count}件")
    if vlm_match_count:
        reasons.append(f"画像解析一致{vlm_match_count}件")
    if call_summary:
        completed = int(call_summary.get("completed") or 0)
        incomplete = sum(int(call_summary.get(key) or 0) for key in ("missed", "unanswered", "canceled"))
        duration = int(call_summary.get("total_duration_sec") or 0)
        if completed:
            reasons.append(f"completed call {completed}件")
        if incomplete and not completed:
            reasons.append(f"未成立通話{incomplete}件")
        if duration:
            reasons.append(f"通話合計{duration}秒")
    if not reasons:
        reasons.append("単発の言及のみ" if classification == "mention_only" else "根拠が少なく判定不能")
    return reasons


def _day_text(day_result: dict[str, Any]) -> str:
    parts: list[str] = []
    for event in day_result.get("events") or []:
        parts.append(_event_text(event))
    for sample in day_result.get("line_samples") or []:
        parts.append(str(sample.get("text") or ""))
    for sample in day_result.get("ocr_samples") or []:
        parts.append(str(sample.get("text") or ""))
    for sample in day_result.get("vlm_samples") or []:
        parts.append(str(sample.get("caption") or ""))
        parts.append(str(sample.get("scene_tags") or ""))
        parts.append(str(sample.get("object_tags") or ""))
        parts.append(str(sample.get("activity_tags") or ""))
        parts.append(str(sample.get("food_cues") or ""))
    return "\n".join(parts)


def _event_text(event: dict[str, Any]) -> str:
    text = "\n".join(
        str(event.get(key) or "")
        for key in ("title", "summary_preview", "location_name")
    )
    # Generated event summaries often use "...候補" as metadata.
    # Treat those as neutral labels so they do not look like plan/candidate talk.
    return (
        text.replace("イベント候補", "イベント")
        .replace("出来事候補", "出来事")
        .replace("場所候補", "場所")
        .replace("活動候補", "活動")
    )


def _matched_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    seen: list[str] = []
    for term in terms:
        if term and term in text and term not in seen:
            seen.append(term)
    return seen


def _filter_conditional_actual_hits(actual_hits: list[str], text: str) -> list[str]:
    """Avoid treating conditional place talk like "着いたら" as arrival evidence."""

    if not actual_hits or not any(pattern in text for pattern in CONDITIONAL_ACTUAL_PATTERNS):
        return actual_hits
    filtered: list[str] = []
    for hit in actual_hits:
        if hit == "着いた" and "着いたら" in text:
            continue
        if hit == "着く" and any(pattern in text for pattern in ("着いたら", "着くなら")):
            continue
        if hit in {"いる", "来た", "待って", "待ってる", "向かってる"} and any(
            pattern in text for pattern in ("かも", "かな", "どっか", "候補", "予定", "行けたら", "行くなら")
        ):
            continue
        filtered.append(hit)
    return filtered


def _multi_evidence_bonus(day_result: dict[str, Any], same_day_photo_count: int) -> float:
    count = 0
    if int(day_result.get("event_count") or 0):
        count += 1
    if int(day_result.get("line_match_count") or 0):
        count += 1
    if (
        same_day_photo_count
        or int(day_result.get("media_match_count") or 0)
        or int(day_result.get("ocr_match_count") or 0)
        or int(day_result.get("vlm_match_count") or 0)
    ):
        count += 1
    return 0.12 if count >= 3 else (0.06 if count >= 2 else 0.0)


def _has_gps(item: dict[str, Any]) -> bool:
    return item.get("gps_lat") is not None and item.get("gps_lon") is not None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
