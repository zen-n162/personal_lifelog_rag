"""Ranking helpers for classified local search results."""

from __future__ import annotations

from typing import Any, Literal

from personal_lifelog_rag.line.call_index import summarize_call_events
from personal_lifelog_rag.retrieval.evidence_classifier import classify_day_evidence
from personal_lifelog_rag.retrieval.query_intent import QueryIntent


SearchMode = Literal["all", "actual", "plan", "mention"]

MODE_TO_CLASSIFICATION = {
    "actual": "actual_or_likely_action",
    "plan": "plan_or_candidate",
    "mention": "mention_only",
}

SECTION_TITLES = {
    "actual_or_likely_action": "実際に行動した可能性が高い日",
    "plan_or_candidate": "予定・相談として話題に出た日",
    "mention_only": "単なる言及",
    "unknown": "判定不能",
}

CLASSIFICATION_ORDER = {
    "actual_or_likely_action": 0,
    "plan_or_candidate": 1,
    "mention_only": 2,
    "unknown": 3,
}


def rank_search_results(
    repository,
    *,
    query: str,
    intent: QueryIntent,
    results: list[dict[str, Any]],
    mode: SearchMode = "all",
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for result in results:
        date = str(result.get("date") or "")
        same_day_media = repository.list_media_items(
            start_date=date,
            end_date=date,
            limit=100_000,
        ) if date else []
        same_day_calls = repository.list_line_call_events(
            start_date=date,
            end_date=date,
            limit=100_000,
        ) if date and intent == "call_activity" else []
        classify_input = dict(result)
        if same_day_calls:
            classify_input["call_summary"] = summarize_call_events(same_day_calls)
        classified = classify_day_evidence(
            query=query,
            intent=intent,
            day_result=classify_input,
            same_day_media_items=same_day_media,
        )
        row = dict(result)
        if same_day_calls:
            row["call_summary"] = classify_input["call_summary"]
        row["classification"] = classified.classification
        row["classification_label"] = SECTION_TITLES[classified.classification]
        row["reason"] = "、".join(classified.reason_parts)
        row["reason_parts"] = classified.reason_parts
        score_components = dict(classified.score_components)
        pinned_boost = _pinned_boost(row)
        verified_boost = _verified_boost(row)
        if pinned_boost:
            score_components["pinned_boost"] = pinned_boost
        if verified_boost:
            score_components["verified_boost"] = verified_boost
        score_components["final_score"] = round(
            min(0.95, float(score_components["final_score"]) + pinned_boost + verified_boost),
            3,
        )
        row["score_components"] = score_components
        row["ranking_score"] = score_components["final_score"]
        row["score"] = row["ranking_score"]
        row["confidence"] = _confidence_from_score(row["ranking_score"])
        row["confidence_label"] = _confidence_label(row["confidence"])
        row["same_day_photo_count"] = len(same_day_media)
        row["same_day_gps_photo_count"] = sum(1 for item in same_day_media if item.get("gps_lat") is not None and item.get("gps_lon") is not None)
        ranked.append(row)

    if mode != "all":
        expected = MODE_TO_CLASSIFICATION[mode]
        ranked = [row for row in ranked if row["classification"] == expected]

    ranked.sort(
        key=lambda row: (
            CLASSIFICATION_ORDER.get(str(row.get("classification")), 99),
            float(row.get("ranking_score") or 0.0),
            str(row.get("date") or ""),
        ),
        reverse=False,
    )
    # Within each classification, higher score and newer date should appear first.
    ranked.sort(
        key=lambda row: (
            CLASSIFICATION_ORDER.get(str(row.get("classification")), 99),
            -float(row.get("ranking_score") or 0.0),
            str(row.get("date") or ""),
        )
    )
    return ranked


def _pinned_boost(row: dict[str, Any]) -> float:
    return 0.12 if any(event.get("is_pinned") for event in row.get("events") or []) else 0.0


def _verified_boost(row: dict[str, Any]) -> float:
    return 0.04 if any(event.get("is_verified") for event in row.get("events") or []) else 0.0


def _confidence_from_score(score: float) -> float:
    return round(max(0.0, min(score, 0.95)), 2)


def _confidence_label(value: float) -> str:
    if value >= 0.75:
        return "高"
    if value >= 0.45:
        return "中"
    return "低"
