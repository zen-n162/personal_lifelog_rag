from __future__ import annotations

from personal_lifelog_rag.retrieval.evidence_classifier import classify_day_evidence


def test_evidence_classifier_actual_place_action() -> None:
    result = _day_result("新宿着いた！")

    classified = classify_day_evidence(query="新宿", intent="place_visit", day_result=result)

    assert classified.classification == "actual_or_likely_action"
    assert "actual語" in "、".join(classified.reason_parts)


def test_evidence_classifier_plan_candidate() -> None:
    result = _day_result("新宿かどっか行くかも")

    classified = classify_day_evidence(query="新宿", intent="place_visit", day_result=result)

    assert classified.classification == "plan_or_candidate"


def test_evidence_classifier_conditional_arrival_is_not_actual() -> None:
    result = _day_result("新宿着いたら連絡して")

    classified = classify_day_evidence(query="新宿", intent="place_visit", day_result=result)

    assert classified.classification == "plan_or_candidate"
    assert "actual語" not in "、".join(classified.reason_parts)


def test_evidence_classifier_mention_only() -> None:
    result = _day_result("新宿ってすごい")

    classified = classify_day_evidence(query="新宿", intent="place_visit", day_result=result)

    assert classified.classification == "mention_only"


def test_event_summary_keyword_alone_does_not_force_actual() -> None:
    result = {
        "date": "2024-12-24",
        "event_count": 1,
        "line_match_count": 1,
        "media_match_count": 0,
        "events": [
            {
                "title": "食事・カフェの可能性",
                "summary_preview": "場所候補: 渋谷、新宿。活動候補: 食べ。",
                "location_name": "",
            }
        ],
        "line_samples": [{"time": "17:30", "sender": "自分", "text": "新宿にもある"}],
        "evidence_types": ["events", "line"],
    }

    classified = classify_day_evidence(query="新宿", intent="place_visit", day_result=result)

    assert classified.classification == "mention_only"


def test_evidence_classifier_completed_call_beats_missed_call_score() -> None:
    completed = classify_day_evidence(query="通話", intent="call_activity", day_result=_day_result("☎ 通話時間 48:03"))
    missed = classify_day_evidence(query="通話", intent="call_activity", day_result=_day_result("☎ 通話に応答がありませんでした"))

    assert completed.classification == "actual_or_likely_action"
    assert missed.classification == "mention_only"
    assert completed.score_components["final_score"] > missed.score_components["final_score"]


def _day_result(text: str) -> dict[str, object]:
    return {
        "date": "2024-12-24",
        "event_count": 0,
        "line_match_count": 1,
        "media_match_count": 0,
        "events": [],
        "line_samples": [{"time": "17:30", "sender": "自分", "text": text}],
        "evidence_types": ["line"],
    }
