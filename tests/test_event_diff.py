from __future__ import annotations

from personal_lifelog_rag.timeline.event_rebuild_analysis import diff_event_snapshots


def test_event_diff_detects_title_confidence_and_evidence_changes() -> None:
    before = {
        "summary": {"event_count": 1, "event_evidence_count": 1, "ocr_evidence_count": 0, "vlm_evidence_count": 0},
        "events": [
            {
                "event_id": "event_1",
                "title": "位置情報付き写真の記録",
                "summary_short": "before",
                "confidence": 0.6,
                "location_name": None,
                "evidence_counts": {"photo": 1},
                "evidence_strength": {"vlm_only": False},
            }
        ],
    }
    after = {
        "summary": {"event_count": 1, "event_evidence_count": 3, "ocr_evidence_count": 1, "vlm_evidence_count": 1},
        "events": [
            {
                "event_id": "event_1",
                "title": "食事・カフェの可能性",
                "summary_short": "after",
                "confidence": 0.72,
                "location_name": "候補地点",
                "evidence_counts": {"photo": 1, "ocr": 1, "vlm": 1},
                "evidence_strength": {"vlm_only": False},
            }
        ],
    }

    diff = diff_event_snapshots(before, after)

    assert diff["event_count_delta"] == 0
    assert diff["event_evidence_count_delta"] == 2
    assert diff["ocr_evidence_delta"] == 1
    assert diff["vlm_evidence_delta"] == 1
    assert diff["changed_titles"][0]["after"] == "食事・カフェの可能性"
    assert diff["changed_confidence"][0]["before"] == 0.6
