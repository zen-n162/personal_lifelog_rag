from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.timeline.event_builder import build_events


def test_ocr_and_vlm_evidence_are_saved_as_auxiliary_evidence(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_media_item(
        id="media_ocr_vlm",
        file_path="/local/photo.jpg",
        file_name="photo.jpg",
        file_hash="hash-ocr-vlm",
        media_type="image",
        captured_at="2024-12-24T12:00:00+09:00",
        gps_lat=35.0,
        gps_lon=139.0,
        ocr_text="新宿 ラーメン",
        caption="ラーメンやご飯の可能性がある料理写真",
        analysis_json={"food_cues": ["ramen_possible"], "scene_tags": ["restaurant"]},
    )

    report = build_events(repository, start_date="2024-12-24", force=True)
    events = repository.list_events(start_date="2024-12-24", end_date="2024-12-24", include_hidden=True)
    evidence = repository.list_event_evidence(events[0]["id"])

    assert report.events_created == 1
    assert {row["evidence_type"] for row in evidence} == {"photo", "ocr", "vlm"}
    assert "食事・カフェの可能性" == events[0]["title"]
    assert "OCR候補" in events[0]["summary"]
    assert "画像解析による推定" in events[0]["summary"]


def test_vlm_only_photo_cluster_does_not_become_high_confidence(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    for index in range(3):
        repository.add_media_item(
            id=f"media_vlm_only_{index}",
            file_path=f"/local/vlm_only_{index}.jpg",
            file_name=f"vlm_only_{index}.jpg",
            file_hash=f"hash-vlm-only-{index}",
            media_type="image",
            captured_at=f"2024-12-24T12:0{index}:00+09:00",
            gps_lat=35.0,
            gps_lon=139.0,
            caption="ラーメンやご飯の可能性がある料理写真",
            analysis_json={"food_cues": ["ramen_possible"], "activity_tags": ["meal_possible"]},
        )

    build_events(repository, start_date="2024-12-24", force=True)
    event = repository.list_events(start_date="2024-12-24", end_date="2024-12-24", include_hidden=True)[0]

    assert event["title"] == "食事・カフェの可能性"
    assert float(event["confidence"]) < 0.8
