from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.places.redaction import format_place_display_preview, place_display_preview
from personal_lifelog_rag.places.schemas import Place
from personal_lifelog_rag.retrieval.answer_builder import build_answer
from personal_lifelog_rag.retrieval.date_parser import parse_date_query
from personal_lifelog_rag.retrieval.temporal_search import search_timeline
from personal_lifelog_rag.timeline.event_reports import format_event_list, list_events_report


def test_sensitive_place_hides_exact_coordinates() -> None:
    rows = place_display_preview(
        [
            Place(
                id="candidate_place_001",
                name="candidate_place_001",
                display_name="候補地点001",
                lat=35.123456,
                lon=139.654321,
                radius_m=500,
                category="unknown",
                privacy_level="sensitive",
                show_exact_location=False,
            )
        ]
    )
    output = format_place_display_preview(rows)

    assert rows[0]["exact_coordinate_display"] == "hidden"
    assert rows[0]["coordinate_display"] == "非表示"
    assert "35.123456" not in output
    assert "139.654321" not in output


def test_normal_place_without_exact_display_is_rounded() -> None:
    rows = place_display_preview(
        [
            Place(
                id="station",
                name="station",
                display_name="駅周辺",
                lat=35.123456,
                lon=139.654321,
                radius_m=800,
                category="station",
                privacy_level="normal",
                show_exact_location=False,
            )
        ]
    )

    assert rows[0]["exact_coordinate_display"] == "hidden"
    assert rows[0]["coordinate_display"] == "35.123, 139.654"


def test_answer_and_event_list_show_location_name_without_coordinates(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_event(
        id="event_sensitive_location",
        date="2024-12-24",
        start_time="10:00:00",
        end_time="10:30:00",
        title="位置情報付き写真の記録",
        summary="場所候補だけを表示するテスト",
        location_name="候補地点001",
        gps_lat=35.123456,
        gps_lon=139.654321,
        confidence=0.8,
    )

    result = search_timeline(repository, "2024年12月24日は何していた？", date_range=parse_date_query("2024年12月24日"))
    answer = build_answer("2024年12月24日は何していた？", result)
    event_list = format_event_list(list_events_report(repository, start_date="2024-12-24", end_date="2024-12-24"))

    assert "候補地点001" in answer
    assert "候補地点001" in event_list
    assert "35.123456" not in answer
    assert "139.654321" not in answer
    assert "35.123456" not in event_list
    assert "139.654321" not in event_list
