from __future__ import annotations

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.retrieval.answer_builder import build_answer
from personal_lifelog_rag.retrieval.date_parser import parse_date_query
from personal_lifelog_rag.retrieval.temporal_search import search_timeline
from personal_lifelog_rag.timeline.event_builder import build_events
from personal_lifelog_rag.timeline.event_reports import format_event_list, list_events_report


def test_event_override_is_saved_and_reflected_in_list_events(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_manual_events(repository)

    repository.upsert_event_override(
        "event_morning",
        title_override="手動修正タイトル",
        summary_override="手動修正した要約",
        location_name_override="テスト駅周辺",
        tags=["旅行", "食事"],
        is_verified=True,
    )
    rows = list_events_report(repository, start_date="2024-12-24", end_date="2024-12-24")
    output = format_event_list(rows)

    assert rows[0]["title"] == "手動修正タイトル"
    assert rows[0]["summary"] == "手動修正した要約"
    assert rows[0]["location_name"] == "テスト駅周辺"
    assert rows[0]["is_verified"] == 1
    assert "manual verified" in output
    assert "旅行" in output


def test_hidden_event_is_excluded_from_ask(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_manual_events(repository)
    repository.upsert_event_override("event_morning", is_hidden=True)

    result = search_timeline(
        repository,
        "2024年12月24日は何していた？",
        date_range=parse_date_query("2024年12月24日は何していた？"),
    )
    answer = build_answer("2024年12月24日は何していた？", result)

    assert "朝のイベント" not in answer
    assert "夜のイベント" in answer
    assert len(result.events) == 1


def test_pinned_event_is_ordered_before_other_events(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_manual_events(repository)
    repository.upsert_event_override("event_night", is_pinned=True, is_verified=True)

    rows = list_events_report(repository, start_date="2024-12-24", end_date="2024-12-24")
    result = search_timeline(
        repository,
        "2024年12月24日は何していた？",
        date_range=parse_date_query("2024年12月24日は何していた？"),
    )
    answer = build_answer("2024年12月24日は何していた？", result)

    assert rows[0]["id"] == "event_night"
    assert result.events[0]["id"] == "event_night"
    assert "手動確認済み / 優先表示" in answer


def test_build_events_does_not_delete_event_overrides(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_build_records(repository)
    build_events(repository, start_date="2024-12-24")
    event_id = repository.list_events(start_date="2024-12-24", end_date="2024-12-24")[0]["id"]
    repository.upsert_event_override(event_id, title_override="再生成後も残るタイトル", is_verified=True)

    build_events(repository, start_date="2024-12-24")
    event = repository.get_event(event_id)

    assert event is not None
    assert event["title"] == "再生成後も残るタイトル"
    assert event["is_verified"] == 1


def test_update_event_cli_saves_override(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_manual_events(repository)

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "update-event",
            "event_morning",
            "--title",
            "CLI修正タイトル",
            "--summary",
            "CLI修正要約",
            "--location",
            "CLI場所",
            "--tag",
            "確認済み",
            "--verified",
            "--pinned",
        ]
    )
    output = capsys.readouterr().out
    event = repository.get_event("event_morning")

    assert exit_code == 0
    assert "Updated event override" in output
    assert event is not None
    assert event["title"] == "CLI修正タイトル"
    assert event["location_name"] == "CLI場所"
    assert event["is_verified"] == 1
    assert event["is_pinned"] == 1


def _seed_manual_events(repository: LifelogRepository) -> None:
    repository.add_event(
        id="event_morning",
        date="2024-12-24",
        start_time="09:00:00",
        end_time="10:00:00",
        title="朝のイベント",
        summary="朝の自動要約",
        confidence=0.5,
    )
    repository.add_event(
        id="event_night",
        date="2024-12-24",
        start_time="20:00:00",
        end_time="21:00:00",
        title="夜のイベント",
        summary="夜の自動要約",
        confidence=0.6,
    )


def _seed_build_records(repository: LifelogRepository) -> None:
    repository.add_line_message(
        id="line_build_1",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T17:30:00+09:00",
        sender="自分",
        text="18時にテスト駅に着く",
    )
    repository.add_media_item(
        id="media_build_1",
        file_path="/local/photos/build_1.jpg",
        file_name="build_1.jpg",
        file_hash="hash-build-1",
        captured_at="2024-12-24T18:00:00+09:00",
        gps_lat=10.0,
        gps_lon=20.0,
    )
