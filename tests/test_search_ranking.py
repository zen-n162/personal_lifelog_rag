from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.retrieval.local_search import LocalSearchOptions, local_text_search


def test_actual_evidence_ranks_above_plan_evidence(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_ranking_records(repository)

    report = local_text_search(
        repository,
        LocalSearchOptions(query="新宿", intent="place_visit", limit=10),
    )

    assert report["results"][0]["date"] == "2024-12-24"
    assert report["results"][0]["classification"] == "actual_or_likely_action"
    assert _day(report, "2024-12-25")["classification"] == "plan_or_candidate"


def test_photo_gps_event_day_ranks_above_line_only_day(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_ranking_records(repository)

    report = local_text_search(
        repository,
        LocalSearchOptions(query="新宿", intent="place_visit", limit=10),
    )
    actual = _day(report, "2024-12-24")
    mention = _day(report, "2024-12-26")

    assert actual["ranking_score"] > mention["ranking_score"]
    assert actual["same_day_gps_photo_count"] == 1


def test_missed_call_only_ranks_below_completed_call(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_call_records(repository)

    report = local_text_search(
        repository,
        LocalSearchOptions(query="通話", intent="call_activity", limit=10),
    )

    assert report["results"][0]["date"] == "2024-12-24"
    assert report["results"][0]["classification"] == "actual_or_likely_action"
    assert _day(report, "2024-12-25")["classification"] == "mention_only"


def test_mode_actual_filters_results(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_ranking_records(repository)

    report = local_text_search(
        repository,
        LocalSearchOptions(query="新宿", intent="place_visit", mode="actual", limit=10),
    )

    assert report["results"]
    assert {row["classification"] for row in report["results"]} == {"actual_or_likely_action"}


def _day(report: dict, date: str) -> dict:
    for result in report["results"]:
        if result["date"] == date:
            return result
    raise AssertionError(f"date not found: {date}")


def _seed_ranking_records(repository: LifelogRepository) -> None:
    repository.add_line_message(
        id="line_actual",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T17:30:00+09:00",
        sender="自分",
        text="18時に新宿着く！",
    )
    repository.add_event(
        id="event_actual",
        date="2024-12-24",
        start_time="17:30:00",
        end_time="20:00:00",
        title="移動・待ち合わせの可能性",
        summary="場所候補: 新宿。活動候補: 着く。",
        location_name="新宿",
        confidence=0.9,
    )
    repository.add_media_item(
        id="media_actual",
        file_path="/local/photos/actual.jpg",
        file_name="actual.jpg",
        file_hash="hash-actual",
        captured_at="2024-12-24T18:00:00+09:00",
        gps_lat=10.0,
        gps_lon=20.0,
    )
    repository.add_line_message(
        id="line_plan",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-25T17:30:00+09:00",
        sender="自分",
        text="王子か新宿のバスじゃないかなーあるとしたら",
    )
    repository.add_line_message(
        id="line_mention",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-26T17:30:00+09:00",
        sender="自分",
        text="新宿ってすごい",
    )


def _seed_call_records(repository: LifelogRepository) -> None:
    repository.add_line_message(
        id="line_call_done",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T10:00:00+09:00",
        sender="自分",
        text="☎ 通話時間 48:03",
    )
    repository.add_line_message(
        id="line_call_missed",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-25T10:00:00+09:00",
        sender="自分",
        text="☎ 通話に応答がありませんでした",
    )

