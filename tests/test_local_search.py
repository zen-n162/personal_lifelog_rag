from __future__ import annotations

import json

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.retrieval.local_search import (
    LocalSearchOptions,
    extract_search_terms,
    format_local_search_report,
    local_text_search,
)


def test_search_line_messages_by_keyword(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_search_records(repository)

    report = local_text_search(repository, LocalSearchOptions(query="新宿", limit=10))

    dates = {result["date"] for result in report["results"]}
    assert "2024-12-24" in dates
    day = _day(report, "2024-12-24")
    assert day["line_match_count"] >= 1
    assert day["line_samples"]
    assert "新宿" in day["line_samples"][0]["text"]


def test_search_events_title_and_summary(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_search_records(repository)

    title_report = local_text_search(repository, LocalSearchOptions(query="通話", limit=10))
    summary_report = local_text_search(repository, LocalSearchOptions(query="アルバム", limit=10))

    assert _day(title_report, "2024-12-25")["event_count"] == 1
    assert _day(summary_report, "2024-12-26")["event_count"] == 1


def test_search_media_filename_caption_and_ocr(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_search_records(repository)

    caption_report = local_text_search(repository, LocalSearchOptions(query="ご飯", limit=10))
    ocr_report = local_text_search(repository, LocalSearchOptions(query="看板", limit=10))

    assert _day(caption_report, "2024-12-24")["media_match_count"] >= 1
    assert _day(ocr_report, "2024-12-27")["media_match_count"] >= 1


def test_search_date_range_filter_and_limit(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_search_records(repository)

    filtered = local_text_search(
        repository,
        LocalSearchOptions(
            query="新宿",
            date_from="2024-12-25",
            date_to="2024-12-31",
            limit=10,
        ),
    )
    limited = local_text_search(repository, LocalSearchOptions(query="新宿", limit=1))

    assert "2024-12-24" not in {result["date"] for result in filtered["results"]}
    assert len(limited["results"]) == 1


def test_search_no_data_does_not_crash(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()

    report = local_text_search(repository, LocalSearchOptions(query="存在しない"))
    output = format_local_search_report(report)

    assert report["results"] == []
    assert "検索結果は見つかりませんでした" in output


def test_search_cli_json_outputs_valid_json(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_search_records(repository)

    exit_code = main(
        [
            "--db-path",
            str(db_path),
            "search",
            "新宿",
            "--limit",
            "10",
            "--date-from",
            "2024-12-01",
            "--date-to",
            "2024-12-31",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["query"] == "新宿"
    assert payload["backend"] == "sqlite_like"
    assert payload["results"]


def test_extract_search_terms_handles_natural_question() -> None:
    assert extract_search_terms("新宿に行ったのはいつ？") == ["新宿"]
    assert extract_search_terms("新宿でご飯") == ["新宿", "ご飯"]
    assert extract_search_terms("通話した日は？") == ["通話"]


def _day(report: dict, date: str) -> dict:
    for result in report["results"]:
        if result["date"] == date:
            return result
    raise AssertionError(f"date not found: {date}")


def _seed_search_records(repository: LifelogRepository) -> None:
    repository.add_line_message(
        id="line_search_shinjuku",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T17:30:00+09:00",
        sender="自分",
        text="18時に新宿着く！",
    )
    repository.add_line_message(
        id="line_search_food",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T22:10:00+09:00",
        sender="自分",
        text="今日のご飯おいしかったね",
    )
    repository.add_event(
        id="event_search_call",
        date="2024-12-25",
        start_time="10:00:00",
        end_time="10:10:00",
        title="通話・連絡",
        summary="LINE通話の記録です。",
        confidence=0.7,
    )
    repository.add_event(
        id="event_search_album",
        date="2024-12-26",
        start_time="19:00:00",
        end_time="19:30:00",
        title="LINEのやりとり",
        summary="アルバムに関する会話があります。",
        confidence=0.5,
    )
    repository.add_media_item(
        id="media_search_food",
        file_path="/local/photos/food.jpg",
        file_name="food.jpg",
        file_hash="hash-search-food",
        captured_at="2024-12-24T22:00:00+09:00",
        caption="新宿でご飯を食べた写真",
    )
    repository.add_media_item(
        id="media_search_ocr",
        file_path="/local/photos/sign.jpg",
        file_name="sign.jpg",
        file_hash="hash-search-sign",
        captured_at="2024-12-27T12:00:00+09:00",
        ocr_text="看板",
    )
