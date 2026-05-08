from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.line.call_index import build_call_index
from personal_lifelog_rag.retrieval.query_router import route_query


def test_route_date_qa_to_existing_answer_builder(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_router_records(repository)

    result = route_query(repository, "2024年12月24日は何していた？")

    assert result.intent == "date_qa"
    assert result.routing == "ask"
    assert "2024年12月24日の記録を確認しました" in result.answer


def test_route_place_visit_to_search(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_router_records(repository)

    result = route_query(repository, "新宿に行ったのはいつ？")

    assert result.intent == "place_visit"
    assert result.routing == "search"
    assert result.results
    assert result.results[0]["classification"] == "actual_or_likely_action"


def test_route_call_activity_to_call_search_ranking(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_router_records(repository)
    build_call_index(repository)

    result = route_query(repository, "通話した日は？")

    assert result.intent == "call_activity"
    assert result.routing == "search"
    assert result.results[0]["call_summary"]["completed"] == 1


def test_route_unknown_returns_helpful_message(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()

    result = route_query(repository, "ふわっといい感じによろしく")

    assert result.intent == "unknown"
    assert result.routing == "unsupported"
    assert "対応しやすい聞き方" in result.answer


def _seed_router_records(repository: LifelogRepository) -> None:
    repository.add_line_message(
        id="line_shinjuku",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T17:30:00+09:00",
        sender="自分",
        text="18時に新宿着く！",
    )
    repository.add_line_message(
        id="line_food",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T22:10:00+09:00",
        sender="自分",
        text="今日のご飯おいしかったね",
    )
    repository.add_line_message(
        id="line_call",
        chat_id="chat_dummy",
        source_file="sample_chat.txt",
        sent_at="2024-12-24T23:00:00+09:00",
        sender="自分",
        text="☎ 通話時間 10:38",
    )
    repository.add_event(
        id="event_shinjuku",
        date="2024-12-24",
        start_time="17:30:00",
        end_time="18:30:00",
        title="移動・待ち合わせの可能性",
        summary="場所候補: 新宿。活動候補: 着く。",
        location_name="新宿",
        confidence=0.9,
    )
