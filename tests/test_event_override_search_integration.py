from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.retrieval.answer_builder import build_answer
from personal_lifelog_rag.retrieval.date_parser import parse_date_query
from personal_lifelog_rag.retrieval.local_search import LocalSearchOptions, local_text_search
from personal_lifelog_rag.retrieval.query_router import route_query
from personal_lifelog_rag.retrieval.temporal_search import search_timeline


def test_hidden_event_is_excluded_from_search_unless_requested(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_search_override_records(repository)
    repository.upsert_event_override("event_hidden", is_hidden=True)

    hidden_excluded = local_text_search(repository, LocalSearchOptions(query="秘密タグ", limit=10))
    hidden_included = local_text_search(
        repository,
        LocalSearchOptions(query="秘密タグ", limit=10, include_hidden=True),
    )

    assert hidden_excluded["results"] == []
    assert hidden_included["results"][0]["events"][0]["is_hidden"] == 1


def test_pinned_event_ranks_first_in_search_and_qa(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_search_override_records(repository)
    repository.upsert_event_override("event_second", is_pinned=True)

    report = local_text_search(repository, LocalSearchOptions(query="新宿", limit=10))
    routed = route_query(repository, "新宿に行ったのはいつ？", limit=10)

    assert report["results"][0]["date"] == "2024-12-25"
    assert routed.results[0]["date"] == "2024-12-25"
    assert routed.results[0]["events"][0]["is_pinned"] == 1


def test_verified_event_is_displayed_in_answer(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_search_override_records(repository)
    repository.upsert_event_override("event_first", is_verified=True)

    result = search_timeline(
        repository,
        "2024年12月24日は何していた？",
        date_range=parse_date_query("2024年12月24日は何していた？"),
    )
    answer = build_answer("2024年12月24日は何していた？", result)

    assert "手動確認済み" in answer


def test_tags_are_search_targets(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _seed_search_override_records(repository)
    repository.upsert_event_override("event_first", tags=["重要", "旅行"])

    report = local_text_search(repository, LocalSearchOptions(query="重要", limit=10))

    assert report["results"][0]["date"] == "2024-12-24"
    assert "重要" in (report["results"][0]["events"][0]["tags_json"] or "")


def _seed_search_override_records(repository: LifelogRepository) -> None:
    repository.add_event(
        id="event_first",
        date="2024-12-24",
        start_time="10:00:00",
        end_time="11:00:00",
        title="新宿の出来事",
        summary="新宿に関する記録",
        confidence=0.6,
    )
    repository.add_event(
        id="event_second",
        date="2024-12-25",
        start_time="10:00:00",
        end_time="11:00:00",
        title="新宿の出来事",
        summary="新宿に関する記録",
        confidence=0.6,
    )
    repository.add_event(
        id="event_hidden",
        date="2024-12-26",
        start_time="10:00:00",
        end_time="11:00:00",
        title="秘密タグ",
        summary="hidden検索テスト",
        confidence=0.6,
    )
