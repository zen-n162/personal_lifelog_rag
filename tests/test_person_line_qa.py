from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.faces.person_service import create_person
from personal_lifelog_rag.line.person_links import link_line_speaker_to_person
from personal_lifelog_rag.retrieval.query_router import route_query


def test_person_line_qa_uses_manual_speaker_link(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_line_message(
        id="line_qa_1",
        chat_id="chat_qa",
        source_file="dummy_line.txt",
        sent_at="2025-04-05T08:00:00+09:00",
        sender="SpeakerQA",
        text="short dummy",
    )
    repository.add_line_message(
        id="line_qa_2",
        chat_id="chat_qa",
        source_file="dummy_line.txt",
        sent_at="2025-04-05T08:10:00+09:00",
        sender="SpeakerQA",
        text="short dummy 2",
    )
    person = create_person(repository, name="人物テストQA", privacy_level="private")
    link_line_speaker_to_person(
        repository,
        chat_id="chat_qa",
        speaker_name="SpeakerQA",
        person_id=person["id"],
        yes=True,
    )

    result = route_query(repository, "人物テストQAとLINEした日は？")

    assert result.intent == "person_line_search"
    assert result.routing == "person-place-qa"
    assert result.results[0]["date"] == "2025-04-05"
    assert result.results[0]["message_count"] == 2
    assert "手動リンク済みLINE話者" in result.answer
    assert "関係性は推定していません" in result.answer


def test_person_line_qa_without_manual_link_does_not_search_messages(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_line_message(
        id="line_qa_unlinked",
        chat_id="chat_qa",
        source_file="dummy_line.txt",
        sent_at="2025-04-05T08:00:00+09:00",
        sender="SpeakerQA",
        text="short dummy",
    )

    result = route_query(repository, "人物テスト未リンクとLINEした日は？")

    assert result.intent == "person_line_search"
    assert result.routing == "person-place-qa"
    assert result.results == []
    assert "手動リンクされたLINE話者は見つかりませんでした" in result.answer
