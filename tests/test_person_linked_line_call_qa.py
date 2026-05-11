from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.faces.person_service import create_person
from personal_lifelog_rag.line.person_links import link_line_speaker_to_person
from personal_lifelog_rag.retrieval.query_router import route_query


def test_person_line_qa_includes_linked_call_counts(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    person = create_person(repository, name="人物A", public_name="人物A", privacy_level="public_alias")
    repository.add_line_message(
        id="line_person_call",
        chat_id="chat_person_call",
        source_file="dummy.txt",
        sent_at="2025-02-03T20:00:00+09:00",
        sender="SpeakerA",
        text="dummy",
    )
    with connect(repository.db_path) as connection:
        connection.execute(
            """
            INSERT INTO line_call_events (
                message_id, chat_id, sent_at, sender, call_status, duration_sec, raw_text_short
            )
            VALUES ('line_person_call', 'chat_person_call', '2025-02-03T20:00:00+09:00', 'SpeakerA', 'completed', 120, 'call')
            """
        )
        connection.commit()
    link_line_speaker_to_person(
        repository,
        chat_id="chat_person_call",
        speaker_name="SpeakerA",
        person_id=person["id"],
        yes=True,
    )

    result = route_query(repository, "人物Aとの通話はいつ？")

    assert result.intent == "person_line_search"
    assert result.results[0]["date"] == "2025-02-03"
    assert result.results[0]["call_count"] == 1
    assert "call=1" in result.answer
