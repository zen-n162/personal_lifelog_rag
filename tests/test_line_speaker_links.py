from __future__ import annotations

import pytest

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.faces.person_service import create_person, get_person
from personal_lifelog_rag.line.person_links import (
    link_line_speaker_to_person,
    list_line_speaker_links,
    list_line_speakers,
    search_person_line_days,
    unlink_line_speaker_from_person,
)


def test_line_speaker_manual_link_and_unlink(tmp_path) -> None:
    repository = _repository_with_line_messages(tmp_path)
    person = create_person(repository, name="人物テストA", public_name="人物A", privacy_level="private")

    speakers = list_line_speakers(repository, limit=10)
    assert speakers[0]["speaker_name"] == "SpeakerA"
    assert speakers[0]["message_count"] == 2

    with pytest.raises(ValueError, match="requires --yes"):
        link_line_speaker_to_person(
            repository,
            chat_id="chat_a",
            speaker_name="SpeakerA",
            person_id=person["id"],
            yes=False,
        )

    result = link_line_speaker_to_person(
        repository,
        chat_id="chat_a",
        speaker_name="SpeakerA",
        person_id=person["id"],
        yes=True,
        add_alias=True,
    )
    assert result["person"]["id"] == person["id"]
    links = list_line_speaker_links(repository)
    assert links[0]["speaker_name"] == "SpeakerA"

    updated_person = get_person(repository, person["id"])
    assert updated_person is not None
    assert any(alias["alias"] == "SpeakerA" and alias["source"] == "line_speaker" for alias in updated_person["aliases"])

    report = search_person_line_days(repository, person_name="SpeakerA", limit=5)
    assert report["results"][0]["date"] == "2025-01-02"
    assert "関係性は推定していません" in report["answer"]

    unlinked = unlink_line_speaker_from_person(
        repository,
        chat_id="chat_a",
        speaker_name="SpeakerA",
        person_id=person["id"],
        yes=True,
    )
    assert unlinked["deleted"] == 1
    assert list_line_speaker_links(repository) == []


def _repository_with_line_messages(tmp_path) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_line_message(
        id="line_a_1",
        chat_id="chat_a",
        source_file="dummy_line.txt",
        sent_at="2025-01-02T10:00:00+09:00",
        sender="SpeakerA",
        text="短いダミーメッセージ",
    )
    repository.add_line_message(
        id="line_a_2",
        chat_id="chat_a",
        source_file="dummy_line.txt",
        sent_at="2025-01-02T10:05:00+09:00",
        sender="SpeakerA",
        text="確認用",
    )
    repository.add_line_message(
        id="line_b_1",
        chat_id="chat_b",
        source_file="dummy_line.txt",
        sent_at="2025-01-03T10:00:00+09:00",
        sender="SpeakerB",
        text="別話者",
    )
    return repository
