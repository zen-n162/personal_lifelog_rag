from __future__ import annotations

import json

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository


def test_line_speakers_cli_list_link_show_unlink(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    repository.add_line_message(
        id="line_cli_1",
        chat_id="chat_cli",
        source_file="dummy_line.txt",
        sent_at="2025-02-01T09:00:00+09:00",
        sender="SpeakerCLI",
        text="短い確認用",
    )

    assert main(["--db-path", str(db_path), "line-speakers", "list", "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["results"][0]["chat_id"] == "chat_cli"
    assert listing["results"][0]["speaker_name"] == "SpeakerCLI"

    assert main(
        [
            "--db-path",
            str(db_path),
            "persons",
            "create",
            "--name",
            "人物テストCLI",
            "--public-name",
            "人物A",
            "--privacy-level",
            "private",
            "--json",
        ]
    ) == 0
    person_id = json.loads(capsys.readouterr().out)["id"]

    assert main(
        [
            "--db-path",
            str(db_path),
            "line-speakers",
            "link-person",
            "--chat-id",
            "chat_cli",
            "--speaker-name",
            "SpeakerCLI",
            "--person-id",
            person_id,
            "--add-alias",
            "--yes",
            "--json",
        ]
    ) == 0
    linked = json.loads(capsys.readouterr().out)
    assert linked["person"]["id"] == person_id

    assert main(["--db-path", str(db_path), "line-speakers", "show-links"]) == 0
    assert "SpeakerCLI" in capsys.readouterr().out

    assert main(["--db-path", str(db_path), "line-speakers", "suggest", "--speaker-name", "SpeakerCLI", "--json"]) == 0
    suggestions = json.loads(capsys.readouterr().out)
    assert suggestions["results"][0]["id"] == person_id

    assert main(
        [
            "--db-path",
            str(db_path),
            "line-speakers",
            "unlink-person",
            "--chat-id",
            "chat_cli",
            "--speaker-name",
            "SpeakerCLI",
            "--person-id",
            person_id,
            "--yes",
        ]
    ) == 0
    assert "deleted=1" in capsys.readouterr().out
