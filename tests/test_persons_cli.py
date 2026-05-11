from __future__ import annotations

import json

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository


def test_persons_cli_create_list_show_update_alias(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    LifelogRepository(db_path).initialize()

    assert main(
        [
            "--db-path",
            str(db_path),
            "persons",
            "create",
            "--name",
            "人物テストA",
            "--public-name",
            "人物A",
            "--privacy-level",
            "private",
            "--json",
        ]
    ) == 0
    created = json.loads(capsys.readouterr().out)
    person_id = created["id"]
    assert created["display_name"] == "人物テストA"

    assert main(["--db-path", str(db_path), "persons", "add-alias", "--person-id", person_id, "--alias", "テスト別名"]) == 0
    capsys.readouterr()
    assert main(
        [
            "--db-path",
            str(db_path),
            "persons",
            "update",
            "--person-id",
            person_id,
            "--name",
            "人物テストA2",
            "--privacy-level",
            "public_alias",
        ]
    ) == 0
    capsys.readouterr()

    assert main(["--db-path", str(db_path), "persons", "list", "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["results"][0]["display_name"] == "人物テストA2"

    assert main(["--db-path", str(db_path), "persons", "show", "--person-id", person_id]) == 0
    output = capsys.readouterr().out
    assert "テスト別名" in output
    assert "face_clusters:" in output


def test_persons_cli_reuses_existing_display_name(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    LifelogRepository(db_path).initialize()

    create_args = [
        "--db-path",
        str(db_path),
        "persons",
        "create",
        "--name",
        "人物同名A",
        "--public-name",
        "人物A",
        "--privacy-level",
        "private",
        "--json",
    ]
    assert main(create_args) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(create_args) == 0
    second = json.loads(capsys.readouterr().out)

    assert second["id"] == first["id"]
    assert main(["--db-path", str(db_path), "persons", "list", "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert len(listing["results"]) == 1


def test_persons_anonymize_preview_hides_public_hidden(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    LifelogRepository(db_path).initialize()
    main(
        [
            "--db-path",
            str(db_path),
            "persons",
            "create",
            "--name",
            "Private Name",
            "--privacy-level",
            "public_hidden",
        ]
    )
    capsys.readouterr()

    assert main(["--db-path", str(db_path), "persons", "anonymize-preview", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["public_name"] == ""
    assert payload["results"][0]["privacy_level"] == "public_hidden"
