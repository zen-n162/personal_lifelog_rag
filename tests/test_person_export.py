from __future__ import annotations

import json

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.faces.person_service import create_person
from personal_lifelog_rag.privacy_controls import person_export


def test_public_person_export_redacts_display_name(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    person = create_person(repository, name="Private Export Name", public_name="人物A", privacy_level="public_alias")

    report = person_export(repository, person_id=person["id"], mode="public_redacted", dry_run=True)

    payload = report["payload"]
    assert payload["person"]["name"] == "人物A"
    assert "display_name" not in payload["person"]
    assert "Private Export Name" not in str(payload)


def test_private_person_export_can_include_display_name(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    person = create_person(repository, name="Private Export Name", privacy_level="private")

    report = person_export(repository, person_id=person["id"], mode="private", dry_run=True)

    assert report["payload"]["person"]["display_name"] == "Private Export Name"


def test_person_export_cli_dry_run_outputs_valid_json(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    person = create_person(repository, name="Private CLI Export Name", public_name="人物A", privacy_level="public_alias")

    assert main(
        [
            "--db-path",
            str(db_path),
            "person-export",
            "--person-id",
            person["id"],
            "--mode",
            "public_redacted",
            "--dry-run",
            "--json",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["payload"]["person"]["name"] == "人物A"
    assert "Private CLI Export Name" not in str(payload)
