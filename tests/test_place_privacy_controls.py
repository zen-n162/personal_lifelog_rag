from __future__ import annotations

import json

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.places.location_store import create_place
from personal_lifelog_rag.retrieval.person_place_qa import resolve_place


def test_places_hide_marks_place_non_searchable(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    place = create_place(repository, display_name="Private Place Control", category="cafe", privacy_level="public_label")

    assert main(["--db-path", str(db_path), "places", "hide", "--place-id", place["id"], "--dry-run", "--json"]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["dry_run"] is True

    assert main(["--db-path", str(db_path), "places", "hide", "--place-id", place["id"], "--yes", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["hidden"] is True

    with connect(db_path) as connection:
        row = connection.execute("SELECT hidden, searchable, privacy_level FROM places WHERE id = ?", (place["id"],)).fetchone()
    assert row["hidden"] == 1
    assert row["searchable"] == 0
    assert row["privacy_level"] == "public_hidden"
    assert resolve_place(repository, "Private Place Control").status == "none"
