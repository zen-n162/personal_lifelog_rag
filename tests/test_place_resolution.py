from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.places.location_store import create_place
from personal_lifelog_rag.retrieval.person_place_qa import resolve_place


def test_place_resolution_uses_alias_and_public_label(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    create_place(
        repository,
        place_id="place_a",
        display_name="場所テストA",
        public_name="場所A",
        category="station",
        privacy_level="public_label",
        aliases=["場所別名A"],
        manual_verified=True,
    )

    result = resolve_place(repository, "場所別名A", public_mode=True)

    assert result.status == "resolved"
    assert result.resolved["id"] == "place_a"
    assert result.resolved["place_label"] == "場所A"
    assert "display_name" not in result.resolved


def test_place_resolution_reports_ambiguous_candidates(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    create_place(repository, place_id="place_a", display_name="場所テストA", aliases=["同じ場所"], manual_verified=True)
    create_place(repository, place_id="place_b", display_name="場所テストB", aliases=["同じ場所"], manual_verified=True)

    result = resolve_place(repository, "同じ場所")

    assert result.status == "ambiguous"
    assert {row["id"] for row in result.candidates} == {"place_a", "place_b"}
