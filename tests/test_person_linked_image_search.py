from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.embeddings.multimodal_search import multimodal_search
from personal_lifelog_rag.embeddings.schemas import MultimodalSearchOptions
from personal_lifelog_rag.faces.person_service import create_person
from personal_lifelog_rag.vlm.schemas import ImageSearchOptions
from personal_lifelog_rag.vlm.vlm_service import image_search


def test_image_search_uses_manual_media_people_for_person_query(tmp_path: Path) -> None:
    repository, person = _seed_person_media(tmp_path)

    report = image_search(repository, ImageSearchOptions(query="人物Aが写っている写真", limit=10))

    assert report["person_resolution"]["resolved_person_id"] == person["id"]
    assert report["results"][0]["media_id"] == "media_person_linked"
    assert report["results"][0]["person_score"] == 1.0
    assert "人物A" in report["results"][0]["related_persons"]
    assert "media_people" in report["results"][0]["evidence_types"]


def test_multimodal_search_adds_person_score_and_downranks_unrelated_person_query(tmp_path: Path) -> None:
    repository, _person = _seed_person_media(tmp_path)
    repository.add_media_item(
        id="media_unrelated",
        file_path=str(tmp_path / "unrelated.jpg"),
        file_name="unrelated.jpg",
        file_hash="hash-unrelated",
        media_type="image",
        captured_at="2025-01-03T12:00:00+09:00",
    )
    repository.upsert_media_vlm(
        media_id="media_unrelated",
        caption="A generic portrait-like photo",
        short_caption="Photo",
        status="success",
        vlm_engine="qwen3_vl_transformers",
    )

    report = multimodal_search(
        repository,
        MultimodalSearchOptions(query="人物Aが写っている写真", backend="vlm_sql", limit=10),
    )

    linked = next(row for row in report["results"] if row["media_id"] == "media_person_linked")
    assert linked["score_components"]["person_score"] == 1.0
    assert linked["score_components"]["person_face_score"] == 1.0
    assert "person" in linked["evidence_types"]
    assert report["results"][0]["media_id"] == "media_person_linked"


def _seed_person_media(tmp_path: Path) -> tuple[LifelogRepository, dict[str, object]]:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    person = create_person(repository, name="人物A", public_name="人物A", privacy_level="public_alias")
    repository.add_media_item(
        id="media_person_linked",
        file_path=str(tmp_path / "person.jpg"),
        file_name="person.jpg",
        file_hash="hash-person",
        media_type="image",
        captured_at="2025-01-02T10:00:00+09:00",
    )
    repository.upsert_media_vlm(
        media_id="media_person_linked",
        caption="A person in a room",
        short_caption="Person candidate",
        status="success",
        vlm_engine="qwen3_vl_transformers",
    )
    with connect(repository.db_path) as connection:
        connection.execute(
            """
            INSERT INTO media_people (
                media_id, person_id, source, confidence, face_cluster_id, verified_by_user
            )
            VALUES ('media_person_linked', ?, 'face_cluster', 0.9, NULL, 1)
            """,
            [person["id"]],
        )
        connection.commit()
    return repository, person
