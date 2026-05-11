from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository


def test_face_embedding_and_cluster_tables_are_created(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    stats = repository.stats()

    assert stats["face_embeddings"] == 0
    assert stats["face_clusters"] == 0
    assert stats["face_cluster_members"] == 0
