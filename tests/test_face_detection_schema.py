from __future__ import annotations

from personal_lifelog_rag.db.repository import LifelogRepository


def test_face_tables_are_created(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    stats = repository.stats()

    assert "face_detections" in stats
    assert "face_detection_runs" in stats
    assert "face_embeddings" in stats
    assert "face_clusters" in stats
    assert "face_cluster_members" in stats
    assert stats["face_detections"] == 0
    assert stats["face_embeddings"] == 0
    assert stats["face_clusters"] == 0
    assert stats["face_cluster_members"] == 0
