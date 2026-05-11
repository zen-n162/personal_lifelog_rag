from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.db.schema import initialize_schema
from personal_lifelog_rag.faces.embedding_service import run_face_clustering
from personal_lifelog_rag.faces.schemas import FaceClusteringOptions


def test_face_clustering_groups_near_embeddings(tmp_path: Path) -> None:
    repository = _repository_with_embeddings(tmp_path)

    report = run_face_clustering(
        repository,
        FaceClusteringOptions(
            start_date="2024-12-01",
            end_date="2024-12-31",
            distance_threshold=0.2,
            min_samples=2,
        ),
    )

    assert report.cluster_candidates == 1
    assert report.clusters_written == 1
    assert report.members_written == 2
    clusters = repository._fetch_all("SELECT * FROM face_clusters", [])
    members = repository._fetch_all("SELECT * FROM face_cluster_members ORDER BY face_id", [])
    assert clusters[0]["cluster_label"] == "person_candidate_001"
    assert clusters[0]["status"] == "unreviewed"
    assert clusters[0]["privacy_level"] == "private"
    assert [row["face_id"] for row in members] == ["face_a", "face_b"]


def test_face_clustering_dry_run_does_not_write(tmp_path: Path) -> None:
    repository = _repository_with_embeddings(tmp_path)

    report = run_face_clustering(
        repository,
        FaceClusteringOptions(distance_threshold=0.2, min_samples=2, dry_run=True),
    )

    assert report.dry_run is True
    assert report.cluster_candidates == 1
    assert repository._fetch_all("SELECT * FROM face_clusters", []) == []
    assert repository._fetch_all("SELECT * FROM face_cluster_members", []) == []


def _repository_with_embeddings(tmp_path: Path) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    image_path = tmp_path / "source.jpg"
    Image.new("RGB", (96, 96), "white").save(image_path)
    repository.add_media_item(
        id="media_cluster",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-cluster",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    rows = [
        ("face_a", "2024-12-24T10:00:00+09:00", _normalized([1.0, 0.0, 0.0])),
        ("face_b", "2024-12-24T10:01:00+09:00", _normalized([0.99, 0.1, 0.0])),
        ("face_c", "2024-12-24T10:02:00+09:00", _normalized([0.0, 1.0, 0.0])),
    ]
    for face_id, detected_at, embedding in rows:
        _insert_face_detection(repository, face_id, detected_at)
        _insert_embedding(repository, face_id, embedding)
    return repository


def _normalized(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array / float(np.linalg.norm(array))


def _insert_face_detection(repository: LifelogRepository, face_id: str, detected_at: str) -> None:
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        connection.execute(
            """
            INSERT INTO face_detections (
                id, media_id, detected_at, engine, model_name, status,
                bbox_x, bbox_y, bbox_w, bbox_h, detection_score,
                image_width, image_height, privacy_level, review_status
            )
            VALUES (?, 'media_cluster', ?, 'fake', 'fake', 'success', 10, 10, 32, 32, 0.9, 96, 96, 'private', 'unreviewed')
            """,
            [face_id, detected_at],
        )
        connection.commit()


def _insert_embedding(repository: LifelogRepository, face_id: str, embedding: np.ndarray) -> None:
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        connection.execute(
            """
            INSERT INTO face_embeddings (
                face_id, embedding_model, embedding_dim, embedding_blob,
                embedding_format, normalized, status
            )
            VALUES (?, 'test_model', ?, ?, 'float32_numpy', 1, 'success')
            """,
            [face_id, int(embedding.shape[0]), embedding.astype(np.float32).tobytes()],
        )
        connection.commit()
