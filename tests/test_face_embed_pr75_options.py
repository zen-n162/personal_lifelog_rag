from __future__ import annotations

from pathlib import Path

from PIL import Image

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.db.schema import initialize_schema
from personal_lifelog_rag.embeddings.similarity import FLOAT32_FORMAT, serialize_embedding
from personal_lifelog_rag.faces.embedding_service import run_face_clustering, run_face_embedding
from personal_lifelog_rag.faces.schemas import FaceClusteringOptions, FaceEmbeddingOptions


def test_face_embed_filters_yunet_success_crop_and_existing_file(tmp_path: Path) -> None:
    repository = _seed_detection_rows(tmp_path)

    report = run_face_embedding(
        repository,
        FaceEmbeddingOptions(
            date="2024-12-24",
            engine="fake",
            detections_engine="opencv_yunet",
            status="success",
            only_with_crop=True,
            only_existing_files=True,
            limit=0,
            dry_run=True,
        ),
    )

    assert report.selected_count == 2
    assert report.detections_engine == "opencv_yunet"
    assert report.only_with_crop is True


def test_face_embed_replace_only_selected_rows(tmp_path: Path) -> None:
    repository = _seed_detection_rows(tmp_path)
    _insert_embedding(repository, "face_yunet", [1.0, 0.0, 0.0])
    _insert_embedding(repository, "face_haar", [0.0, 1.0, 0.0])

    report = run_face_embedding(
        repository,
        FaceEmbeddingOptions(
            date="2024-12-24",
            engine="fake",
            detections_engine="opencv_yunet",
            status="success",
            only_with_crop=True,
            only_existing_files=True,
            limit=0,
            replace=True,
            batch_size=2,
        ),
    )

    rows = repository._fetch_all("SELECT face_id, embedding_model, status FROM face_embeddings ORDER BY face_id", [])
    assert report.deleted_embedding_count == 1
    assert report.success_count == 2
    assert rows == [
        {"face_id": "face_haar", "embedding_model": "seed_model", "status": "success"},
        {"face_id": "face_yunet", "embedding_model": "fake_face_embedding", "status": "success"},
        {"face_id": "face_yunet_2", "embedding_model": "fake_face_embedding", "status": "success"},
    ]


def test_face_cluster_replace_scope_keeps_other_scope(tmp_path: Path) -> None:
    repository = _seed_detection_rows(tmp_path)
    _insert_embedding(repository, "face_yunet", [1.0, 0.0, 0.0])
    _insert_embedding(repository, "face_yunet_2", [0.99, 0.1, 0.0])
    _insert_cluster(repository, "old_a", "dbscan_cosine:scope_a")
    _insert_cluster(repository, "old_b", "dbscan_cosine:scope_b")

    report = run_face_clustering(
        repository,
        FaceClusteringOptions(
            date="2024-12-24",
            distance_threshold=0.2,
            min_samples=2,
            replace=True,
            scope="scope_a",
        ),
    )

    clusters = repository._fetch_all("SELECT id, clustering_method FROM face_clusters ORDER BY id", [])
    assert report.replace_count == 1
    assert report.clusters_written == 1
    assert any(row["id"] == "old_b" for row in clusters)
    assert all(row["id"] != "old_a" for row in clusters)


def _seed_detection_rows(tmp_path: Path) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    image_path = tmp_path / "source.jpg"
    crop_path = tmp_path / "crop.jpg"
    Image.new("RGB", (96, 96), "white").save(image_path)
    Image.new("RGB", (48, 48), "white").save(crop_path)
    repository.add_media_item(
        id="media_pr75",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-pr75",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    _insert_detection(repository, "face_yunet", "opencv_yunet", "success", crop_path)
    _insert_detection(repository, "face_haar", "opencv_haar", "success", crop_path)
    _insert_detection(repository, "face_no_crop", "opencv_yunet", "success", None)
    _insert_detection(repository, "face_no_face", "opencv_yunet", "no_face_detected", crop_path)
    _insert_detection(repository, "face_yunet_2", "opencv_yunet", "success", crop_path)
    return repository


def _insert_detection(repository: LifelogRepository, face_id: str, engine: str, status: str, crop_path: Path | None) -> None:
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        connection.execute(
            """
            INSERT INTO face_detections (
                id, media_id, detected_at, engine, model_name, status,
                bbox_x, bbox_y, bbox_w, bbox_h, detection_score,
                image_width, image_height, crop_path, thumbnail_path,
                privacy_level, review_status, created_at, updated_at
            )
            VALUES (?, 'media_pr75', '2024-12-24T10:00:00+09:00', ?, ?, ?,
                    1, 1, 40, 40, 0.9, 96, 96, ?, ?, 'private', 'unreviewed',
                    '2024-12-24T10:00:00', '2024-12-24T10:00:00')
            """,
            (face_id, engine, engine, status, str(crop_path) if crop_path else None, str(crop_path) if crop_path else None),
        )
        connection.commit()


def _insert_embedding(repository: LifelogRepository, face_id: str, vector: list[float]) -> None:
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        connection.execute(
            """
            INSERT OR REPLACE INTO face_embeddings (
                face_id, embedding_model, embedding_dim, embedding_blob, embedding_format,
                normalized, status, created_at, updated_at
            )
            VALUES (?, 'seed_model', ?, ?, ?, 1, 'success',
                    '2024-12-24T10:00:00', '2024-12-24T10:00:00')
            """,
            (face_id, len(vector), serialize_embedding(vector, embedding_format=FLOAT32_FORMAT), FLOAT32_FORMAT),
        )
        connection.commit()


def _insert_cluster(repository: LifelogRepository, cluster_id: str, method: str) -> None:
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        connection.execute(
            """
            INSERT INTO face_clusters (
                id, cluster_label, representative_face_id, face_count,
                clustering_method, distance_threshold, status, review_status,
                privacy_level, created_at, updated_at
            )
            VALUES (?, ?, NULL, 0, ?, 0.2, 'unreviewed', 'unreviewed',
                    'private', '2024-12-24T10:00:00', '2024-12-24T10:00:00')
            """,
            (cluster_id, cluster_id, method),
        )
        connection.commit()
