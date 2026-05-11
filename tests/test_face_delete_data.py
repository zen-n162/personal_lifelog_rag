from __future__ import annotations

import numpy as np

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.privacy_controls import face_delete_data


def test_face_delete_data_dry_run_does_not_remove_files_or_embedding(tmp_path) -> None:
    repository, face_id, crop_path = _seed_face_data(tmp_path)

    report = face_delete_data(repository, face_id=face_id, delete_crop=True, delete_embedding=True, dry_run=True)

    assert report["has_embedding"] is True
    assert crop_path.exists()
    with connect(repository.db_path) as connection:
        assert connection.execute("SELECT 1 FROM face_embeddings WHERE face_id = ?", (face_id,)).fetchone() is not None


def test_face_delete_data_removes_crop_and_embedding_when_confirmed(tmp_path) -> None:
    repository, face_id, crop_path = _seed_face_data(tmp_path)

    report = face_delete_data(repository, face_id=face_id, delete_crop=True, delete_embedding=True, dry_run=False, yes=True)

    assert report["deleted_embeddings"] == 1
    assert not crop_path.exists()
    with connect(repository.db_path) as connection:
        assert connection.execute("SELECT 1 FROM face_embeddings WHERE face_id = ?", (face_id,)).fetchone() is None
        row = connection.execute("SELECT crop_path, hidden FROM face_detections WHERE id = ?", (face_id,)).fetchone()
    assert row["crop_path"] is None
    assert row["hidden"] == 1


def _seed_face_data(tmp_path):
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    image_path = tmp_path / "source.jpg"
    image_path.write_bytes(b"fake")
    crop_path = tmp_path / "crop.jpg"
    crop_path.write_bytes(b"crop")
    thumbnail_path = tmp_path / "thumb.jpg"
    thumbnail_path.write_bytes(b"thumb")
    media_id = repository.add_media_item(
        id="media_face_privacy",
        file_path=str(image_path),
        file_name=image_path.name,
        captured_at="2025-01-01T10:00:00",
    )
    face_id = "face_privacy"
    blob = np.asarray([0.1, 0.2], dtype=np.float32).tobytes()
    with connect(repository.db_path) as connection:
        connection.execute(
            """
            INSERT INTO face_detections (
                id, media_id, detected_at, engine, model_name, status,
                bbox_x, bbox_y, bbox_w, bbox_h, detection_score,
                image_width, image_height, crop_path, thumbnail_path
            )
            VALUES (?, ?, '2025-01-01T10:00:00', 'fake', 'fake', 'success',
                    1, 1, 10, 10, 0.9, 100, 100, ?, ?)
            """,
            (face_id, media_id, str(crop_path), str(thumbnail_path)),
        )
        connection.execute(
            """
            INSERT INTO face_embeddings (
                face_id, embedding_model, embedding_dim, embedding_blob,
                embedding_format, normalized, status
            )
            VALUES (?, 'fake', 2, ?, 'float32_numpy', 1, 'success')
            """,
            (face_id, blob),
        )
        connection.commit()
    return repository, face_id, crop_path
