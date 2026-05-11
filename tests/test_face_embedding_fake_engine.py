from __future__ import annotations

from pathlib import Path

from PIL import Image

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.faces.embedding_service import run_face_embedding
from personal_lifelog_rag.faces.face_service import run_face_detection
from personal_lifelog_rag.faces.schemas import FaceDetectOptions, FaceEmbeddingOptions


def test_fake_face_embedding_engine_saves_private_blob(tmp_path: Path) -> None:
    repository = _repository_with_fake_face(tmp_path)

    report = run_face_embedding(
        repository,
        FaceEmbeddingOptions(date="2024-12-24", engine="fake", limit=10, skip_existing=True),
    )

    rows = repository._fetch_all("SELECT * FROM face_embeddings", [])
    assert report.success_count == 1
    assert rows[0]["status"] == "success"
    assert rows[0]["embedding_model"] == "fake_face_embedding"
    assert rows[0]["embedding_format"] == "float32_numpy"
    assert rows[0]["embedding_dim"] > 0
    assert rows[0]["embedding_blob"]


def test_face_embed_dry_run_does_not_write(tmp_path: Path) -> None:
    repository = _repository_with_fake_face(tmp_path)

    report = run_face_embedding(
        repository,
        FaceEmbeddingOptions(date="2024-12-24", engine="fake", limit=10, dry_run=True),
    )

    assert report.dry_run is True
    assert report.selected_count == 1
    assert repository._fetch_all("SELECT * FROM face_embeddings", []) == []


def _repository_with_fake_face(tmp_path: Path) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    image_path = tmp_path / "face.jpg"
    Image.new("RGB", (96, 96), "white").save(image_path)
    repository.add_media_item(
        id="media_face_embed",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-face-embed",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    run_face_detection(
        repository,
        FaceDetectOptions(date="2024-12-24", engine="fake", limit=1, save_crops=True),
        faces_dir=tmp_path / "faces",
    )
    return repository
