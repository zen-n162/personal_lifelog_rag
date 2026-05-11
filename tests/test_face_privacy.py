from __future__ import annotations

from pathlib import Path

from PIL import Image

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.faces.face_service import run_face_detection
from personal_lifelog_rag.faces.schemas import FaceDetectOptions
from personal_lifelog_rag.ui.face_review_service import face_detail_for_ui


def test_face_detail_hides_crops_when_private_display_is_off(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    image_path = tmp_path / "face.jpg"
    Image.new("RGB", (80, 80), "white").save(image_path)
    repository.add_media_item(
        id="media_face",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-face",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    run_face_detection(
        repository,
        FaceDetectOptions(date="2024-12-24", engine="fake", limit=1, save_crops=True),
        faces_dir=tmp_path / "faces",
    )
    face_id = repository._fetch_all("SELECT id FROM face_detections", [])[0]["id"]

    public_detail = face_detail_for_ui(repository, face_id, show_private_crops=False)
    private_detail = face_detail_for_ui(repository, face_id, show_private_crops=True)

    assert public_detail["face_thumbnail"] is None
    assert "hidden" in public_detail["crop_note"]
    assert private_detail["face_thumbnail"]
    assert "identity: not inferred" in private_detail["summary"]


def test_face_detail_repairs_missing_private_thumbnail_from_bbox(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    image_path = tmp_path / "face.jpg"
    Image.new("RGB", (120, 120), "white").save(image_path)
    repository.add_media_item(
        id="media_face_repair",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-face-repair",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    run_face_detection(
        repository,
        FaceDetectOptions(date="2024-12-24", engine="fake", limit=1, save_crops=False),
        faces_dir=tmp_path / "faces",
    )
    row = repository._fetch_all("SELECT id, thumbnail_path FROM face_detections", [])[0]
    assert row["thumbnail_path"] is None

    detail = face_detail_for_ui(repository, row["id"], show_private_crops=True)

    assert detail["face_thumbnail"]
    assert Path(detail["face_thumbnail"]).exists()


def test_face_detail_explains_no_face_rows_have_no_thumbnail(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    image_path = tmp_path / "noface.jpg"
    Image.new("RGB", (120, 120), "white").save(image_path)
    repository.add_media_item(
        id="media_no_face_detail",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-no-face-detail",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    run_face_detection(
        repository,
        FaceDetectOptions(date="2024-12-24", engine="fake", limit=1, save_crops=True),
        faces_dir=tmp_path / "faces",
    )
    face_id = repository._fetch_all("SELECT id FROM face_detections", [])[0]["id"]

    detail = face_detail_for_ui(repository, face_id, show_private_crops=True)

    assert detail["face_thumbnail"] is None
    assert "no_face_detected" in detail["crop_note"]
    assert "status=success" in detail["summary"]
