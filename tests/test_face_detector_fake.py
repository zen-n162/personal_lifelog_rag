from __future__ import annotations

from pathlib import Path

from PIL import Image

from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.faces.engines import OpenCvYuNetFaceDetector, _face_candidate_shape_ok, _resolve_haar_cascade_path
from personal_lifelog_rag.faces.face_service import _clip_detection_to_image, run_face_detection
from personal_lifelog_rag.faces.schemas import FaceDetectOptions, FaceDetection


def test_fake_detector_saves_success_and_crop(tmp_path: Path) -> None:
    repository = _repository_with_image(tmp_path, "face.jpg")

    report = run_face_detection(
        repository,
        FaceDetectOptions(date="2024-12-24", engine="fake", limit=10, save_crops=True),
        faces_dir=tmp_path / "faces",
    )

    rows = repository._fetch_all("SELECT * FROM face_detections", [])
    assert report.success_count == 1
    assert rows[0]["status"] == "success"
    assert rows[0]["review_status"] == "unreviewed"
    assert Path(rows[0]["crop_path"]).exists()
    assert Path(rows[0]["thumbnail_path"]).exists()


def test_fake_detector_saves_no_face_status(tmp_path: Path) -> None:
    repository = _repository_with_image(tmp_path, "noface.jpg")

    report = run_face_detection(repository, FaceDetectOptions(date="2024-12-24", engine="fake", limit=10))

    rows = repository._fetch_all("SELECT status FROM face_detections", [])
    assert report.no_face_count == 1
    assert rows[0]["status"] == "no_face_detected"


def test_opencv_haar_resolver_uses_conda_share_fallback(tmp_path: Path) -> None:
    cascade = tmp_path / "share" / "opencv4" / "haarcascades" / "haarcascade_frontalface_default.xml"
    cascade.parent.mkdir(parents=True)
    cascade.write_text("<opencv_storage></opencv_storage>", encoding="utf-8")

    class FakeCv2:
        data = None
        __file__ = str(tmp_path / "lib" / "python" / "cv2" / "__init__.py")

    assert _resolve_haar_cascade_path(FakeCv2(), sys_prefix=str(tmp_path)) == str(cascade)


def test_opencv_haar_resolver_accepts_eye_cascade_filename(tmp_path: Path) -> None:
    cascade = tmp_path / "share" / "opencv4" / "haarcascades" / "haarcascade_eye.xml"
    cascade.parent.mkdir(parents=True)
    cascade.write_text("<opencv_storage></opencv_storage>", encoding="utf-8")

    class FakeCv2:
        data = None
        __file__ = str(tmp_path / "lib" / "python" / "cv2" / "__init__.py")

    assert _resolve_haar_cascade_path(FakeCv2(), filename="haarcascade_eye.xml", sys_prefix=str(tmp_path)) == str(cascade)


def test_opencv_haar_candidate_shape_filter_rejects_non_face_ratios() -> None:
    assert _face_candidate_shape_ok(52, 52)
    assert not _face_candidate_shape_ok(120, 40)
    assert not _face_candidate_shape_ok(40, 120)


def test_opencv_yunet_requires_local_model_path(tmp_path: Path) -> None:
    detector = OpenCvYuNetFaceDetector(model_path=str(tmp_path / "missing_yunet.onnx"))

    assert not detector.is_available()
    assert "model file not found" in str(detector.unavailable_reason())
    result = detector.detect(tmp_path / "image.jpg")
    assert result.status == "engine_unavailable"


def test_detection_bbox_is_clipped_to_image_bounds() -> None:
    detection = FaceDetection(bbox_x=564.47, bbox_y=209.21, bbox_w=213.08, bbox_h=339.71, detection_score=0.93)

    clipped = _clip_detection_to_image(detection, image_width=768, image_height=1024)

    assert clipped is not None
    assert clipped.bbox_x == detection.bbox_x
    assert clipped.bbox_y == detection.bbox_y
    assert clipped.bbox_w <= 768 - detection.bbox_x
    assert clipped.bbox_y + clipped.bbox_h <= 1024


def _repository_with_image(tmp_path: Path, name: str) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    image_path = tmp_path / name
    Image.new("RGB", (96, 96), "white").save(image_path)
    repository.add_media_item(
        id="media_face",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash=f"hash-{name}",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    return repository
