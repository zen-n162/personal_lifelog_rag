"""Local face detector engines.

PR66 deliberately stops at face detection. Engines return bounding boxes only;
they do not compute identity embeddings, names, emotions, or relationships.
"""

from __future__ import annotations

from pathlib import Path
import sys
import traceback

from PIL import Image

from personal_lifelog_rag.faces.schemas import FaceDetection, FaceDetectionEngineResult


class FaceDetector:
    name = "base"
    model_name = "none"

    def is_available(self) -> bool:
        return False

    def unavailable_reason(self) -> str | None:
        return "engine is not implemented"

    def detect(self, image_path: Path) -> FaceDetectionEngineResult:
        return FaceDetectionEngineResult(status="engine_unavailable", error_message=self.unavailable_reason())


class FakeFaceDetector(FaceDetector):
    """Deterministic detector for tests; never use for production evidence."""

    name = "fake"
    model_name = "fake_face_detector"

    def is_available(self) -> bool:
        return True

    def unavailable_reason(self) -> str | None:
        return None

    def detect(self, image_path: Path) -> FaceDetectionEngineResult:
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception:
            width, height = 100, 100
        if "noface" in image_path.name.lower():
            return FaceDetectionEngineResult(status="no_face_detected", image_width=width, image_height=height)
        bbox_w = max(10.0, width * 0.25)
        bbox_h = max(10.0, height * 0.25)
        return FaceDetectionEngineResult(
            status="success",
            image_width=width,
            image_height=height,
            detections=[
                FaceDetection(
                    bbox_x=max(0.0, (width - bbox_w) / 2),
                    bbox_y=max(0.0, (height - bbox_h) / 2),
                    bbox_w=bbox_w,
                    bbox_h=bbox_h,
                    detection_score=0.99,
                )
            ],
        )


class NoopFaceDetector(FaceDetector):
    name = "noop"
    model_name = "noop"

    def is_available(self) -> bool:
        return False

    def unavailable_reason(self) -> str | None:
        return "noop face detector is intentionally unavailable"


class OpenCvHaarFaceDetector(FaceDetector):
    name = "opencv_haar"
    model_name = "haarcascade_frontalface_default"

    def __init__(self) -> None:
        self._cv2 = None
        self._cascade_path: str | None = None
        self._eye_cascade_path: str | None = None
        self._import_error: str | None = None
        try:
            import cv2  # type: ignore

            self._cv2 = cv2
        except Exception as exc:  # pragma: no cover - environment dependent
            self._import_error = f"{exc.__class__.__name__}: {exc!r}"
            return
        self._cascade_path = _resolve_haar_cascade_path(self._cv2)
        self._eye_cascade_path = _resolve_haar_cascade_path(self._cv2, filename="haarcascade_eye_tree_eyeglasses.xml")
        if self._eye_cascade_path is None:
            self._eye_cascade_path = _resolve_haar_cascade_path(self._cv2, filename="haarcascade_eye.xml")

    @property
    def cascade_path(self) -> str | None:
        return self._cascade_path

    def is_available(self) -> bool:
        return bool(self._cv2 is not None and self._cascade_path and Path(self._cascade_path).exists())

    def unavailable_reason(self) -> str | None:
        if self._import_error:
            return f"opencv import failed: {self._import_error}"
        if not self._cascade_path:
            return "opencv haar cascade path unavailable"
        if not Path(self._cascade_path).exists():
            return f"opencv haar cascade file not found: {self._cascade_path}"
        return None

    def detect(self, image_path: Path) -> FaceDetectionEngineResult:
        if not self.is_available():
            return FaceDetectionEngineResult(status="engine_unavailable", error_message=self.unavailable_reason())
        assert self._cv2 is not None
        try:
            image = self._cv2.imread(str(image_path))
            if image is None:
                return FaceDetectionEngineResult(status="failed", error_message=f"cv2.imread returned None for {image_path.name}")
            height, width = image.shape[:2]
            gray = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2GRAY)
            cascade = self._cv2.CascadeClassifier(str(self._cascade_path))
            if cascade.empty():
                return FaceDetectionEngineResult(status="engine_unavailable", error_message="opencv haar cascade failed to load")
            min_face_size = _dynamic_min_face_size(width, height)
            faces, weights = _detect_haar_faces(cascade, gray, min_face_size=min_face_size)
            eye_cascade = self._load_eye_cascade()
            detections: list[FaceDetection] = []
            for index, (x, y, w, h) in enumerate(faces):
                if not _face_candidate_shape_ok(float(w), float(h)):
                    continue
                eye_count = _count_eyes(eye_cascade, gray, int(x), int(y), int(w), int(h)) if eye_cascade is not None else None
                # Haar face cascades are prone to slippers/posters/round objects.
                # If a local eye cascade is available, require at least one eye-like
                # feature before saving a candidate for review.
                if eye_cascade is not None and not eye_count:
                    continue
                score = weights[index] if index < len(weights) else None
                landmarks = {"eye_count": eye_count, "eye_filter": eye_cascade is not None}
                if score is not None:
                    landmarks["haar_weight"] = score
                detections.append(
                    FaceDetection(
                        bbox_x=float(x),
                        bbox_y=float(y),
                        bbox_w=float(w),
                        bbox_h=float(h),
                        detection_score=score,
                        landmarks=landmarks,
                    )
                )
            if not detections:
                return FaceDetectionEngineResult(status="no_face_detected", image_width=width, image_height=height)
            return FaceDetectionEngineResult(status="success", detections=detections, image_width=width, image_height=height)
        except Exception as exc:  # pragma: no cover - depends on image/OpenCV details
            tail = "".join(traceback.format_exception(exc.__class__, exc, exc.__traceback__, limit=4))
            return FaceDetectionEngineResult(status="failed", error_message=tail[-1500:])

    def _load_eye_cascade(self):
        if self._cv2 is None or not self._eye_cascade_path or not Path(self._eye_cascade_path).exists():
            return None
        cascade = self._cv2.CascadeClassifier(str(self._eye_cascade_path))
        return None if cascade.empty() else cascade


class OpenCvYuNetFaceDetector(FaceDetector):
    name = "opencv_yunet"
    model_name = "opencv_yunet"

    def __init__(
        self,
        *,
        model_path: str | None,
        score_threshold: float = 0.85,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        max_input_size: int | None = 1280,
    ) -> None:
        self.model_path = str(Path(model_path).expanduser()) if model_path else None
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.max_input_size = int(max_input_size) if max_input_size else None
        self._cv2 = None
        self._import_error: str | None = None
        self._detector = None
        self._input_size: tuple[int, int] | None = None
        try:
            import cv2  # type: ignore

            self._cv2 = cv2
        except Exception as exc:  # pragma: no cover - environment dependent
            self._import_error = f"{exc.__class__.__name__}: {exc!r}"

    def is_available(self) -> bool:
        if self._cv2 is None or not self.model_path or not Path(self.model_path).exists():
            return False
        return hasattr(self._cv2, "FaceDetectorYN_create")

    def unavailable_reason(self) -> str | None:
        if self._import_error:
            return f"opencv import failed: {self._import_error}"
        if not self.model_path:
            return "opencv_yunet model_path is not configured"
        if not Path(self.model_path).exists():
            return f"opencv_yunet model file not found: {Path(self.model_path).name}"
        if self._cv2 is not None and not hasattr(self._cv2, "FaceDetectorYN_create"):
            return "cv2.FaceDetectorYN_create is unavailable"
        return None

    def detect(self, image_path: Path) -> FaceDetectionEngineResult:
        if not self.is_available():
            return FaceDetectionEngineResult(status="engine_unavailable", error_message=self.unavailable_reason())
        assert self._cv2 is not None
        try:
            image = self._cv2.imread(str(image_path))
            if image is None:
                return FaceDetectionEngineResult(status="failed", error_message=f"cv2.imread returned None for {image_path.name}")
            height, width = image.shape[:2]
            input_image, scale_x, scale_y = self._resize_for_detection(image)
            input_height, input_width = input_image.shape[:2]
            detector = self._get_detector((int(input_width), int(input_height)))
            _retval, faces = detector.detect(input_image)
            if faces is None or len(faces) == 0:
                return FaceDetectionEngineResult(status="no_face_detected", image_width=width, image_height=height)
            detections: list[FaceDetection] = []
            for face in faces:
                values = [float(value) for value in face.flatten().tolist()]
                if len(values) < 5:
                    continue
                x, y, w, h = values[:4]
                x *= scale_x
                y *= scale_y
                w *= scale_x
                h *= scale_y
                score = values[14] if len(values) > 14 else values[4]
                landmarks = _yunet_landmarks(values, scale_x=scale_x, scale_y=scale_y)
                detections.append(
                    FaceDetection(
                        bbox_x=x,
                        bbox_y=y,
                        bbox_w=w,
                        bbox_h=h,
                        detection_score=score,
                        landmarks=landmarks,
                    )
                )
            if not detections:
                return FaceDetectionEngineResult(status="no_face_detected", image_width=width, image_height=height)
            return FaceDetectionEngineResult(status="success", detections=detections, image_width=width, image_height=height)
        except Exception as exc:  # pragma: no cover - depends on OpenCV build/model
            tail = "".join(traceback.format_exception(exc.__class__, exc, exc.__traceback__, limit=4))
            return FaceDetectionEngineResult(status="failed", error_message=tail[-1500:])

    def _get_detector(self, input_size: tuple[int, int]):
        assert self._cv2 is not None
        if self._detector is None:
            self._detector = self._cv2.FaceDetectorYN_create(
                self.model_path,
                "",
                input_size,
                self.score_threshold,
                self.nms_threshold,
                self.top_k,
            )
            self._input_size = input_size
            return self._detector
        if self._input_size != input_size:
            self._detector.setInputSize(input_size)
            self._input_size = input_size
        return self._detector

    def _resize_for_detection(self, image):
        if not self.max_input_size:
            return image, 1.0, 1.0
        height, width = image.shape[:2]
        longest = max(width, height)
        if longest <= self.max_input_size:
            return image, 1.0, 1.0
        assert self._cv2 is not None
        ratio = self.max_input_size / float(longest)
        resized_width = max(1, int(round(width * ratio)))
        resized_height = max(1, int(round(height * ratio)))
        resized = self._cv2.resize(image, (resized_width, resized_height), interpolation=self._cv2.INTER_AREA)
        return resized, width / float(resized_width), height / float(resized_height)


def get_face_detector(
    name: str | None,
    *,
    model_path: str | None = None,
    score_threshold: float = 0.85,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
    max_input_size: int | None = 1280,
) -> FaceDetector:
    engine_name = (name or "opencv_haar").strip().lower()
    if engine_name == "fake":
        return FakeFaceDetector()
    if engine_name == "noop":
        return NoopFaceDetector()
    if engine_name in {"opencv_yunet", "yunet", "opencv_facedetectoryn"}:
        return OpenCvYuNetFaceDetector(
            model_path=model_path,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
            max_input_size=max_input_size,
        )
    if engine_name in {"opencv_haar", "haar"}:
        return OpenCvHaarFaceDetector()
    return NoopFaceDetector()


def opencv_diagnostics(*, yunet_model_path: str | None = None) -> dict[str, object]:
    detector = OpenCvHaarFaceDetector()
    yunet = OpenCvYuNetFaceDetector(model_path=yunet_model_path)
    cv2_version = None
    if detector._cv2 is not None:
        cv2_version = str(getattr(detector._cv2, "__version__", "unknown"))
    return {
        "opencv_import_ok": detector._cv2 is not None,
        "cv2_version": cv2_version,
        "haar_model_path": detector.cascade_path,
        "haar_model_exists": bool(detector.cascade_path and Path(detector.cascade_path).exists()),
        "yunet_model_path": yunet.model_path,
        "yunet_model_exists": bool(yunet.model_path and Path(yunet.model_path).exists()),
        "yunet_available": yunet.is_available(),
        "yunet_unavailable_reason": yunet.unavailable_reason(),
        "unavailable_reason": detector.unavailable_reason(),
    }


def _resolve_haar_cascade_path(
    cv2_module: object,
    *,
    filename: str = "haarcascade_frontalface_default.xml",
    sys_prefix: str | None = None,
) -> str | None:
    """Find OpenCV's bundled Haar cascade XML without downloading anything."""
    candidates: list[Path] = []
    data = getattr(cv2_module, "data", None)
    haar_dir = getattr(data, "haarcascades", None) if data is not None else None
    if haar_dir:
        candidates.append(Path(str(haar_dir)) / filename)

    prefix = Path(sys_prefix or sys.prefix)
    candidates.extend(
        [
            prefix / "share" / "opencv4" / "haarcascades" / filename,
            prefix / "share" / "OpenCV" / "haarcascades" / filename,
            prefix / "Library" / "etc" / "haarcascades" / filename,
        ]
    )

    cv2_file = getattr(cv2_module, "__file__", None)
    if cv2_file:
        base = Path(str(cv2_file)).resolve().parent
        candidates.extend(
            [
                base / "data" / filename,
                base.parent / "share" / "opencv4" / "haarcascades" / filename,
            ]
        )

    candidates.extend(
        [
            Path("/usr/share/opencv4/haarcascades") / filename,
            Path("/usr/local/share/opencv4/haarcascades") / filename,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _dynamic_min_face_size(width: int, height: int) -> tuple[int, int]:
    short_side = max(1, min(width, height))
    size = max(36, int(short_side * 0.035))
    return size, size


def _detect_haar_faces(cascade, gray, *, min_face_size: tuple[int, int]) -> tuple[list[tuple[int, int, int, int]], list[float | None]]:
    try:
        faces, _reject_levels, level_weights = cascade.detectMultiScale3(
            gray,
            scaleFactor=1.08,
            minNeighbors=7,
            minSize=min_face_size,
            outputRejectLevels=True,
        )
        return [tuple(int(v) for v in face) for face in faces], [float(weight) for weight in level_weights]
    except Exception:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=7, minSize=min_face_size)
        return [tuple(int(v) for v in face) for face in faces], [None for _ in faces]


def _face_candidate_shape_ok(width: float, height: float) -> bool:
    if width <= 0 or height <= 0:
        return False
    ratio = width / height
    return 0.72 <= ratio <= 1.38


def _count_eyes(eye_cascade, gray, x: int, y: int, w: int, h: int) -> int:
    if eye_cascade is None or w <= 0 or h <= 0:
        return 0
    upper = gray[max(0, y) : max(0, y) + max(1, int(h * 0.68)), max(0, x) : max(0, x) + w]
    if upper.size == 0:
        return 0
    min_eye = max(6, int(min(w, h) * 0.12))
    eyes = eye_cascade.detectMultiScale(upper, scaleFactor=1.08, minNeighbors=3, minSize=(min_eye, min_eye))
    return int(len(eyes))


def _yunet_landmarks(values: list[float], *, scale_x: float = 1.0, scale_y: float = 1.0) -> dict[str, list[float]]:
    if len(values) < 14:
        return {}
    names = ["right_eye", "left_eye", "nose_tip", "right_mouth", "left_mouth"]
    coords = values[4:14]
    return {name: [coords[index * 2] * scale_x, coords[index * 2 + 1] * scale_y] for index, name in enumerate(names)}
