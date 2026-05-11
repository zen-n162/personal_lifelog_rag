"""Local-only face embedding engines.

Embeddings are sensitive biometric-adjacent data. This module never downloads
models and never sends images or vectors over the network.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import traceback

from personal_lifelog_rag.embeddings.similarity import normalize
from personal_lifelog_rag.faces.schemas import FaceEmbeddingEngineResult


class FaceEmbeddingEngine:
    name = "base"
    model_name = "none"

    def is_available(self) -> bool:
        return False

    def unavailable_reason(self) -> str | None:
        return "engine is not implemented"

    def embed_face(self, image_path: Path, *, face_id: str) -> FaceEmbeddingEngineResult:
        return FaceEmbeddingEngineResult(status="engine_unavailable", error_message=self.unavailable_reason())


class FakeFaceEmbeddingEngine(FaceEmbeddingEngine):
    name = "fake"
    model_name = "fake_face_embedding"

    def __init__(self, *, embedding_dim: int = 8, normalize_vector: bool = True) -> None:
        self.embedding_dim = embedding_dim
        self.normalize_vector = normalize_vector

    def is_available(self) -> bool:
        return True

    def unavailable_reason(self) -> str | None:
        return None

    def embed_face(self, image_path: Path, *, face_id: str) -> FaceEmbeddingEngineResult:
        digest = hashlib.sha256(face_id.encode("utf-8")).digest()
        vector = [((digest[index % len(digest)] / 255.0) * 2.0) - 1.0 for index in range(self.embedding_dim)]
        if self.normalize_vector:
            vector = normalize(vector)
        return FaceEmbeddingEngineResult(status="success", vector=vector)


class OpenCvSFaceEmbeddingEngine(FaceEmbeddingEngine):
    name = "opencv_sface"
    model_name = "opencv_sface"

    def __init__(self, *, model_path: str | None, embedding_dim: int | None = 128, normalize_vector: bool = True) -> None:
        self.model_path = str(Path(model_path).expanduser()) if model_path else None
        self.embedding_dim = embedding_dim
        self.normalize_vector = normalize_vector
        self._cv2 = None
        self._recognizer = None
        self._import_error: str | None = None
        try:
            import cv2  # type: ignore

            self._cv2 = cv2
        except Exception as exc:  # pragma: no cover - environment dependent
            self._import_error = f"{exc.__class__.__name__}: {exc!r}"

    def is_available(self) -> bool:
        if self._cv2 is None or not self.model_path or not Path(self.model_path).exists():
            return False
        return hasattr(self._cv2, "FaceRecognizerSF_create")

    def unavailable_reason(self) -> str | None:
        if self._import_error:
            return f"opencv import failed: {self._import_error}"
        if not self.model_path:
            return "opencv_sface model_path is not configured"
        if not Path(self.model_path).exists():
            return f"opencv_sface model file not found: {Path(self.model_path).name}"
        if self._cv2 is not None and not hasattr(self._cv2, "FaceRecognizerSF_create"):
            return "cv2.FaceRecognizerSF_create is unavailable"
        return None

    def embed_face(self, image_path: Path, *, face_id: str) -> FaceEmbeddingEngineResult:
        del face_id
        if not self.is_available():
            return FaceEmbeddingEngineResult(status="engine_unavailable", error_message=self.unavailable_reason())
        assert self._cv2 is not None
        try:
            if self._recognizer is None:
                self._recognizer = self._cv2.FaceRecognizerSF_create(self.model_path, "")
            image = self._cv2.imread(str(image_path))
            if image is None:
                return FaceEmbeddingEngineResult(status="failed", error_message=f"cv2.imread returned None for {image_path.name}")
            raw = self._recognizer.feature(image)
            vector = [float(value) for value in raw.flatten().tolist()]
            if self.embedding_dim and len(vector) != self.embedding_dim:
                # Keep the actual dim; diagnostics can reveal the mismatch.
                pass
            if self.normalize_vector:
                vector = normalize(vector)
            return FaceEmbeddingEngineResult(status="success", vector=vector)
        except Exception as exc:  # pragma: no cover - depends on OpenCV build/model
            tail = "".join(traceback.format_exception(exc.__class__, exc, exc.__traceback__, limit=4))
            return FaceEmbeddingEngineResult(status="failed", error_message=tail[-1500:])


def get_face_embedding_engine(
    name: str | None,
    *,
    model_path: str | None = None,
    embedding_dim: int | None = 128,
    normalize_vector: bool = True,
) -> FaceEmbeddingEngine:
    engine_name = (name or "opencv_sface").strip().lower()
    if engine_name == "fake":
        return FakeFaceEmbeddingEngine(embedding_dim=embedding_dim or 8, normalize_vector=normalize_vector)
    if engine_name in {"opencv_sface", "sface"}:
        return OpenCvSFaceEmbeddingEngine(model_path=model_path, embedding_dim=embedding_dim, normalize_vector=normalize_vector)
    return OpenCvSFaceEmbeddingEngine(model_path=model_path, embedding_dim=embedding_dim, normalize_vector=normalize_vector)


def opencv_sface_diagnostics(*, model_path: str | None) -> dict[str, object]:
    engine = OpenCvSFaceEmbeddingEngine(model_path=model_path)
    cv2_version = None
    if engine._cv2 is not None:
        cv2_version = str(getattr(engine._cv2, "__version__", "unknown"))
    return {
        "opencv_import_ok": engine._cv2 is not None,
        "cv2_version": cv2_version,
        "model_path_configured": engine.model_path,
        "model_file_exists": bool(engine.model_path and Path(engine.model_path).exists()),
        "engine_available": engine.is_available(),
        "unavailable_reason": engine.unavailable_reason(),
    }
