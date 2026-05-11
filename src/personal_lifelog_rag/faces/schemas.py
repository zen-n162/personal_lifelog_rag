"""Schemas for local face detection rows and reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FACE_STATUSES = {"success", "failed", "skipped", "engine_unavailable", "no_face_detected"}
FACE_REVIEW_STATUSES = {"unreviewed", "accepted", "rejected", "bad_detection"}
FACE_PRIVACY_LEVELS = {"private"}
FACE_EMBEDDING_STATUSES = {"success", "failed", "skipped", "engine_unavailable"}
FACE_CLUSTER_STATUSES = {"unreviewed", "accepted", "rejected", "merged", "split"}
FACE_CLUSTER_REVIEW_STATUSES = {"unreviewed", "reviewed", "bad_cluster"}


@dataclass
class FaceDetection:
    """One detected face bbox from a local detector."""

    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    detection_score: float | None = None
    landmarks: dict[str, Any] | list[Any] | None = None


@dataclass
class FaceDetectionEngineResult:
    """Raw detector result before persistence."""

    status: str
    detections: list[FaceDetection] = field(default_factory=list)
    image_width: int | None = None
    image_height: int | None = None
    error_message: str | None = None


@dataclass
class FaceDetectOptions:
    date: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    limit: int = 100
    engine: str = "opencv_haar"
    model_path: str | None = None
    score_threshold: float = 0.85
    nms_threshold: float = 0.3
    top_k: int = 5000
    max_input_size: int | None = 1280
    dry_run: bool = False
    skip_existing: bool = False
    force: bool = False
    min_score: float | None = None
    save_crops: bool = False
    only_existing_files: bool = True
    include_hidden: bool = False


@dataclass
class FaceDetectReport:
    run_id: str
    engine: str
    model_name: str
    date_from: str | None
    date_to: str | None
    selected_count: int = 0
    processed_count: int = 0
    success_count: int = 0
    no_face_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    engine_unavailable_count: int = 0
    crop_count: int = 0
    dry_run: bool = False
    output_dirs: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "engine": self.engine,
            "model_name": self.model_name,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "selected_count": self.selected_count,
            "processed_count": self.processed_count,
            "success_count": self.success_count,
            "no_face_count": self.no_face_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "engine_unavailable_count": self.engine_unavailable_count,
            "crop_count": self.crop_count,
            "dry_run": self.dry_run,
            "output_dirs": self.output_dirs,
            "errors": self.errors,
        }


@dataclass
class FaceDiagnostics:
    available_engines: list[str]
    opencv_import_ok: bool
    cv2_version: str | None
    haar_model_path: str | None
    haar_model_exists: bool
    faces_dir: Path
    yunet_model_path: str | None = None
    yunet_model_exists: bool = False
    yunet_available: bool = False
    yunet_unavailable_reason: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available_engines": self.available_engines,
            "opencv_import_ok": self.opencv_import_ok,
            "cv2_version": self.cv2_version,
            "haar_model_path": self.haar_model_path,
            "haar_model_exists": self.haar_model_exists,
            "yunet_model_path": self.yunet_model_path,
            "yunet_model_exists": self.yunet_model_exists,
            "yunet_available": self.yunet_available,
            "yunet_unavailable_reason": self.yunet_unavailable_reason,
            "faces_dir": str(self.faces_dir),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class FaceDetectionConfig:
    engine: str = "opencv_haar"
    model_path: str | None = None
    local_only: bool = True
    score_threshold: float = 0.85
    nms_threshold: float = 0.3
    top_k: int = 5000
    max_input_size: int | None = 1280


@dataclass(frozen=True)
class FaceEmbeddingConfig:
    engine: str = "opencv_sface"
    model_path: str | None = None
    embedding_dim: int | None = 128
    local_only: bool = True
    normalize: bool = True


@dataclass(frozen=True)
class FaceClusteringConfig:
    method: str = "dbscan_cosine"
    distance_threshold: float = 0.45
    min_samples: int = 2


@dataclass
class FaceEmbeddingOptions:
    date: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    limit: int = 100
    engine: str = "opencv_sface"
    detections_engine: str | None = None
    status: str = "success"
    model_path: str | None = None
    embedding_dim: int | None = 128
    normalize: bool = True
    dry_run: bool = False
    skip_existing: bool = False
    force: bool = False
    replace: bool = False
    batch_size: int = 500
    only_with_crop: bool = False
    only_existing_files: bool = False
    only_reviewed_detections: bool = False
    include_unreviewed_detections: bool = True
    min_detection_score: float | None = None


@dataclass
class FaceEmbeddingEngineResult:
    status: str
    vector: list[float] = field(default_factory=list)
    error_message: str | None = None


@dataclass
class FaceEmbeddingReport:
    engine: str
    model_name: str
    date_from: str | None
    date_to: str | None
    selected_count: int = 0
    existing_embedding_count: int = 0
    deleted_embedding_count: int = 0
    processed_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    engine_unavailable_count: int = 0
    dry_run: bool = False
    replace: bool = False
    batch_size: int = 500
    detections_engine: str | None = None
    target_status: str = "success"
    only_with_crop: bool = False
    only_existing_files: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "model_name": self.model_name,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "selected_count": self.selected_count,
            "existing_embedding_count": self.existing_embedding_count,
            "deleted_embedding_count": self.deleted_embedding_count,
            "processed_count": self.processed_count,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "engine_unavailable_count": self.engine_unavailable_count,
            "dry_run": self.dry_run,
            "replace": self.replace,
            "batch_size": self.batch_size,
            "detections_engine": self.detections_engine,
            "target_status": self.target_status,
            "only_with_crop": self.only_with_crop,
            "only_existing_files": self.only_existing_files,
            "errors": self.errors,
        }


@dataclass
class FaceEmbeddingDiagnostics:
    selected_engine: str
    opencv_import_ok: bool
    cv2_version: str | None
    model_path_configured: str | None
    model_file_exists: bool
    local_only: bool
    embedding_dim: int | None
    engine_available: bool
    unavailable_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_engine": self.selected_engine,
            "opencv_import_ok": self.opencv_import_ok,
            "cv2_version": self.cv2_version,
            "model_path_configured": self.model_path_configured,
            "model_file_exists": self.model_file_exists,
            "local_only": self.local_only,
            "embedding_dim": self.embedding_dim,
            "engine_available": self.engine_available,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass
class FaceClusteringOptions:
    date: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    method: str = "dbscan_cosine"
    distance_threshold: float = 0.45
    min_samples: int = 2
    dry_run: bool = False
    replace: bool = False
    scope: str | None = None
    embedding_model: str | None = None


@dataclass
class FaceClusterCandidate:
    cluster_label: str
    face_ids: list[str]
    representative_face_id: str | None
    first_seen_at: str | None
    last_seen_at: str | None
    confidence: float | None
    distances: dict[str, float] = field(default_factory=dict)


@dataclass
class FaceClusterReport:
    method: str
    distance_threshold: float
    min_samples: int
    date_from: str | None
    date_to: str | None
    selected_embeddings: int = 0
    replace_count: int = 0
    cluster_candidates: int = 0
    clusters_written: int = 0
    members_written: int = 0
    singleton_count: int = 0
    largest_cluster_size: int = 0
    dry_run: bool = False
    replace: bool = False
    scope: str | None = None
    embedding_model: str | None = None
    candidates: list[FaceClusterCandidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "distance_threshold": self.distance_threshold,
            "min_samples": self.min_samples,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "selected_embeddings": self.selected_embeddings,
            "replace_count": self.replace_count,
            "cluster_candidates": self.cluster_candidates,
            "clusters_written": self.clusters_written,
            "members_written": self.members_written,
            "singleton_count": self.singleton_count,
            "largest_cluster_size": self.largest_cluster_size,
            "dry_run": self.dry_run,
            "replace": self.replace,
            "scope": self.scope,
            "embedding_model": self.embedding_model,
            "candidates": [
                {
                    "cluster_label": candidate.cluster_label,
                    "face_ids": candidate.face_ids,
                    "representative_face_id": candidate.representative_face_id,
                    "first_seen_at": candidate.first_seen_at,
                    "last_seen_at": candidate.last_seen_at,
                    "confidence": candidate.confidence,
                }
                for candidate in self.candidates
            ],
        }
