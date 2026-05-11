"""Service layer for local-only face detection and review."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any
import uuid

from PIL import Image, ImageOps

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.db.schema import initialize_schema
from personal_lifelog_rag.faces.engines import get_face_detector, opencv_diagnostics
from personal_lifelog_rag.faces.schemas import (
    FACE_REVIEW_STATUSES,
    FaceDetectOptions,
    FaceDetectReport,
    FaceDetection,
    FaceDetectionEngineResult,
    FaceDiagnostics,
)

DEFAULT_FACE_DIR = Path("data/faces")
DEFAULT_FACE_CROP_DIR = DEFAULT_FACE_DIR / "crops"
DEFAULT_FACE_THUMBNAIL_DIR = DEFAULT_FACE_DIR / "thumbnails"


def face_diagnostics(
    *,
    faces_dir: Path = DEFAULT_FACE_DIR,
    yunet_model_path: str | None = None,
) -> FaceDiagnostics:
    diag = opencv_diagnostics(yunet_model_path=yunet_model_path)
    available = ["fake"]
    if diag["haar_model_exists"]:
        available.append("opencv_haar")
    if diag["yunet_available"]:
        available.append("opencv_yunet")
    return FaceDiagnostics(
        available_engines=available,
        opencv_import_ok=bool(diag["opencv_import_ok"]),
        cv2_version=str(diag["cv2_version"]) if diag["cv2_version"] else None,
        haar_model_path=str(diag["haar_model_path"]) if diag["haar_model_path"] else None,
        haar_model_exists=bool(diag["haar_model_exists"]),
        faces_dir=faces_dir,
        yunet_model_path=str(diag["yunet_model_path"]) if diag["yunet_model_path"] else None,
        yunet_model_exists=bool(diag["yunet_model_exists"]),
        yunet_available=bool(diag["yunet_available"]),
        yunet_unavailable_reason=str(diag["yunet_unavailable_reason"]) if diag["yunet_unavailable_reason"] else None,
        notes=[
            "local-only face detection; no identity, emotion, or relationship inference",
            "model files are never downloaded automatically",
            "face crops are private local artifacts and are not used by normal QA/search/report flows",
        ],
    )


def run_face_detection(
    repository: LifelogRepository,
    options: FaceDetectOptions,
    *,
    faces_dir: Path = DEFAULT_FACE_DIR,
) -> FaceDetectReport:
    repository.initialize()
    detector = get_face_detector(
        options.engine,
        model_path=options.model_path,
        score_threshold=options.score_threshold,
        nms_threshold=options.nms_threshold,
        top_k=options.top_k,
        max_input_size=options.max_input_size,
    )
    date_from, date_to = _resolve_date_range(options)
    run_id = f"face_run_{uuid.uuid4().hex[:16]}"
    report = FaceDetectReport(
        run_id=run_id,
        engine=detector.name,
        model_name=detector.model_name,
        date_from=date_from,
        date_to=date_to,
        dry_run=options.dry_run,
        output_dirs={
            "crops": str(faces_dir / "crops"),
            "thumbnails": str(faces_dir / "thumbnails"),
        },
    )
    targets = _select_media(repository, date_from=date_from, date_to=date_to, limit=options.limit, include_hidden=options.include_hidden)
    report.selected_count = len(targets)
    if options.dry_run:
        if not detector.is_available():
            report.engine_unavailable_count = len(targets)
            reason = detector.unavailable_reason()
            if reason:
                report.errors.append(reason)
        return report

    crop_dir = faces_dir / "crops"
    thumbnail_dir = faces_dir / "thumbnails"
    if options.save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)
        thumbnail_dir.mkdir(parents=True, exist_ok=True)

    started_at = _now()
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        connection.execute(
            """
            INSERT INTO face_detection_runs (
                id, started_at, engine, model_name, date_from, date_to,
                selected_count, processed_count, success_count, no_face_count,
                failed_count, engine_unavailable_count, config_json, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, 0, ?, 'running')
            """,
            (
                run_id,
                started_at,
                detector.name,
                detector.model_name,
                date_from,
                date_to,
                len(targets),
                json.dumps({"save_crops": options.save_crops, "min_score": options.min_score}, ensure_ascii=False),
            ),
        )
        connection.commit()

    engine_available = detector.is_available()
    engine_unavailable_message = detector.unavailable_reason()
    existing_media_ids = set() if options.force else _existing_detection_media_ids(repository, engine=detector.name, final_only=options.skip_existing)
    with closing(connect(repository.db_path)) as detection_connection:
        initialize_schema(detection_connection)
        for row in targets:
            media_id = str(row["id"])
            image_path = Path(str(row["file_path"])).expanduser()
            if options.only_existing_files and not image_path.exists():
                report.skipped_count += 1
                continue
            if media_id in existing_media_ids and not options.force:
                report.skipped_count += 1
                continue
            if options.force:
                _delete_existing_detections(repository, media_id=media_id, engine=detector.name)
            report.processed_count += 1
            if not engine_available:
                result = FaceDetectionEngineResult(status="engine_unavailable", error_message=engine_unavailable_message)
            else:
                result = detector.detect(image_path)
            saved = _persist_detection_result_in_connection(
                detection_connection,
                media_id=media_id,
                image_path=image_path,
                engine=detector.name,
                model_name=detector.model_name,
                result=result,
                min_score=options.min_score,
                save_crops=options.save_crops,
                crop_dir=crop_dir,
                thumbnail_dir=thumbnail_dir,
            )
            if not options.force:
                existing_media_ids.add(media_id)
            report.crop_count += saved["crop_count"]
            if result.status == "success" and saved["success_count"] > 0:
                report.success_count += saved["success_count"]
            elif result.status == "no_face_detected" or (result.status == "success" and saved["success_count"] == 0):
                report.no_face_count += 1
            elif result.status == "engine_unavailable":
                report.engine_unavailable_count += 1
            else:
                report.failed_count += 1
                if result.error_message:
                    report.errors.append(f"{media_id}: {result.error_message[:240]}")
            if report.processed_count % 100 == 0:
                detection_connection.commit()
        detection_connection.commit()

    status = "completed" if not (report.failed_count or report.engine_unavailable_count) else "partial"
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        connection.execute(
            """
            UPDATE face_detection_runs
            SET finished_at = ?, processed_count = ?, success_count = ?, no_face_count = ?,
                failed_count = ?, engine_unavailable_count = ?, status = ?
            WHERE id = ?
            """,
            (
                _now(),
                report.processed_count,
                report.success_count,
                report.no_face_count,
                report.failed_count,
                report.engine_unavailable_count,
                status,
                run_id,
            ),
        )
        connection.commit()
    return report


def face_stats(repository: LifelogRepository, *, date_from: str | None = None, date_to: str | None = None) -> dict[str, Any]:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        where, params = _face_date_where(date_from, date_to)
        return {
            "total": _count(connection, f"SELECT COUNT(*) FROM face_detections {where}", params),
            "status_counts": _rows(connection, f"SELECT status, COUNT(*) AS count FROM face_detections {where} GROUP BY status ORDER BY count DESC", params),
            "review_status_counts": _rows(connection, f"SELECT review_status, COUNT(*) AS count FROM face_detections {where} GROUP BY review_status ORDER BY count DESC", params),
            "daily_counts": _rows(
                connection,
                f"""
                SELECT SUBSTR(COALESCE(media_items.captured_at, media_items.fallback_captured_at), 1, 10) AS date,
                       COUNT(*) AS count
                FROM face_detections
                JOIN media_items ON media_items.id = face_detections.media_id
                {_join_date_filter(where)}
                GROUP BY date
                ORDER BY date ASC
                """,
                params,
            ),
            "top_dates": _rows(
                connection,
                f"""
                SELECT SUBSTR(COALESCE(media_items.captured_at, media_items.fallback_captured_at), 1, 10) AS date,
                       COUNT(*) AS count
                FROM face_detections
                JOIN media_items ON media_items.id = face_detections.media_id
                {_join_date_filter(where)}
                GROUP BY date
                ORDER BY count DESC, date ASC
                LIMIT 10
                """,
                params,
            ),
            "crop_count": _count(connection, f"SELECT COUNT(*) FROM face_detections {where} {'AND' if where else 'WHERE'} crop_path IS NOT NULL AND crop_path != ''", params),
        }


def list_face_detections(
    repository: LifelogRepository,
    *,
    date: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    status: str | None = None,
    review_status: str | None = None,
    min_score: float | None = None,
    has_crop: bool = False,
    limit: int = 20,
) -> list[dict[str, Any]]:
    start = date or date_from
    end = date or date_to
    clauses: list[str] = []
    params: list[Any] = []
    if start:
        clauses.append("SUBSTR(COALESCE(media_items.captured_at, media_items.fallback_captured_at), 1, 10) >= ?")
        params.append(start)
    if end:
        clauses.append("SUBSTR(COALESCE(media_items.captured_at, media_items.fallback_captured_at), 1, 10) <= ?")
        params.append(end)
    if status:
        clauses.append("face_detections.status = ?")
        params.append(status)
    if review_status:
        clauses.append("face_detections.review_status = ?")
        params.append(review_status)
    if min_score is not None:
        clauses.append("face_detections.detection_score IS NOT NULL AND face_detections.detection_score >= ?")
        params.append(min_score)
    if has_crop:
        clauses.append("face_detections.crop_path IS NOT NULL AND face_detections.crop_path != ''")
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(limit)
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        return _rows(
            connection,
            f"""
            SELECT face_detections.*, media_items.file_name, media_items.file_path,
                   media_items.thumbnail_path AS media_thumbnail_path,
                   media_items.captured_at, media_items.fallback_captured_at
            FROM face_detections
            JOIN media_items ON media_items.id = face_detections.media_id
            {where}
            ORDER BY COALESCE(media_items.captured_at, media_items.fallback_captured_at) DESC,
                     face_detections.updated_at DESC
            LIMIT ?
            """,
            params,
        )


def update_face_review_status(repository: LifelogRepository, *, face_id: str, review_status: str) -> dict[str, Any]:
    if review_status not in FACE_REVIEW_STATUSES:
        raise ValueError(f"unknown face review_status: {review_status}")
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        connection.execute(
            "UPDATE face_detections SET review_status = ?, updated_at = ? WHERE id = ?",
            (review_status, _now(), face_id),
        )
        connection.commit()
        row = connection.execute("SELECT * FROM face_detections WHERE id = ?", (face_id,)).fetchone()
        if row is None:
            raise ValueError(f"face detection not found: {face_id}")
        return dict(row)


def ensure_face_thumbnail(
    repository: LifelogRepository,
    *,
    face_id: str,
    faces_dir: Path = DEFAULT_FACE_DIR,
) -> dict[str, str | None]:
    """Create a private crop/thumbnail for an existing success row if missing."""
    crop_dir = faces_dir / "crops"
    thumbnail_dir = faces_dir / "thumbnails"
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        row = connection.execute(
            """
            SELECT face_detections.*, media_items.file_path
            FROM face_detections
            JOIN media_items ON media_items.id = face_detections.media_id
            WHERE face_detections.id = ?
            """,
            (face_id,),
        ).fetchone()
        if row is None:
            return {"crop_path": None, "thumbnail_path": None}
        face = dict(row)
        thumbnail_path = str(face.get("thumbnail_path") or "")
        crop_path = str(face.get("crop_path") or "")
        if thumbnail_path and Path(thumbnail_path).expanduser().exists():
            return {"crop_path": crop_path or None, "thumbnail_path": thumbnail_path}
        if face.get("status") != "success" or face.get("bbox_x") is None:
            return {"crop_path": crop_path or None, "thumbnail_path": thumbnail_path or None}
        image_path = Path(str(face.get("file_path") or "")).expanduser()
        if not image_path.exists():
            return {"crop_path": crop_path or None, "thumbnail_path": thumbnail_path or None}

        crop_dir.mkdir(parents=True, exist_ok=True)
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        detection = FaceDetection(
            bbox_x=float(face["bbox_x"]),
            bbox_y=float(face["bbox_y"]),
            bbox_w=float(face["bbox_w"]),
            bbox_h=float(face["bbox_h"]),
            detection_score=face.get("detection_score"),
        )
        saved = _save_crop(image_path, detection, face_id=face_id, crop_dir=crop_dir, thumbnail_dir=thumbnail_dir)
        if saved:
            now = _now()
            connection.execute(
                "UPDATE face_detections SET crop_path = ?, thumbnail_path = ?, updated_at = ? WHERE id = ?",
                (saved.get("crop_path"), saved.get("thumbnail_path"), now, face_id),
            )
            connection.commit()
        return {"crop_path": saved.get("crop_path") or crop_path or None, "thumbnail_path": saved.get("thumbnail_path") or thumbnail_path or None}


def format_face_diagnostics(diag: FaceDiagnostics) -> str:
    lines = ["Face diagnostics", f"- available engines: {', '.join(diag.available_engines) or 'none'}"]
    lines.extend(
        [
            f"- opencv import: {'ok' if diag.opencv_import_ok else 'unavailable'}",
            f"- cv2 version: {diag.cv2_version or 'unknown'}",
            f"- haar model path: {diag.haar_model_path or 'unavailable'}",
            f"- haar model exists: {diag.haar_model_exists}",
            f"- yunet model path: {diag.yunet_model_path or 'not configured'}",
            f"- yunet model exists: {diag.yunet_model_exists}",
            f"- yunet available: {diag.yunet_available}",
            f"- yunet unavailable reason: {diag.yunet_unavailable_reason or '-'}",
            f"- data/faces path: {diag.faces_dir}",
            "- local-only: true",
        ]
    )
    lines.extend(f"- note: {note}" for note in diag.notes)
    return "\n".join(lines)


def format_face_detect_report(report: FaceDetectReport) -> str:
    return "\n".join(
        [
            "Face detection report",
            f"- run_id: {report.run_id}",
            f"- engine: {report.engine}",
            f"- model: {report.model_name}",
            f"- date range: {report.date_from or '-'} .. {report.date_to or '-'}",
            f"- dry_run: {report.dry_run}",
            f"- selected: {report.selected_count}",
            f"- processed: {report.processed_count}",
            f"- success faces: {report.success_count}",
            f"- no_face_detected media: {report.no_face_count}",
            f"- failed media: {report.failed_count}",
            f"- engine_unavailable media: {report.engine_unavailable_count}",
            f"- skipped media: {report.skipped_count}",
            f"- crops saved: {report.crop_count}",
            f"- crop dir: {report.output_dirs.get('crops', '')}",
            f"- thumbnail dir: {report.output_dirs.get('thumbnails', '')}",
        ]
        + ([f"- first error: {report.errors[0]}"] if report.errors else [])
    )


def format_face_stats(stats: dict[str, Any]) -> str:
    lines = ["Face stats", f"- total: {stats['total']}", "- status counts:"]
    lines.extend(_count_lines(stats["status_counts"], "status"))
    lines.append("- review_status counts:")
    lines.extend(_count_lines(stats["review_status_counts"], "review_status"))
    lines.append("- top dates:")
    lines.extend(_count_lines(stats["top_dates"], "date"))
    lines.append(f"- crop count: {stats['crop_count']}")
    return "\n".join(lines)


def format_face_rows(rows: list[dict[str, Any]], *, show_errors: bool = False) -> str:
    lines = ["Face detections"]
    if not rows:
        return "Face detections\n- none"
    for row in rows:
        bbox = ""
        if row.get("bbox_x") is not None:
            bbox = f"bbox=({row.get('bbox_x'):.0f},{row.get('bbox_y'):.0f},{row.get('bbox_w'):.0f},{row.get('bbox_h'):.0f})"
        lines.append(
            " - ".join(
                part
                for part in [
                    str(row.get("id") or ""),
                    str(row.get("media_id") or ""),
                    str(row.get("captured_at") or row.get("fallback_captured_at") or ""),
                    str(row.get("file_name") or ""),
                    str(row.get("status") or ""),
                    bbox,
                    f"score={row.get('detection_score')}" if row.get("detection_score") is not None else "",
                    f"review={row.get('review_status') or ''}",
                    f"crop={row.get('crop_path') or ''}" if row.get("crop_path") else "",
                ]
                if part
            )
        )
        if show_errors and row.get("error_message"):
            lines.append(f"   error: {str(row.get('error_message'))[:500]}")
    return "\n".join(lines)


def _select_media(
    repository: LifelogRepository,
    *,
    date_from: str | None,
    date_to: str | None,
    limit: int,
    include_hidden: bool,
) -> list[dict[str, Any]]:
    clauses = ["media_items.media_type LIKE 'image%'"]
    params: list[Any] = []
    if date_from:
        clauses.append("SUBSTR(COALESCE(media_items.captured_at, media_items.fallback_captured_at), 1, 10) >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("SUBSTR(COALESCE(media_items.captured_at, media_items.fallback_captured_at), 1, 10) <= ?")
        params.append(date_to)
    if not include_hidden:
        clauses.append("COALESCE(media_vlm_overrides.is_hidden, 0) = 0")
    params.append(limit)
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        return _rows(
            connection,
            f"""
            SELECT media_items.*
            FROM media_items
            LEFT JOIN media_vlm_overrides ON media_vlm_overrides.media_id = media_items.id
            WHERE {' AND '.join(clauses)}
            ORDER BY COALESCE(media_items.captured_at, media_items.fallback_captured_at) ASC, media_items.id ASC
            LIMIT ?
            """,
            params,
        )


def _persist_detection_result(
    repository: LifelogRepository,
    *,
    media_id: str,
    image_path: Path,
    engine: str,
    model_name: str,
    result: FaceDetectionEngineResult,
    min_score: float | None,
    save_crops: bool,
    crop_dir: Path,
    thumbnail_dir: Path,
) -> dict[str, int]:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        result_counts = _persist_detection_result_in_connection(
            connection,
            media_id=media_id,
            image_path=image_path,
            engine=engine,
            model_name=model_name,
            result=result,
            min_score=min_score,
            save_crops=save_crops,
            crop_dir=crop_dir,
            thumbnail_dir=thumbnail_dir,
        )
        connection.commit()
    return result_counts


def _persist_detection_result_in_connection(
    connection,
    *,
    media_id: str,
    image_path: Path,
    engine: str,
    model_name: str,
    result: FaceDetectionEngineResult,
    min_score: float | None,
    save_crops: bool,
    crop_dir: Path,
    thumbnail_dir: Path,
) -> dict[str, int]:
    now = _now()
    crop_count = 0
    success_count = 0
    if result.status == "success":
        detections = []
        for detection in result.detections:
            if min_score is not None and detection.detection_score is not None and detection.detection_score < min_score:
                continue
            clipped = _clip_detection_to_image(detection, image_width=result.image_width, image_height=result.image_height)
            if clipped is not None:
                detections.append(clipped)
        if not detections:
            _insert_status_row(
                connection,
                face_id=_face_id(media_id, engine, "no_face_after_filter"),
                media_id=media_id,
                engine=engine,
                model_name=model_name,
                status="no_face_detected",
                result=result,
                now=now,
            )
        for index, detection in enumerate(detections):
            face_id = _face_id(media_id, engine, f"{index}:{detection.bbox_x:.1f}:{detection.bbox_y:.1f}:{detection.bbox_w:.1f}:{detection.bbox_h:.1f}")
            crop_path = None
            thumb_path = None
            if save_crops:
                saved_paths = _save_crop(image_path, detection, face_id=face_id, crop_dir=crop_dir, thumbnail_dir=thumbnail_dir)
                crop_path = saved_paths.get("crop_path")
                thumb_path = saved_paths.get("thumbnail_path")
                if crop_path:
                    crop_count += 1
            connection.execute(
                """
                INSERT OR REPLACE INTO face_detections (
                    id, media_id, detected_at, engine, model_name, status,
                    bbox_x, bbox_y, bbox_w, bbox_h, landmarks_json, detection_score,
                    image_width, image_height, crop_path, thumbnail_path,
                    privacy_level, review_status, error_message, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, 'success', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'private', 'unreviewed', NULL, ?, ?)
                """,
                (
                    face_id,
                    media_id,
                    now,
                    engine,
                    model_name,
                    detection.bbox_x,
                    detection.bbox_y,
                    detection.bbox_w,
                    detection.bbox_h,
                    json.dumps(detection.landmarks, ensure_ascii=False) if detection.landmarks is not None else None,
                    detection.detection_score,
                    result.image_width,
                    result.image_height,
                    crop_path,
                    thumb_path,
                    now,
                    now,
                ),
            )
            success_count += 1
    else:
        status = result.status if result.status in {"failed", "engine_unavailable", "no_face_detected"} else "failed"
        _insert_status_row(
            connection,
            face_id=_face_id(media_id, engine, status),
            media_id=media_id,
            engine=engine,
            model_name=model_name,
            status=status,
            result=result,
            now=now,
        )
    return {"crop_count": crop_count, "success_count": success_count}


def _insert_status_row(
    connection,
    *,
    face_id: str,
    media_id: str,
    engine: str,
    model_name: str,
    status: str,
    result: FaceDetectionEngineResult,
    now: str,
) -> None:
    connection.execute(
        """
        INSERT OR REPLACE INTO face_detections (
            id, media_id, detected_at, engine, model_name, status,
            bbox_x, bbox_y, bbox_w, bbox_h, landmarks_json, detection_score,
            image_width, image_height, crop_path, thumbnail_path,
            privacy_level, review_status, error_message, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?, NULL, NULL, 'private', 'unreviewed', ?, ?, ?)
        """,
        (
            face_id,
            media_id,
            now,
            engine,
            model_name,
            status,
            result.image_width,
            result.image_height,
            result.error_message,
            now,
            now,
        ),
    )


def _save_crop(
    image_path: Path,
    detection: FaceDetection,
    *,
    face_id: str,
    crop_dir: Path,
    thumbnail_dir: Path,
) -> dict[str, str]:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        x0 = max(0, int(detection.bbox_x))
        y0 = max(0, int(detection.bbox_y))
        x1 = min(width, int(detection.bbox_x + detection.bbox_w))
        y1 = min(height, int(detection.bbox_y + detection.bbox_h))
        if x1 <= x0 or y1 <= y0:
            return {}
        crop = image.crop((x0, y0, x1, y1))
        crop_path = crop_dir / f"{face_id}.jpg"
        thumb_path = thumbnail_dir / f"{face_id}.jpg"
        crop.save(crop_path, format="JPEG", quality=90)
        thumb = crop.copy()
        thumb.thumbnail((160, 160))
        thumb.save(thumb_path, format="JPEG", quality=85)
    return {"crop_path": str(crop_path), "thumbnail_path": str(thumb_path)}


def _clip_detection_to_image(
    detection: FaceDetection,
    *,
    image_width: int | None,
    image_height: int | None,
) -> FaceDetection | None:
    if image_width is None or image_height is None:
        return detection
    x0 = max(0.0, float(detection.bbox_x))
    y0 = max(0.0, float(detection.bbox_y))
    x1 = min(float(image_width), float(detection.bbox_x) + float(detection.bbox_w))
    y1 = min(float(image_height), float(detection.bbox_y) + float(detection.bbox_h))
    if x1 <= x0 or y1 <= y0:
        return None
    return FaceDetection(
        bbox_x=x0,
        bbox_y=y0,
        bbox_w=x1 - x0,
        bbox_h=y1 - y0,
        detection_score=detection.detection_score,
        landmarks=detection.landmarks,
    )


def _has_existing_detection(repository: LifelogRepository, *, media_id: str, engine: str, final_only: bool) -> bool:
    statuses = ("success", "no_face_detected") if final_only else ("success", "no_face_detected", "failed", "engine_unavailable")
    placeholders = ",".join("?" for _ in statuses)
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        row = connection.execute(
            f"SELECT 1 FROM face_detections WHERE media_id = ? AND engine = ? AND status IN ({placeholders}) LIMIT 1",
            (media_id, engine, *statuses),
        ).fetchone()
        return row is not None


def _existing_detection_media_ids(repository: LifelogRepository, *, engine: str, final_only: bool) -> set[str]:
    statuses = ("success", "no_face_detected") if final_only else ("success", "no_face_detected", "failed", "engine_unavailable")
    placeholders = ",".join("?" for _ in statuses)
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        rows = connection.execute(
            f"SELECT DISTINCT media_id FROM face_detections WHERE engine = ? AND status IN ({placeholders})",
            (engine, *statuses),
        ).fetchall()
        return {str(row[0]) for row in rows}


def _delete_existing_detections(repository: LifelogRepository, *, media_id: str, engine: str) -> None:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        connection.execute("DELETE FROM face_detections WHERE media_id = ? AND engine = ?", (media_id, engine))
        connection.commit()


def _resolve_date_range(options: FaceDetectOptions) -> tuple[str | None, str | None]:
    if options.date:
        return options.date, options.date
    return options.start_date, options.end_date


def _face_date_where(date_from: str | None, date_to: str | None) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if date_from:
        clauses.append("SUBSTR(COALESCE(media_items.captured_at, media_items.fallback_captured_at), 1, 10) >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("SUBSTR(COALESCE(media_items.captured_at, media_items.fallback_captured_at), 1, 10) <= ?")
        params.append(date_to)
    if not clauses:
        return "", []
    return "JOIN media_items ON media_items.id = face_detections.media_id WHERE " + " AND ".join(clauses), params


def _join_date_filter(where: str) -> str:
    if not where:
        return ""
    return where.replace("JOIN media_items ON media_items.id = face_detections.media_id ", "")


def _face_id(media_id: str, engine: str, suffix: str) -> str:
    digest = hashlib.sha256(f"{media_id}|{engine}|{suffix}".encode("utf-8")).hexdigest()[:24]
    return f"face_{digest}"


def _count(connection, query: str, params: list[Any] | tuple[Any, ...] = ()) -> int:
    row = connection.execute(query, params).fetchone()
    return int(row[0]) if row else 0


def _rows(connection, query: str, params: list[Any] | tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(query, params).fetchall()]


def _count_lines(rows: list[dict[str, Any]], key: str) -> list[str]:
    if not rows:
        return ["  - none"]
    return [f"  - {row.get(key) or 'NULL'}: {row.get('count')}" for row in rows]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
