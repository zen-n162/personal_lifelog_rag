"""Face embedding and clustering services.

These services are private diagnostics/review tools. They do not connect face
clusters to names and do not expose clusters to normal search/QA/report flows.
"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime
import json
from pathlib import Path
import tempfile
from typing import Any
import uuid

from PIL import Image, ImageOps

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.db.schema import initialize_schema
from personal_lifelog_rag.embeddings.similarity import (
    FLOAT32_FORMAT,
    cosine_similarity,
    deserialize_embedding,
    normalize,
    serialize_embedding,
)
from personal_lifelog_rag.faces.embedding_engines import get_face_embedding_engine, opencv_sface_diagnostics
from personal_lifelog_rag.faces.schemas import (
    FaceClusterCandidate,
    FaceClusterReport,
    FaceClusteringConfig,
    FaceClusteringOptions,
    FaceDetectionConfig,
    FaceEmbeddingConfig,
    FaceEmbeddingDiagnostics,
    FaceEmbeddingOptions,
    FaceEmbeddingReport,
)


def load_face_runtime_config(path: str | Path | None = None) -> tuple[FaceEmbeddingConfig, FaceClusteringConfig]:
    if path is None:
        return FaceEmbeddingConfig(), FaceClusteringConfig()
    source = Path(path).expanduser()
    if not source.exists():
        return FaceEmbeddingConfig(), FaceClusteringConfig()
    payload = _parse_simple_mapping(source.read_text(encoding="utf-8"))
    face_embedding = payload.get("face_embedding") or (payload.get("models", {}) if isinstance(payload.get("models"), dict) else {}).get("face_embedding", {})
    face_clustering = payload.get("face_clustering", {})
    emb = face_embedding if isinstance(face_embedding, dict) else {}
    cluster = face_clustering if isinstance(face_clustering, dict) else {}
    return (
        FaceEmbeddingConfig(
            engine=str(emb.get("engine") or "opencv_sface"),
            model_path=_none_or_str(emb.get("model_path")),
            embedding_dim=_none_or_int(emb.get("embedding_dim")) or 128,
            local_only=_bool_or_default(emb.get("local_only"), True),
            normalize=_bool_or_default(emb.get("normalize"), True),
        ),
        FaceClusteringConfig(
            method=str(cluster.get("method") or "dbscan_cosine"),
            distance_threshold=_none_or_float(cluster.get("distance_threshold")) or 0.45,
            min_samples=_none_or_int(cluster.get("min_samples")) or 2,
        ),
    )


def load_face_detection_runtime_config(path: str | Path | None = None) -> FaceDetectionConfig:
    if path is None:
        return FaceDetectionConfig()
    source = Path(path).expanduser()
    if not source.exists():
        return FaceDetectionConfig()
    payload = _parse_simple_mapping(source.read_text(encoding="utf-8"))
    models = payload.get("models", {}) if isinstance(payload.get("models"), dict) else {}
    face_detection = payload.get("face_detection") or models.get("face_detection", {})
    det = face_detection if isinstance(face_detection, dict) else {}
    return FaceDetectionConfig(
        engine=str(det.get("engine") or "opencv_haar"),
        model_path=_none_or_str(det.get("model_path")),
        local_only=_bool_or_default(det.get("local_only"), True),
        score_threshold=_none_or_float(det.get("score_threshold")) or 0.85,
        nms_threshold=_none_or_float(det.get("nms_threshold")) or 0.3,
        top_k=_none_or_int(det.get("top_k")) or 5000,
        max_input_size=_none_or_int(det.get("max_input_size")) or 1280,
    )


def face_embedding_diagnostics(
    *,
    config_path: str | Path | None = None,
    engine_name: str | None = None,
) -> FaceEmbeddingDiagnostics:
    embedding_config, _ = load_face_runtime_config(config_path)
    engine = engine_name or embedding_config.engine
    diag = opencv_sface_diagnostics(model_path=embedding_config.model_path)
    selected = get_face_embedding_engine(
        engine,
        model_path=embedding_config.model_path,
        embedding_dim=embedding_config.embedding_dim,
        normalize_vector=embedding_config.normalize,
    )
    return FaceEmbeddingDiagnostics(
        selected_engine=selected.name,
        opencv_import_ok=bool(diag["opencv_import_ok"]),
        cv2_version=str(diag["cv2_version"]) if diag["cv2_version"] else None,
        model_path_configured=str(diag["model_path_configured"]) if diag["model_path_configured"] else None,
        model_file_exists=bool(diag["model_file_exists"]),
        local_only=embedding_config.local_only,
        embedding_dim=embedding_config.embedding_dim,
        engine_available=selected.is_available(),
        unavailable_reason=selected.unavailable_reason(),
    )


def run_face_embedding(
    repository: LifelogRepository,
    options: FaceEmbeddingOptions,
) -> FaceEmbeddingReport:
    repository.initialize()
    date_from, date_to = _resolve_date_range(options.date, options.start_date, options.end_date)
    engine = get_face_embedding_engine(
        options.engine,
        model_path=options.model_path,
        embedding_dim=options.embedding_dim,
        normalize_vector=options.normalize,
    )
    rows = _select_face_rows(repository, options=options, date_from=date_from, date_to=date_to)
    report = FaceEmbeddingReport(
        engine=engine.name,
        model_name=engine.model_name,
        date_from=date_from,
        date_to=date_to,
        selected_count=len(rows),
        existing_embedding_count=_count_face_embeddings_for_rows(repository, rows),
        dry_run=options.dry_run,
        replace=options.replace,
        batch_size=options.batch_size,
        detections_engine=options.detections_engine,
        target_status=options.status,
        only_with_crop=options.only_with_crop,
        only_existing_files=options.only_existing_files,
    )
    if options.dry_run:
        return report
    if options.replace and rows:
        report.deleted_embedding_count = _delete_face_embeddings_for_rows(
            repository,
            [str(row["id"]) for row in rows],
            batch_size=max(1, options.batch_size),
        )
    engine_available = engine.is_available()
    unavailable_reason = engine.unavailable_reason()
    for row in rows:
        face_id = str(row["id"])
        if options.skip_existing and _has_success_face_embedding(repository, face_id):
            report.skipped_count += 1
            continue
        if options.force:
            _delete_face_embedding(repository, face_id)
        report.processed_count += 1
        tempdir: tempfile.TemporaryDirectory[str] | None = None
        try:
            if not engine_available:
                _upsert_face_embedding_status(
                    repository,
                    face_id=face_id,
                    model_name=engine.model_name,
                    embedding_dim=options.embedding_dim,
                    status="engine_unavailable",
                    error_message=unavailable_reason,
                )
                report.engine_unavailable_count += 1
                continue
            prepared = _face_image_for_embedding(row)
            if prepared is None:
                _upsert_face_embedding_status(
                    repository,
                    face_id=face_id,
                    model_name=engine.model_name,
                    embedding_dim=options.embedding_dim,
                    status="skipped",
                    error_message="face crop and original media file are unavailable",
                )
                report.skipped_count += 1
                continue
            face_image_path, tempdir = prepared
            result = engine.embed_face(face_image_path, face_id=face_id)
            if result.status != "success":
                _upsert_face_embedding_status(
                    repository,
                    face_id=face_id,
                    model_name=engine.model_name,
                    embedding_dim=options.embedding_dim,
                    status=result.status,
                    error_message=result.error_message,
                )
                if result.status == "engine_unavailable":
                    report.engine_unavailable_count += 1
                else:
                    report.failed_count += 1
                    if result.error_message:
                        report.errors.append(f"{face_id}: {result.error_message[:240]}")
                continue
            vector = normalize(result.vector) if options.normalize else result.vector
            _upsert_face_embedding_success(
                repository,
                face_id=face_id,
                model_name=engine.model_name,
                vector=vector,
                normalized=options.normalize,
            )
            report.success_count += 1
        finally:
            if tempdir is not None:
                tempdir.cleanup()
    return report


def run_face_clustering(repository: LifelogRepository, options: FaceClusteringOptions) -> FaceClusterReport:
    repository.initialize()
    date_from, date_to = _resolve_date_range(options.date, options.start_date, options.end_date)
    rows = _load_success_face_embeddings(
        repository,
        date_from=date_from,
        date_to=date_to,
        embedding_model=options.embedding_model,
    )
    method_name = _cluster_method_name(options.method, options.scope)
    report = FaceClusterReport(
        method=options.method,
        distance_threshold=options.distance_threshold,
        min_samples=options.min_samples,
        date_from=date_from,
        date_to=date_to,
        selected_embeddings=len(rows),
        dry_run=options.dry_run,
        replace=options.replace,
        scope=options.scope,
        embedding_model=options.embedding_model,
    )
    if not rows:
        return report
    report.replace_count = _count_replaceable_clusters(repository, method_name=method_name) if options.replace else 0
    candidates, singleton_count = _cluster_embeddings_for_method(
        rows,
        method=options.method,
        threshold=options.distance_threshold,
        min_samples=options.min_samples,
    )
    report.candidates = candidates
    report.cluster_candidates = len(candidates)
    report.singleton_count = singleton_count
    report.largest_cluster_size = max((len(candidate.face_ids) for candidate in candidates), default=0)
    if options.dry_run:
        return report
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        if options.replace:
            cluster_ids = [
                str(row["id"])
                for row in connection.execute(
                    "SELECT id FROM face_clusters WHERE clustering_method = ?",
                    (method_name,),
                ).fetchall()
            ]
            if cluster_ids:
                placeholders = ",".join("?" for _ in cluster_ids)
                linked_count = _count(
                    connection,
                    f"SELECT COUNT(*) FROM person_face_clusters WHERE face_cluster_id IN ({placeholders})",
                    cluster_ids,
                )
                if linked_count:
                    raise ValueError(
                        "refusing to replace face clusters with manual person links; "
                        "detach person_face_clusters first"
                    )
            for cluster_id in cluster_ids:
                connection.execute("DELETE FROM face_cluster_members WHERE cluster_id = ?", (cluster_id,))
            connection.execute("DELETE FROM face_clusters WHERE clustering_method = ?", (method_name,))
        start_index = _count(connection, "SELECT COUNT(*) FROM face_clusters") + 1
        now = _now()
        for offset, candidate in enumerate(candidates, start=start_index):
            cluster_id = f"face_cluster_{uuid.uuid4().hex[:16]}"
            label = f"person_candidate_{offset:03d}"
            connection.execute(
                """
                INSERT INTO face_clusters (
                    id, cluster_label, representative_face_id, face_count, first_seen_at,
                    last_seen_at, clustering_method, distance_threshold, status,
                    review_status, confidence, privacy_level, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'unreviewed', 'unreviewed', ?, 'private', ?, ?)
                """,
                (
                    cluster_id,
                    label,
                    candidate.representative_face_id,
                    len(candidate.face_ids),
                    candidate.first_seen_at,
                    candidate.last_seen_at,
                    method_name,
                    options.distance_threshold,
                    candidate.confidence,
                    now,
                    now,
                ),
            )
            for face_id in candidate.face_ids:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO face_cluster_members (
                        cluster_id, face_id, distance_to_centroid, confidence, created_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        cluster_id,
                        face_id,
                        candidate.distances.get(face_id),
                        candidate.confidence,
                        now,
                    ),
                )
                report.members_written += 1
            report.clusters_written += 1
        connection.commit()
    return report


def face_cluster_stats(repository: LifelogRepository) -> dict[str, Any]:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        return {
            "face_embeddings_total": _count(connection, "SELECT COUNT(*) FROM face_embeddings"),
            "face_embedding_status_counts": _rows(
                connection,
                "SELECT status, COUNT(*) AS count FROM face_embeddings GROUP BY status ORDER BY count DESC, status ASC",
            ),
            "clusters_total": _count(connection, "SELECT COUNT(*) FROM face_clusters"),
            "unreviewed_clusters": _count(connection, "SELECT COUNT(*) FROM face_clusters WHERE review_status = 'unreviewed' OR status = 'unreviewed'"),
            "cluster_status_counts": _rows(
                connection,
                "SELECT status, COUNT(*) AS count FROM face_clusters GROUP BY status ORDER BY count DESC, status ASC",
            ),
            "cluster_size_distribution": _rows(
                connection,
                """
                SELECT face_count, COUNT(*) AS count
                FROM face_clusters
                GROUP BY face_count
                ORDER BY face_count ASC
                """,
            ),
            "largest_clusters": _rows(
                connection,
                """
                SELECT id, cluster_label, face_count, first_seen_at, last_seen_at, review_status
                FROM face_clusters
                ORDER BY face_count DESC, id ASC
                LIMIT 10
                """,
            ),
            "singleton_count": _count(connection, "SELECT COUNT(*) FROM face_clusters WHERE face_count = 1"),
            "date_range": _rows(
                connection,
                """
                SELECT MIN(COALESCE(media_items.captured_at, media_items.fallback_captured_at)) AS first_seen_at,
                       MAX(COALESCE(media_items.captured_at, media_items.fallback_captured_at)) AS last_seen_at
                FROM face_embeddings
                JOIN face_detections ON face_detections.id = face_embeddings.face_id
                JOIN media_items ON media_items.id = face_detections.media_id
                WHERE face_embeddings.status = 'success'
                """,
            )[0],
        }


def list_face_clusters(
    repository: LifelogRepository,
    *,
    cluster_id: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if cluster_id:
        clauses.append("face_clusters.id = ?")
        params.append(cluster_id)
    if status:
        clauses.append("face_clusters.status = ?")
        params.append(status)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(limit)
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        clusters = _rows(
            connection,
            f"""
            SELECT *
            FROM face_clusters
            {where}
            ORDER BY face_count DESC, first_seen_at ASC
            LIMIT ?
            """,
            params,
        )
        for cluster in clusters:
            cluster["members"] = _rows(
                connection,
                """
                SELECT face_cluster_members.*, face_detections.thumbnail_path, face_detections.crop_path,
                       media_items.captured_at, media_items.file_name
                FROM face_cluster_members
                JOIN face_detections ON face_detections.id = face_cluster_members.face_id
                JOIN media_items ON media_items.id = face_detections.media_id
                WHERE face_cluster_members.cluster_id = ?
                ORDER BY media_items.captured_at ASC, face_cluster_members.face_id ASC
                LIMIT ?
                """,
                [cluster["id"], limit],
            )
    return clusters


def update_face_cluster_status(repository: LifelogRepository, *, cluster_id: str, status: str) -> dict[str, Any]:
    if status not in {"unreviewed", "accepted", "rejected", "merged", "split", "bad_cluster"}:
        raise ValueError(f"unknown face cluster status: {status}")
    stored_status = "rejected" if status == "bad_cluster" else status
    review_status = "bad_cluster" if status == "bad_cluster" else ("unreviewed" if status == "unreviewed" else "reviewed")
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        connection.execute(
            "UPDATE face_clusters SET status = ?, review_status = ?, updated_at = ? WHERE id = ?",
            (stored_status, review_status, _now(), cluster_id),
        )
        connection.commit()
        row = connection.execute("SELECT * FROM face_clusters WHERE id = ?", (cluster_id,)).fetchone()
        if row is None:
            raise ValueError(f"face cluster not found: {cluster_id}")
        return dict(row)


def format_face_embedding_diagnostics(diag: FaceEmbeddingDiagnostics) -> str:
    return "\n".join(
        [
            "Face embedding diagnostics",
            f"- selected engine: {diag.selected_engine}",
            f"- opencv import: {'ok' if diag.opencv_import_ok else 'unavailable'}",
            f"- cv2 version: {diag.cv2_version or 'unknown'}",
            f"- model path configured: {Path(diag.model_path_configured).name if diag.model_path_configured else 'none'}",
            f"- model file exists: {diag.model_file_exists}",
            f"- local_only: {diag.local_only}",
            f"- embedding_dim: {diag.embedding_dim}",
            f"- engine_available: {diag.engine_available}",
            f"- unavailable_reason: {diag.unavailable_reason or ''}",
        ]
    )


def format_face_embedding_report(report: FaceEmbeddingReport) -> str:
    return "\n".join(
        [
            "Face embedding report",
            f"- engine: {report.engine}",
            f"- model: {report.model_name}",
            f"- date range: {report.date_from or '-'} .. {report.date_to or '-'}",
            f"- dry_run: {report.dry_run}",
            f"- replace: {report.replace}",
            f"- detections_engine: {report.detections_engine or '-'}",
            f"- target_status: {report.target_status}",
            f"- only_with_crop: {report.only_with_crop}",
            f"- only_existing_files: {report.only_existing_files}",
            f"- batch_size: {report.batch_size}",
            f"- selected: {report.selected_count}",
            f"- existing embeddings in selection: {report.existing_embedding_count}",
            f"- deleted embeddings: {report.deleted_embedding_count}",
            f"- processed: {report.processed_count}",
            f"- success: {report.success_count}",
            f"- failed: {report.failed_count}",
            f"- skipped: {report.skipped_count}",
            f"- engine_unavailable: {report.engine_unavailable_count}",
        ]
        + ([f"- first error: {report.errors[0]}"] if report.errors else [])
    )


def format_face_cluster_report(report: FaceClusterReport) -> str:
    lines = [
        "Face clustering report",
        f"- method: {report.method}",
        f"- distance_threshold: {report.distance_threshold}",
        f"- min_samples: {report.min_samples}",
        f"- date range: {report.date_from or '-'} .. {report.date_to or '-'}",
        f"- dry_run: {report.dry_run}",
        f"- replace: {report.replace}",
        f"- scope: {report.scope or '-'}",
        f"- embedding_model: {report.embedding_model or '-'}",
        f"- selected embeddings: {report.selected_embeddings}",
        f"- replace target clusters: {report.replace_count}",
        f"- cluster candidates: {report.cluster_candidates}",
        f"- clusters written: {report.clusters_written}",
        f"- members written: {report.members_written}",
        f"- singleton candidates skipped: {report.singleton_count}",
        f"- largest cluster size: {report.largest_cluster_size}",
    ]
    for candidate in report.candidates[:10]:
        lines.append(
            f"  - {candidate.cluster_label}: faces={len(candidate.face_ids)} "
            f"rep={candidate.representative_face_id} first={candidate.first_seen_at or ''}"
        )
    return "\n".join(lines)


def format_face_cluster_stats(stats: dict[str, Any]) -> str:
    lines = [
        "Face cluster stats",
        f"- face embeddings total: {stats['face_embeddings_total']}",
        "- embedding status counts:",
    ]
    lines.extend(_count_lines(stats["face_embedding_status_counts"], "status"))
    lines.extend(
        [
            f"- clusters total: {stats['clusters_total']}",
            f"- unreviewed clusters: {stats['unreviewed_clusters']}",
            "- cluster status counts:",
        ]
    )
    lines.extend(_count_lines(stats["cluster_status_counts"], "status"))
    lines.append("- cluster size distribution:")
    lines.extend(_count_lines(stats["cluster_size_distribution"], "face_count"))
    lines.append("- largest clusters:")
    if stats["largest_clusters"]:
        for row in stats["largest_clusters"]:
            lines.append(f"  - {row['id']} {row.get('cluster_label')}: {row.get('face_count')} faces")
    else:
        lines.append("  - none")
    return "\n".join(lines)


def format_face_embedding_stats(stats: dict[str, Any]) -> str:
    lines = [
        "Face embedding stats",
        f"- date range: {stats.get('date_from') or '-'} .. {stats.get('date_to') or '-'}",
        f"- face detections success: {stats.get('face_detections_success_count', 0)}",
        f"- embedding success: {stats.get('embedding_success_count', 0)}",
        f"- missing embeddings: {stats.get('missing_embeddings_count', 0)}",
        f"- crop missing rows: {stats.get('crop_missing_count', 0)}",
        "- embedding engine counts:",
    ]
    if stats.get("engine_counts"):
        for row in stats["engine_counts"]:
            lines.append(
                f"  - {row.get('embedding_model') or 'NULL'} / {row.get('status') or 'NULL'}: {row.get('count')}"
            )
    else:
        lines.append("  - none")
    lines.append("- detection engine counts:")
    if stats.get("detection_engine_counts"):
        for row in stats["detection_engine_counts"]:
            lines.append(
                f"  - {row.get('engine') or 'NULL'} / {row.get('model_name') or 'NULL'} / "
                f"{row.get('status') or 'NULL'}: {row.get('count')}"
            )
    else:
        lines.append("  - none")
    return "\n".join(lines)


def format_face_clusters(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "Face clusters\n- none"
    lines = ["Face clusters"]
    for row in rows:
        lines.append(
            f"- {row.get('id')} {row.get('cluster_label')} faces={row.get('face_count')} "
            f"status={row.get('status')} review={row.get('review_status')} "
            f"first={row.get('first_seen_at') or ''} last={row.get('last_seen_at') or ''} "
            f"rep={row.get('representative_face_id') or ''}"
        )
        for member in row.get("members", [])[:20]:
            lines.append(
                f"  - {member.get('face_id')} distance={member.get('distance_to_centroid')} "
                f"thumb={member.get('thumbnail_path') or ''}"
            )
    return "\n".join(lines)


def _select_face_rows(
    repository: LifelogRepository,
    *,
    options: FaceEmbeddingOptions,
    date_from: str | None,
    date_to: str | None,
) -> list[dict[str, Any]]:
    clauses = ["face_detections.status = ?"]
    params: list[Any] = [options.status]
    if date_from:
        clauses.append("SUBSTR(COALESCE(media_items.captured_at, media_items.fallback_captured_at), 1, 10) >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("SUBSTR(COALESCE(media_items.captured_at, media_items.fallback_captured_at), 1, 10) <= ?")
        params.append(date_to)
    if options.detections_engine:
        clauses.append("(LOWER(face_detections.engine) = ? OR LOWER(face_detections.model_name) LIKE ?)")
        engine_text = options.detections_engine.lower()
        params.extend([engine_text, f"%{engine_text}%"])
    if options.only_with_crop:
        clauses.append("face_detections.crop_path IS NOT NULL AND face_detections.crop_path != ''")
    if options.only_reviewed_detections:
        clauses.append("face_detections.review_status = 'accepted'")
    elif not options.include_unreviewed_detections:
        clauses.append("face_detections.review_status != 'unreviewed'")
    if options.min_detection_score is not None:
        clauses.append("face_detections.detection_score IS NOT NULL AND face_detections.detection_score >= ?")
        params.append(options.min_detection_score)
    limit_clause = ""
    if options.limit and options.limit > 0:
        limit_clause = "LIMIT ?"
        params.append(options.limit)
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        rows = _rows(
            connection,
            f"""
            SELECT face_detections.*, media_items.file_path, media_items.file_name,
                   media_items.captured_at, media_items.fallback_captured_at
            FROM face_detections
            JOIN media_items ON media_items.id = face_detections.media_id
            WHERE {' AND '.join(clauses)}
            ORDER BY COALESCE(media_items.captured_at, media_items.fallback_captured_at) ASC,
                     face_detections.id ASC
            {limit_clause}
            """,
            params,
        )
    if options.only_existing_files:
        rows = [
            row
            for row in rows
            if _path_exists(row.get("crop_path")) or _path_exists(row.get("file_path"))
        ]
    return rows


def _face_image_for_embedding(row: dict[str, Any]) -> tuple[Path, tempfile.TemporaryDirectory[str] | None] | None:
    crop_path = row.get("crop_path")
    if crop_path and Path(str(crop_path)).expanduser().exists():
        return Path(str(crop_path)).expanduser(), None
    file_path = row.get("file_path")
    if not file_path or not Path(str(file_path)).expanduser().exists():
        return None
    if row.get("bbox_x") is None:
        return None
    tempdir = tempfile.TemporaryDirectory()
    output_path = Path(tempdir.name) / f"{row.get('id')}.jpg"
    with Image.open(Path(str(file_path)).expanduser()) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        x0 = max(0, int(float(row.get("bbox_x") or 0)))
        y0 = max(0, int(float(row.get("bbox_y") or 0)))
        x1 = min(width, int(float(row.get("bbox_x") or 0) + float(row.get("bbox_w") or 0)))
        y1 = min(height, int(float(row.get("bbox_y") or 0) + float(row.get("bbox_h") or 0)))
        if x1 <= x0 or y1 <= y0:
            tempdir.cleanup()
            return None
        image.crop((x0, y0, x1, y1)).save(output_path, format="JPEG", quality=90)
    return output_path, tempdir


def _upsert_face_embedding_success(
    repository: LifelogRepository,
    *,
    face_id: str,
    model_name: str,
    vector: list[float],
    normalized: bool,
) -> None:
    now = _now()
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        connection.execute(
            """
            INSERT OR REPLACE INTO face_embeddings (
                face_id, embedding_model, embedding_dim, embedding_blob, embedding_format,
                normalized, status, error_message, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 'success', NULL, ?, ?)
            """,
            (
                face_id,
                model_name,
                len(vector),
                serialize_embedding(vector, embedding_format=FLOAT32_FORMAT),
                FLOAT32_FORMAT,
                1 if normalized else 0,
                now,
                now,
            ),
        )
        connection.commit()


def _upsert_face_embedding_status(
    repository: LifelogRepository,
    *,
    face_id: str,
    model_name: str,
    embedding_dim: int | None,
    status: str,
    error_message: str | None,
) -> None:
    now = _now()
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        connection.execute(
            """
            INSERT OR REPLACE INTO face_embeddings (
                face_id, embedding_model, embedding_dim, embedding_blob, embedding_format,
                normalized, status, error_message, created_at, updated_at
            )
            VALUES (?, ?, ?, NULL, ?, 1, ?, ?, ?, ?)
            """,
            (face_id, model_name, embedding_dim, FLOAT32_FORMAT, status, error_message, now, now),
        )
        connection.commit()


def _has_success_face_embedding(repository: LifelogRepository, face_id: str) -> bool:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        row = connection.execute(
            "SELECT 1 FROM face_embeddings WHERE face_id = ? AND status = 'success'",
            (face_id,),
        ).fetchone()
        return row is not None


def _delete_face_embedding(repository: LifelogRepository, face_id: str) -> None:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        connection.execute("DELETE FROM face_embeddings WHERE face_id = ?", (face_id,))
        connection.commit()


def _count_face_embeddings_for_rows(repository: LifelogRepository, rows: list[dict[str, Any]]) -> int:
    face_ids = [str(row["id"]) for row in rows]
    if not face_ids:
        return 0
    total = 0
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        for batch in _chunks(face_ids, 900):
            placeholders = ",".join("?" for _ in batch)
            total += _count(
                connection,
                f"SELECT COUNT(*) FROM face_embeddings WHERE face_id IN ({placeholders})",
                batch,
            )
    return total


def _delete_face_embeddings_for_rows(repository: LifelogRepository, face_ids: list[str], *, batch_size: int) -> int:
    if not face_ids:
        return 0
    deleted = 0
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        for batch in _chunks(face_ids, min(max(batch_size, 1), 900)):
            placeholders = ",".join("?" for _ in batch)
            before = _count(
                connection,
                f"SELECT COUNT(*) FROM face_embeddings WHERE face_id IN ({placeholders})",
                batch,
            )
            connection.execute(f"DELETE FROM face_embeddings WHERE face_id IN ({placeholders})", batch)
            connection.commit()
            deleted += before
    return deleted


def _load_success_face_embeddings(
    repository: LifelogRepository,
    *,
    date_from: str | None,
    date_to: str | None,
    embedding_model: str | None = None,
) -> list[dict[str, Any]]:
    clauses = ["face_embeddings.status = 'success'"]
    params: list[Any] = []
    if date_from:
        clauses.append("SUBSTR(COALESCE(media_items.captured_at, media_items.fallback_captured_at), 1, 10) >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("SUBSTR(COALESCE(media_items.captured_at, media_items.fallback_captured_at), 1, 10) <= ?")
        params.append(date_to)
    if embedding_model:
        clauses.append("face_embeddings.embedding_model = ?")
        params.append(embedding_model)
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        rows = _rows(
            connection,
            f"""
            SELECT face_embeddings.*, face_detections.thumbnail_path, face_detections.crop_path,
                   media_items.captured_at, media_items.fallback_captured_at, media_items.file_name
            FROM face_embeddings
            JOIN face_detections ON face_detections.id = face_embeddings.face_id
            JOIN media_items ON media_items.id = face_detections.media_id
            WHERE {' AND '.join(clauses)}
            ORDER BY COALESCE(media_items.captured_at, media_items.fallback_captured_at) ASC,
                     face_embeddings.face_id ASC
            """,
            params,
        )
    for row in rows:
        row["vector"] = deserialize_embedding(
            row.get("embedding_blob"),
            embedding_format=row.get("embedding_format"),
            dim=row.get("embedding_dim"),
        )
    return [row for row in rows if row.get("vector")]


def face_embedding_stats(
    repository: LifelogRepository,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    clauses: list[str] = []
    params: list[Any] = []
    if date_from:
        clauses.append("SUBSTR(COALESCE(media_items.captured_at, media_items.fallback_captured_at), 1, 10) >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("SUBSTR(COALESCE(media_items.captured_at, media_items.fallback_captured_at), 1, 10) <= ?")
        params.append(date_to)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        return {
            "date_from": date_from,
            "date_to": date_to,
            "face_detections_success_count": _count(
                connection,
                f"""
                SELECT COUNT(*)
                FROM face_detections
                JOIN media_items ON media_items.id = face_detections.media_id
                {where} {'AND' if where else 'WHERE'} face_detections.status = 'success'
                """,
                params,
            ),
            "embedding_success_count": _count(
                connection,
                f"""
                SELECT COUNT(*)
                FROM face_embeddings
                JOIN face_detections ON face_detections.id = face_embeddings.face_id
                JOIN media_items ON media_items.id = face_detections.media_id
                {where} {'AND' if where else 'WHERE'} face_embeddings.status = 'success'
                """,
                params,
            ),
            "missing_embeddings_count": _count(
                connection,
                f"""
                SELECT COUNT(*)
                FROM face_detections
                JOIN media_items ON media_items.id = face_detections.media_id
                LEFT JOIN face_embeddings ON face_embeddings.face_id = face_detections.id
                {where} {'AND' if where else 'WHERE'} face_detections.status = 'success'
                  AND (face_embeddings.face_id IS NULL OR face_embeddings.status != 'success')
                """,
                params,
            ),
            "engine_counts": _rows(
                connection,
                f"""
                SELECT face_embeddings.embedding_model, face_embeddings.status, COUNT(*) AS count
                FROM face_embeddings
                JOIN face_detections ON face_detections.id = face_embeddings.face_id
                JOIN media_items ON media_items.id = face_detections.media_id
                {where}
                GROUP BY face_embeddings.embedding_model, face_embeddings.status
                ORDER BY count DESC
                """,
                params,
            ),
            "detection_engine_counts": _rows(
                connection,
                f"""
                SELECT face_detections.engine, face_detections.model_name, face_detections.status, COUNT(*) AS count
                FROM face_detections
                JOIN media_items ON media_items.id = face_detections.media_id
                {where}
                GROUP BY face_detections.engine, face_detections.model_name, face_detections.status
                ORDER BY count DESC
                """,
                params,
            ),
            "crop_missing_count": _count(
                connection,
                f"""
                SELECT COUNT(*)
                FROM face_detections
                JOIN media_items ON media_items.id = face_detections.media_id
                {where} {'AND' if where else 'WHERE'} face_detections.status = 'success'
                  AND (face_detections.crop_path IS NULL OR face_detections.crop_path = '')
                """,
                params,
            ),
        }


def _cluster_embeddings(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    min_samples: int,
) -> tuple[list[FaceClusterCandidate], int]:
    parent = {str(row["face_id"]): str(row["face_id"]) for row in rows}
    vectors = {str(row["face_id"]): list(row["vector"]) for row in rows}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    face_ids = list(vectors)
    for left_index, left_id in enumerate(face_ids):
        for right_id in face_ids[left_index + 1 :]:
            distance = 1.0 - cosine_similarity(vectors[left_id], vectors[right_id])
            if distance <= threshold:
                union(left_id, right_id)

    groups: dict[str, list[str]] = {}
    for face_id in face_ids:
        groups.setdefault(find(face_id), []).append(face_id)

    row_by_face = {str(row["face_id"]): row for row in rows}
    candidates: list[FaceClusterCandidate] = []
    singleton_count = 0
    for index, members in enumerate(groups.values(), start=1):
        if len(members) < min_samples:
            singleton_count += len(members)
            continue
        centroid = _centroid([vectors[face_id] for face_id in members])
        distances = {face_id: 1.0 - cosine_similarity(vectors[face_id], centroid) for face_id in members}
        representative = min(members, key=lambda face_id: distances[face_id])
        dates = [
            row_by_face[face_id].get("captured_at") or row_by_face[face_id].get("fallback_captured_at")
            for face_id in members
            if row_by_face[face_id].get("captured_at") or row_by_face[face_id].get("fallback_captured_at")
        ]
        confidence = max(0.0, min(1.0, 1.0 - (sum(distances.values()) / max(len(distances), 1))))
        candidates.append(
            FaceClusterCandidate(
                cluster_label=f"person_candidate_{index:03d}",
                face_ids=sorted(members),
                representative_face_id=representative,
                first_seen_at=min(dates) if dates else None,
                last_seen_at=max(dates) if dates else None,
                confidence=round(confidence, 4),
                distances={face_id: round(distance, 6) for face_id, distance in distances.items()},
            )
        )
    return candidates, singleton_count


def _cluster_embeddings_for_method(
    rows: list[dict[str, Any]],
    *,
    method: str,
    threshold: float,
    min_samples: int,
) -> tuple[list[FaceClusterCandidate], int]:
    if method == "dbscan_cosine":
        return _cluster_embeddings_block_cosine(rows, threshold=threshold, min_samples=min_samples)
    return _cluster_embeddings(rows, threshold=threshold, min_samples=min_samples)


def _cluster_embeddings_block_cosine(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    min_samples: int,
) -> tuple[list[FaceClusterCandidate], int]:
    import numpy as np
    from scipy.spatial import cKDTree

    face_ids = [str(row["face_id"]) for row in rows]
    vectors = np.asarray([row["vector"] for row in rows], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    parent = list(range(len(face_ids)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    total = len(face_ids)
    radius = float((2.0 * threshold) ** 0.5)
    tree = cKDTree(vectors)
    for left, right in tree.query_pairs(radius):
        union(int(left), int(right))

    row_by_face = {str(row["face_id"]): row for row in rows}
    vector_by_face = {face_id: list(vectors[index]) for index, face_id in enumerate(face_ids)}
    groups_by_index: dict[int, list[int]] = {}
    for index in range(total):
        groups_by_index.setdefault(find(index), []).append(index)

    candidates: list[FaceClusterCandidate] = []
    singleton_count = 0
    for index, member_indexes in enumerate(groups_by_index.values(), start=1):
        if len(member_indexes) < min_samples:
            singleton_count += len(member_indexes)
            continue
        members = [face_ids[member_index] for member_index in member_indexes]
        centroid = _centroid([vector_by_face[face_id] for face_id in members])
        distances = {face_id: 1.0 - cosine_similarity(vector_by_face[face_id], centroid) for face_id in members}
        representative = min(members, key=lambda face_id: distances[face_id])
        dates = [
            row_by_face[face_id].get("captured_at") or row_by_face[face_id].get("fallback_captured_at")
            for face_id in members
            if row_by_face[face_id].get("captured_at") or row_by_face[face_id].get("fallback_captured_at")
        ]
        confidence = max(0.0, min(1.0, 1.0 - (sum(distances.values()) / max(len(distances), 1))))
        candidates.append(
            FaceClusterCandidate(
                cluster_label=f"person_candidate_{index:03d}",
                face_ids=sorted(members),
                representative_face_id=representative,
                first_seen_at=min(dates) if dates else None,
                last_seen_at=max(dates) if dates else None,
                confidence=round(confidence, 4),
                distances={face_id: round(distance, 6) for face_id, distance in distances.items()},
            )
        )
    return candidates, singleton_count


def _centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    length = len(vectors[0])
    values = [0.0] * length
    for vector in vectors:
        for index, value in enumerate(vector[:length]):
            values[index] += value
    return normalize([value / len(vectors) for value in values])


def _resolve_date_range(date: str | None, start: str | None, end: str | None) -> tuple[str | None, str | None]:
    if date:
        return date, date
    return start, end


def _parse_simple_mapping(raw_text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in raw_text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, raw_value = line.strip().split(":", 1)
        value_text = raw_value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if value_text == "":
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_scalar(value_text)
    return root


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _none_or_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _none_or_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _none_or_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_default(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _count(connection, query: str, params: list[Any] | tuple[Any, ...] = ()) -> int:
    row = connection.execute(query, params).fetchone()
    return int(row[0]) if row else 0


def _rows(connection, query: str, params: list[Any] | tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(query, params).fetchall()]


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _path_exists(value: Any) -> bool:
    if not value:
        return False
    return Path(str(value)).expanduser().exists()


def _cluster_method_name(method: str, scope: str | None) -> str:
    return f"{method}:{scope}" if scope else method


def _count_replaceable_clusters(repository: LifelogRepository, *, method_name: str) -> int:
    with closing(connect(repository.db_path)) as connection:
        initialize_schema(connection)
        return _count(connection, "SELECT COUNT(*) FROM face_clusters WHERE clustering_method = ?", (method_name,))


def _count_lines(rows: list[dict[str, Any]], key: str) -> list[str]:
    if not rows:
        return ["  - none"]
    return [f"  - {row.get(key) or 'NULL'}: {row.get('count')}" for row in rows]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
