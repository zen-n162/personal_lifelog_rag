"""Repository for local media embeddings."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Any

from personal_lifelog_rag.db.repository import connect
from personal_lifelog_rag.db.schema import initialize_schema
from personal_lifelog_rag.embeddings.similarity import (
    FLOAT32_FORMAT,
    deserialize_embedding,
    serialize_embedding,
    vector_dim_from_blob,
)


class MediaEmbeddingRepository:
    """Persistence boundary for `media_embeddings` rows."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()

    def upsert_embedding(
        self,
        *,
        media_id: str,
        embedding_type: str,
        embedding_model: str,
        vector: list[float] | None = None,
        embedding_dim: int | None = None,
        embedding_format: str = FLOAT32_FORMAT,
        source_text: str | None = None,
        status: str = "success",
        error_message: str | None = None,
    ) -> None:
        blob = serialize_embedding(vector or [], embedding_format=embedding_format) if vector else None
        dim = embedding_dim if embedding_dim is not None else (len(vector or []) or None)
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            connection.execute(
                """
                INSERT OR REPLACE INTO media_embeddings (
                    media_id,
                    embedding_type,
                    embedding_model,
                    embedding_dim,
                    embedding,
                    embedding_format,
                    source_text,
                    status,
                    error_message,
                    created_at,
                    updated_at
                )
                VALUES (
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    COALESCE(
                        (SELECT created_at FROM media_embeddings
                         WHERE media_id = ? AND embedding_type = ? AND embedding_model = ?),
                        CURRENT_TIMESTAMP
                    ),
                    CURRENT_TIMESTAMP
                )
                """,
                (
                    media_id,
                    embedding_type,
                    embedding_model,
                    dim,
                    blob,
                    embedding_format,
                    source_text,
                    status,
                    error_message,
                    media_id,
                    embedding_type,
                    embedding_model,
                ),
            )
            connection.commit()

    def get_embedding(self, media_id: str, embedding_type: str, embedding_model: str) -> dict[str, Any] | None:
        rows = self.list_embeddings(
            media_id=media_id,
            embedding_type=embedding_type,
            embedding_model=embedding_model,
            limit=1,
        )
        return rows[0] if rows else None

    def list_embeddings(
        self,
        *,
        media_id: str | None = None,
        embedding_type: str | None = None,
        embedding_model: str | None = None,
        statuses: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 100_000,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        timestamp = "COALESCE(media_items.captured_at, media_items.fallback_captured_at)"
        if media_id is not None:
            clauses.append("media_embeddings.media_id = ?")
            params.append(media_id)
        if embedding_type is not None:
            clauses.append("media_embeddings.embedding_type = ?")
            params.append(embedding_type)
        if embedding_model is not None:
            clauses.append("media_embeddings.embedding_model = ?")
            params.append(embedding_model)
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"media_embeddings.status IN ({placeholders})")
            params.extend(statuses)
        if start_date is not None:
            clauses.append(f"substr({timestamp}, 1, 10) >= ?")
            params.append(start_date)
        if end_date is not None:
            clauses.append(f"substr({timestamp}, 1, 10) <= ?")
            params.append(end_date)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with closing(connect(self.db_path)) as connection:
            initialize_schema(connection)
            rows = connection.execute(
                f"""
                SELECT
                    media_embeddings.*,
                    media_items.file_name,
                    media_items.file_path,
                    media_items.captured_at,
                    media_items.fallback_captured_at,
                    media_items.thumbnail_path,
                    media_items.gps_lat,
                    media_items.gps_lon,
                    media_vlm.caption,
                    media_vlm.short_caption,
                    media_vlm.scene_tags_json,
                    media_vlm.object_tags_json,
                    media_vlm.activity_tags_json,
                    media_vlm.location_cues_json,
                    media_vlm.food_cues_json,
                    media_vlm.safety_flags_json,
                    media_vlm.evidence_strength AS vlm_evidence_strength,
                    media_ocr.ocr_text,
                    media_ocr.ocr_text_redacted
                FROM media_embeddings
                LEFT JOIN media_items ON media_items.id = media_embeddings.media_id
                LEFT JOIN media_vlm ON media_vlm.media_id = media_embeddings.media_id
                LEFT JOIN media_ocr ON media_ocr.media_id = media_embeddings.media_id
                {where_sql}
                ORDER BY {timestamp} ASC, media_embeddings.media_id ASC, media_embeddings.embedding_type ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [dict(row) for row in rows]

    def existing_success(self, media_id: str, embedding_type: str, embedding_model: str) -> bool:
        row = self.get_embedding(media_id, embedding_type, embedding_model)
        return bool(row and row.get("status") == "success")

    def stats(self, *, start_date: str | None = None, end_date: str | None = None) -> dict[str, Any]:
        rows = self.list_embeddings(start_date=start_date, end_date=end_date, limit=1_000_000)
        by_type: dict[str, int] = {}
        by_model: dict[str, int] = {}
        by_status: dict[str, int] = {}
        by_dim: dict[str, int] = {}
        for row in rows:
            by_type[str(row.get("embedding_type") or "unknown")] = by_type.get(str(row.get("embedding_type") or "unknown"), 0) + 1
            by_model[str(row.get("embedding_model") or "unknown")] = by_model.get(str(row.get("embedding_model") or "unknown"), 0) + 1
            by_status[str(row.get("status") or "unknown")] = by_status.get(str(row.get("status") or "unknown"), 0) + 1
            by_dim[str(row.get("embedding_dim") or "unknown")] = by_dim.get(str(row.get("embedding_dim") or "unknown"), 0) + 1
        return {
            "range": {"from": start_date, "to": end_date},
            "total": len(rows),
            "by_type": dict(sorted(by_type.items())),
            "by_model": dict(sorted(by_model.items())),
            "by_status": dict(sorted(by_status.items())),
            "embedding_dim_distribution": dict(sorted(by_dim.items())),
        }


def embedding_vector(row: dict[str, Any]) -> list[float]:
    return deserialize_embedding(
        row.get("embedding"),
        embedding_format=row.get("embedding_format") or FLOAT32_FORMAT,
        dim=int(row["embedding_dim"]) if row.get("embedding_dim") is not None else None,
    )


def actual_embedding_dim(row: dict[str, Any]) -> int | None:
    return vector_dim_from_blob(row.get("embedding"), embedding_format=row.get("embedding_format"))

