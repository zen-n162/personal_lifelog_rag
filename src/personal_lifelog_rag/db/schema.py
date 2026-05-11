"""SQLite schema for the local lifelog database."""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 21

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS media_items (
        id TEXT PRIMARY KEY,
        file_path TEXT NOT NULL,
        file_name TEXT,
        file_hash TEXT UNIQUE,
        media_type TEXT,
        captured_at TEXT,
        fallback_captured_at TEXT,
        gps_lat REAL,
        gps_lon REAL,
        camera_model TEXT,
        width INTEGER,
        height INTEGER,
        thumbnail_path TEXT,
        caption TEXT,
        ocr_text TEXT,
        analysis_json TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS line_messages (
        id TEXT PRIMARY KEY,
        chat_id TEXT,
        source_file TEXT,
        sent_at TEXT,
        sender TEXT,
        text TEXT,
        message_type TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        date TEXT,
        start_time TEXT,
        end_time TEXT,
        title TEXT,
        summary TEXT,
        location_name TEXT,
        gps_lat REAL,
        gps_lon REAL,
        participants_json TEXT,
        confidence REAL,
        source TEXT,
        generation_method TEXT,
        is_user_edited INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS text_embeddings (
        source_type TEXT NOT NULL,
        source_id TEXT NOT NULL,
        text TEXT NOT NULL,
        embedding_json TEXT NOT NULL,
        embedding_model TEXT NOT NULL,
        embedding_dim INTEGER NOT NULL,
        content_hash TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (source_type, source_id, embedding_model)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS event_overrides (
        event_id TEXT PRIMARY KEY,
        title_override TEXT,
        summary_override TEXT,
        location_name_override TEXT,
        tags_json TEXT,
        is_verified INTEGER NOT NULL DEFAULT 0,
        is_hidden INTEGER NOT NULL DEFAULT 0,
        is_pinned INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS event_evidence (
        event_id TEXT,
        evidence_type TEXT,
        evidence_id TEXT,
        weight REAL,
        PRIMARY KEY (event_id, evidence_type, evidence_id),
        FOREIGN KEY (event_id) REFERENCES events (id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS line_call_events (
        message_id TEXT PRIMARY KEY,
        chat_id TEXT,
        sent_at TEXT,
        sender TEXT,
        call_status TEXT,
        duration_sec INTEGER,
        raw_text_short TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (message_id) REFERENCES line_messages (id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS media_ocr (
        media_id TEXT PRIMARY KEY,
        ocr_text TEXT,
        ocr_text_redacted TEXT,
        ocr_engine TEXT,
        ocr_languages TEXT,
        confidence REAL,
        blocks_json TEXT,
        status TEXT,
        error_message TEXT,
        analyzed_at TEXT,
        analysis_version TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS media_vlm (
        media_id TEXT PRIMARY KEY,
        caption TEXT,
        short_caption TEXT,
        scene_tags_json TEXT,
        object_tags_json TEXT,
        activity_tags_json TEXT,
        location_cues_json TEXT,
        food_cues_json TEXT,
        text_cues_json TEXT,
        uncertainty_notes_json TEXT,
        evidence_strength TEXT,
        people_count INTEGER,
        contains_text_hint INTEGER,
        safety_flags_json TEXT,
        vlm_engine TEXT,
        model_name TEXT,
        prompt_version TEXT,
        confidence REAL,
        status TEXT,
        error_message TEXT,
        analyzed_at TEXT,
        analysis_version TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS media_vlm_overrides (
        media_id TEXT PRIMARY KEY,
        caption_override TEXT,
        short_caption_override TEXT,
        scene_tags_override_json TEXT,
        object_tags_override_json TEXT,
        activity_tags_override_json TEXT,
        food_cues_override_json TEXT,
        location_cues_override_json TEXT,
        is_verified INTEGER NOT NULL DEFAULT 0,
        is_hidden INTEGER NOT NULL DEFAULT 0,
        is_wrong INTEGER NOT NULL DEFAULT 0,
        is_searchable INTEGER NOT NULL DEFAULT 1,
        is_event_usable INTEGER NOT NULL DEFAULT 1,
        review_status TEXT NOT NULL DEFAULT 'unreviewed',
        review_note TEXT,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (media_id) REFERENCES media_items (id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS media_embeddings (
        media_id TEXT,
        embedding_type TEXT,
        embedding_model TEXT,
        embedding_dim INTEGER,
        embedding BLOB,
        embedding_format TEXT,
        source_text TEXT,
        status TEXT,
        error_message TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (media_id, embedding_type, embedding_model),
        FOREIGN KEY (media_id) REFERENCES media_items (id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS analysis_jobs (
        job_id TEXT PRIMARY KEY,
        job_type TEXT,
        status TEXT,
        target_scope_json TEXT,
        engine TEXT,
        model_name TEXT,
        prompt_version TEXT,
        analysis_version TEXT,
        total_items INTEGER,
        processed_items INTEGER,
        success_items INTEGER,
        failed_items INTEGER,
        skipped_items INTEGER,
        started_at TEXT,
        finished_at TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        error_message TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS analysis_job_items (
        job_id TEXT,
        item_id TEXT,
        item_type TEXT,
        status TEXT,
        error_message TEXT,
        started_at TEXT,
        finished_at TEXT,
        latency_sec REAL,
        PRIMARY KEY (job_id, item_id),
        FOREIGN KEY (job_id) REFERENCES analysis_jobs (job_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS location_points (
        id TEXT PRIMARY KEY,
        media_id TEXT,
        event_id TEXT,
        captured_at TEXT,
        source TEXT,
        lat REAL,
        lon REAL,
        altitude REAL,
        accuracy_m REAL,
        geohash TEXT,
        privacy_level TEXT NOT NULL DEFAULT 'exact_private',
        cluster_id TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (media_id) REFERENCES media_items (id) ON DELETE CASCADE,
        FOREIGN KEY (event_id) REFERENCES events (id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS place_clusters (
        id TEXT PRIMARY KEY,
        centroid_lat REAL,
        centroid_lon REAL,
        radius_m REAL,
        point_count INTEGER,
        photo_count INTEGER,
        event_count INTEGER,
        first_seen_at TEXT,
        last_seen_at TEXT,
        clustering_method TEXT,
        status TEXT NOT NULL DEFAULT 'unreviewed',
        privacy_level TEXT NOT NULL DEFAULT 'exact_private',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS places (
        id TEXT PRIMARY KEY,
        display_name TEXT,
        public_name TEXT,
        category TEXT,
        cluster_id TEXT,
        aliases_json TEXT,
        privacy_level TEXT NOT NULL DEFAULT 'private',
        manual_verified INTEGER NOT NULL DEFAULT 0,
        hidden INTEGER NOT NULL DEFAULT 0,
        searchable INTEGER NOT NULL DEFAULT 1,
        notes TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (cluster_id) REFERENCES place_clusters (id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS event_places (
        event_id TEXT,
        place_id TEXT,
        source TEXT,
        confidence REAL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (event_id, place_id, source),
        FOREIGN KEY (event_id) REFERENCES events (id) ON DELETE CASCADE,
        FOREIGN KEY (place_id) REFERENCES places (id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS media_places (
        media_id TEXT,
        place_id TEXT,
        source TEXT,
        confidence REAL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (media_id, place_id, source),
        FOREIGN KEY (media_id) REFERENCES media_items (id) ON DELETE CASCADE,
        FOREIGN KEY (place_id) REFERENCES places (id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS face_detections (
        id TEXT PRIMARY KEY,
        media_id TEXT NOT NULL,
        detected_at TEXT,
        engine TEXT,
        model_name TEXT,
        status TEXT,
        bbox_x REAL,
        bbox_y REAL,
        bbox_w REAL,
        bbox_h REAL,
        landmarks_json TEXT,
        detection_score REAL,
        image_width INTEGER,
        image_height INTEGER,
        crop_path TEXT,
        thumbnail_path TEXT,
        privacy_level TEXT NOT NULL DEFAULT 'private',
        review_status TEXT NOT NULL DEFAULT 'unreviewed',
        hidden INTEGER NOT NULL DEFAULT 0,
        error_message TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (media_id) REFERENCES media_items (id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS face_detection_runs (
        id TEXT PRIMARY KEY,
        started_at TEXT,
        finished_at TEXT,
        engine TEXT,
        model_name TEXT,
        date_from TEXT,
        date_to TEXT,
        selected_count INTEGER,
        processed_count INTEGER,
        success_count INTEGER,
        no_face_count INTEGER,
        failed_count INTEGER,
        engine_unavailable_count INTEGER,
        config_json TEXT,
        status TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS face_embeddings (
        face_id TEXT PRIMARY KEY,
        embedding_model TEXT,
        embedding_dim INTEGER,
        embedding_blob BLOB,
        embedding_format TEXT,
        normalized INTEGER NOT NULL DEFAULT 1,
        status TEXT,
        error_message TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (face_id) REFERENCES face_detections (id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS face_clusters (
        id TEXT PRIMARY KEY,
        cluster_label TEXT,
        representative_face_id TEXT,
        face_count INTEGER,
        first_seen_at TEXT,
        last_seen_at TEXT,
        clustering_method TEXT,
        distance_threshold REAL,
        status TEXT NOT NULL DEFAULT 'unreviewed',
        review_status TEXT NOT NULL DEFAULT 'unreviewed',
        confidence REAL,
        privacy_level TEXT NOT NULL DEFAULT 'private',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (representative_face_id) REFERENCES face_detections (id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS face_cluster_members (
        cluster_id TEXT,
        face_id TEXT,
        distance_to_centroid REAL,
        confidence REAL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (cluster_id, face_id),
        FOREIGN KEY (cluster_id) REFERENCES face_clusters (id) ON DELETE CASCADE,
        FOREIGN KEY (face_id) REFERENCES face_detections (id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS persons (
        id TEXT PRIMARY KEY,
        display_name TEXT NOT NULL,
        public_name TEXT,
        aliases_json TEXT,
        privacy_level TEXT NOT NULL DEFAULT 'private',
        notes TEXT,
        manual_verified INTEGER NOT NULL DEFAULT 1,
        deleted_at TEXT,
        hidden INTEGER NOT NULL DEFAULT 0,
        searchable INTEGER NOT NULL DEFAULT 1,
        event_usable INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS person_face_clusters (
        person_id TEXT,
        face_cluster_id TEXT,
        verified_by_user INTEGER NOT NULL DEFAULT 1,
        verified_at TEXT,
        source TEXT NOT NULL DEFAULT 'manual',
        confidence REAL NOT NULL DEFAULT 1.0,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (person_id, face_cluster_id),
        FOREIGN KEY (person_id) REFERENCES persons (id) ON DELETE CASCADE,
        FOREIGN KEY (face_cluster_id) REFERENCES face_clusters (id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS person_aliases (
        id TEXT PRIMARY KEY,
        person_id TEXT NOT NULL,
        alias TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'manual',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (person_id) REFERENCES persons (id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS line_speaker_links (
        id TEXT PRIMARY KEY,
        chat_id TEXT,
        speaker_name TEXT,
        person_id TEXT,
        source TEXT NOT NULL DEFAULT 'manual',
        confidence REAL NOT NULL DEFAULT 1.0,
        verified_by_user INTEGER NOT NULL DEFAULT 1,
        verified_at TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (chat_id, speaker_name, person_id),
        FOREIGN KEY (person_id) REFERENCES persons (id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS person_line_mentions (
        id TEXT PRIMARY KEY,
        person_id TEXT,
        message_id TEXT,
        mention_type TEXT,
        source TEXT,
        confidence REAL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (person_id) REFERENCES persons (id) ON DELETE CASCADE,
        FOREIGN KEY (message_id) REFERENCES line_messages (id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS media_people (
        media_id TEXT NOT NULL,
        person_id TEXT NOT NULL,
        source TEXT NOT NULL,
        confidence REAL,
        face_id TEXT,
        face_cluster_id TEXT,
        verified_by_user INTEGER NOT NULL DEFAULT 1,
        hidden INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (media_id, person_id, source, face_id),
        FOREIGN KEY (media_id) REFERENCES media_items (id) ON DELETE CASCADE,
        FOREIGN KEY (person_id) REFERENCES persons (id) ON DELETE CASCADE,
        FOREIGN KEY (face_id) REFERENCES face_detections (id) ON DELETE SET NULL,
        FOREIGN KEY (face_cluster_id) REFERENCES face_clusters (id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS event_people (
        event_id TEXT NOT NULL,
        person_id TEXT NOT NULL,
        source TEXT NOT NULL,
        confidence REAL,
        evidence_count INTEGER NOT NULL DEFAULT 0,
        media_count INTEGER NOT NULL DEFAULT 0,
        line_count INTEGER NOT NULL DEFAULT 0,
        verified_by_user INTEGER NOT NULL DEFAULT 0,
        hidden INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (event_id, person_id, source),
        FOREIGN KEY (event_id) REFERENCES events (id) ON DELETE CASCADE,
        FOREIGN KEY (person_id) REFERENCES persons (id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS person_event_notes (
        id TEXT PRIMARY KEY,
        person_id TEXT NOT NULL,
        event_id TEXT NOT NULL,
        note TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (person_id) REFERENCES persons (id) ON DELETE CASCADE,
        FOREIGN KEY (event_id) REFERENCES events (id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS privacy_actions (
        id TEXT PRIMARY KEY,
        action_type TEXT NOT NULL,
        target_type TEXT NOT NULL,
        target_id TEXT,
        mode TEXT NOT NULL,
        details_json TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_media_items_captured_at ON media_items (captured_at)",
    "CREATE INDEX IF NOT EXISTS idx_media_items_fallback_at ON media_items (fallback_captured_at)",
    "CREATE INDEX IF NOT EXISTS idx_media_items_file_hash ON media_items (file_hash)",
    "CREATE INDEX IF NOT EXISTS idx_media_items_gps ON media_items (gps_lat, gps_lon)",
    "CREATE INDEX IF NOT EXISTS idx_line_messages_sent_at ON line_messages (sent_at)",
    "CREATE INDEX IF NOT EXISTS idx_line_messages_chat_id ON line_messages (chat_id)",
    "CREATE INDEX IF NOT EXISTS idx_events_date ON events (date)",
    "CREATE INDEX IF NOT EXISTS idx_text_embeddings_source ON text_embeddings (source_type, source_id)",
    "CREATE INDEX IF NOT EXISTS idx_text_embeddings_model ON text_embeddings (embedding_model)",
    "CREATE INDEX IF NOT EXISTS idx_event_overrides_flags ON event_overrides (is_hidden, is_pinned, is_verified)",
    "CREATE INDEX IF NOT EXISTS idx_event_evidence_event_id ON event_evidence (event_id)",
    "CREATE INDEX IF NOT EXISTS idx_line_call_events_sent_at ON line_call_events (sent_at)",
    "CREATE INDEX IF NOT EXISTS idx_line_call_events_status ON line_call_events (call_status)",
    "CREATE INDEX IF NOT EXISTS idx_media_ocr_status ON media_ocr (status)",
    "CREATE INDEX IF NOT EXISTS idx_media_ocr_engine ON media_ocr (ocr_engine)",
    "CREATE INDEX IF NOT EXISTS idx_media_vlm_status ON media_vlm (status)",
    "CREATE INDEX IF NOT EXISTS idx_media_vlm_engine ON media_vlm (vlm_engine)",
    "CREATE INDEX IF NOT EXISTS idx_media_vlm_overrides_status ON media_vlm_overrides (review_status)",
    "CREATE INDEX IF NOT EXISTS idx_media_vlm_overrides_flags ON media_vlm_overrides (is_hidden, is_wrong, is_searchable, is_event_usable, is_verified)",
    "CREATE INDEX IF NOT EXISTS idx_media_embeddings_media ON media_embeddings (media_id)",
    "CREATE INDEX IF NOT EXISTS idx_media_embeddings_type_model ON media_embeddings (embedding_type, embedding_model)",
    "CREATE INDEX IF NOT EXISTS idx_media_embeddings_status ON media_embeddings (status)",
    "CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status ON analysis_jobs (status, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_analysis_jobs_type ON analysis_jobs (job_type, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_analysis_job_items_status ON analysis_job_items (job_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_location_points_media ON location_points (media_id)",
    "CREATE INDEX IF NOT EXISTS idx_location_points_event ON location_points (event_id)",
    "CREATE INDEX IF NOT EXISTS idx_location_points_captured ON location_points (captured_at)",
    "CREATE INDEX IF NOT EXISTS idx_location_points_cluster ON location_points (cluster_id)",
    "CREATE INDEX IF NOT EXISTS idx_place_clusters_status ON place_clusters (status)",
    "CREATE INDEX IF NOT EXISTS idx_places_cluster ON places (cluster_id)",
    "CREATE INDEX IF NOT EXISTS idx_places_privacy ON places (privacy_level)",
    "CREATE INDEX IF NOT EXISTS idx_event_places_event ON event_places (event_id)",
    "CREATE INDEX IF NOT EXISTS idx_event_places_place ON event_places (place_id)",
    "CREATE INDEX IF NOT EXISTS idx_media_places_media ON media_places (media_id)",
    "CREATE INDEX IF NOT EXISTS idx_media_places_place ON media_places (place_id)",
    "CREATE INDEX IF NOT EXISTS idx_face_detections_media ON face_detections (media_id)",
    "CREATE INDEX IF NOT EXISTS idx_face_detections_status ON face_detections (status)",
    "CREATE INDEX IF NOT EXISTS idx_face_detections_review ON face_detections (review_status)",
    "CREATE INDEX IF NOT EXISTS idx_face_detection_runs_status ON face_detection_runs (status, started_at)",
    "CREATE INDEX IF NOT EXISTS idx_face_embeddings_status ON face_embeddings (status)",
    "CREATE INDEX IF NOT EXISTS idx_face_clusters_status ON face_clusters (status)",
    "CREATE INDEX IF NOT EXISTS idx_face_clusters_review ON face_clusters (review_status)",
    "CREATE INDEX IF NOT EXISTS idx_face_cluster_members_face ON face_cluster_members (face_id)",
    "CREATE INDEX IF NOT EXISTS idx_persons_privacy ON persons (privacy_level)",
    "CREATE INDEX IF NOT EXISTS idx_persons_display_name ON persons (display_name)",
    "CREATE INDEX IF NOT EXISTS idx_person_face_clusters_cluster ON person_face_clusters (face_cluster_id)",
    "CREATE INDEX IF NOT EXISTS idx_person_aliases_person ON person_aliases (person_id)",
    "CREATE INDEX IF NOT EXISTS idx_person_aliases_alias ON person_aliases (alias)",
    "CREATE INDEX IF NOT EXISTS idx_line_speaker_links_person ON line_speaker_links (person_id)",
    "CREATE INDEX IF NOT EXISTS idx_line_speaker_links_speaker ON line_speaker_links (chat_id, speaker_name)",
    "CREATE INDEX IF NOT EXISTS idx_person_line_mentions_person ON person_line_mentions (person_id)",
    "CREATE INDEX IF NOT EXISTS idx_person_line_mentions_message ON person_line_mentions (message_id)",
    "CREATE INDEX IF NOT EXISTS idx_media_people_person ON media_people (person_id)",
    "CREATE INDEX IF NOT EXISTS idx_media_people_face ON media_people (face_id, face_cluster_id)",
    "CREATE INDEX IF NOT EXISTS idx_event_people_person ON event_people (person_id)",
    "CREATE INDEX IF NOT EXISTS idx_event_people_event ON event_people (event_id)",
    "CREATE INDEX IF NOT EXISTS idx_person_event_notes_person ON person_event_notes (person_id)",
    "CREATE INDEX IF NOT EXISTS idx_person_event_notes_event ON person_event_notes (event_id)",
    "CREATE INDEX IF NOT EXISTS idx_privacy_actions_target ON privacy_actions (target_type, target_id)",
    "CREATE INDEX IF NOT EXISTS idx_privacy_actions_created ON privacy_actions (created_at)",
    f"PRAGMA user_version = {SCHEMA_VERSION}",
]

TABLE_NAMES = (
    "media_items",
    "line_messages",
    "events",
    "event_evidence",
    "line_call_events",
    "media_ocr",
    "media_vlm",
    "media_vlm_overrides",
    "media_embeddings",
    "analysis_jobs",
    "analysis_job_items",
    "location_points",
    "place_clusters",
    "places",
    "event_places",
    "media_places",
    "face_detections",
    "face_detection_runs",
    "face_embeddings",
    "face_clusters",
    "face_cluster_members",
    "persons",
    "person_face_clusters",
    "person_aliases",
    "line_speaker_links",
    "person_line_mentions",
    "media_people",
    "event_people",
    "person_event_notes",
    "privacy_actions",
)

REQUIRED_COLUMNS = {
    "media_items": {
        "id",
        "file_path",
        "file_name",
        "file_hash",
        "media_type",
        "captured_at",
        "fallback_captured_at",
        "gps_lat",
        "gps_lon",
        "camera_model",
        "width",
        "height",
        "thumbnail_path",
        "created_at",
    },
    "line_messages": {
        "id",
        "chat_id",
        "source_file",
        "sent_at",
        "sender",
        "text",
        "message_type",
        "created_at",
    },
    "events": {
        "id",
        "date",
        "start_time",
        "end_time",
        "title",
        "summary",
        "location_name",
        "gps_lat",
        "gps_lon",
        "participants_json",
        "confidence",
        "created_at",
        "updated_at",
    },
    "event_evidence": {"event_id", "evidence_type", "evidence_id", "weight"},
    "line_call_events": {
        "message_id",
        "chat_id",
        "sent_at",
        "sender",
        "call_status",
        "duration_sec",
        "raw_text_short",
        "created_at",
    },
    "media_ocr": {
        "media_id",
        "ocr_text",
        "ocr_text_redacted",
        "ocr_engine",
        "ocr_languages",
        "confidence",
        "blocks_json",
        "status",
        "error_message",
        "analyzed_at",
        "analysis_version",
    },
    "media_vlm": {
        "media_id",
        "caption",
        "short_caption",
        "scene_tags_json",
        "object_tags_json",
        "activity_tags_json",
        "location_cues_json",
        "food_cues_json",
        "people_count",
        "contains_text_hint",
        "safety_flags_json",
        "vlm_engine",
        "model_name",
        "prompt_version",
        "confidence",
        "status",
        "error_message",
        "analyzed_at",
        "analysis_version",
    },
    "media_vlm_overrides": {
        "media_id",
        "caption_override",
        "short_caption_override",
        "scene_tags_override_json",
        "object_tags_override_json",
        "activity_tags_override_json",
        "food_cues_override_json",
        "location_cues_override_json",
        "is_verified",
        "is_hidden",
        "is_wrong",
        "is_searchable",
        "is_event_usable",
        "review_status",
        "review_note",
        "updated_at",
    },
    "media_embeddings": {
        "media_id",
        "embedding_type",
        "embedding_model",
        "embedding_dim",
        "embedding",
        "embedding_format",
        "source_text",
        "status",
        "error_message",
        "created_at",
        "updated_at",
    },
    "analysis_jobs": {
        "job_id",
        "job_type",
        "status",
        "target_scope_json",
        "engine",
        "model_name",
        "prompt_version",
        "analysis_version",
        "total_items",
        "processed_items",
        "success_items",
        "failed_items",
        "skipped_items",
        "started_at",
        "finished_at",
        "created_at",
        "error_message",
    },
    "analysis_job_items": {
        "job_id",
        "item_id",
        "item_type",
        "status",
        "error_message",
        "started_at",
        "finished_at",
        "latency_sec",
    },
    "location_points": {
        "id",
        "media_id",
        "event_id",
        "captured_at",
        "source",
        "lat",
        "lon",
        "altitude",
        "accuracy_m",
        "geohash",
        "privacy_level",
        "cluster_id",
        "created_at",
        "updated_at",
    },
    "place_clusters": {
        "id",
        "centroid_lat",
        "centroid_lon",
        "radius_m",
        "point_count",
        "photo_count",
        "event_count",
        "first_seen_at",
        "last_seen_at",
        "clustering_method",
        "status",
        "privacy_level",
        "created_at",
        "updated_at",
    },
    "places": {
        "id",
        "display_name",
        "public_name",
        "category",
        "cluster_id",
        "aliases_json",
        "privacy_level",
        "manual_verified",
        "notes",
        "created_at",
        "updated_at",
    },
    "event_places": {
        "event_id",
        "place_id",
        "source",
        "confidence",
        "created_at",
        "updated_at",
    },
    "media_places": {
        "media_id",
        "place_id",
        "source",
        "confidence",
        "created_at",
        "updated_at",
    },
    "face_detections": {
        "id",
        "media_id",
        "detected_at",
        "engine",
        "model_name",
        "status",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "landmarks_json",
        "detection_score",
        "image_width",
        "image_height",
        "crop_path",
        "thumbnail_path",
        "privacy_level",
        "review_status",
        "error_message",
        "created_at",
        "updated_at",
    },
    "face_detection_runs": {
        "id",
        "started_at",
        "finished_at",
        "engine",
        "model_name",
        "date_from",
        "date_to",
        "selected_count",
        "processed_count",
        "success_count",
        "no_face_count",
        "failed_count",
        "engine_unavailable_count",
        "config_json",
        "status",
    },
    "face_embeddings": {
        "face_id",
        "embedding_model",
        "embedding_dim",
        "embedding_blob",
        "embedding_format",
        "normalized",
        "status",
        "error_message",
        "created_at",
        "updated_at",
    },
    "face_clusters": {
        "id",
        "cluster_label",
        "representative_face_id",
        "face_count",
        "first_seen_at",
        "last_seen_at",
        "clustering_method",
        "distance_threshold",
        "status",
        "review_status",
        "confidence",
        "privacy_level",
        "created_at",
        "updated_at",
    },
    "face_cluster_members": {
        "cluster_id",
        "face_id",
        "distance_to_centroid",
        "confidence",
        "created_at",
    },
    "persons": {
        "id",
        "display_name",
        "public_name",
        "aliases_json",
        "privacy_level",
        "notes",
        "manual_verified",
        "created_at",
        "updated_at",
    },
    "person_face_clusters": {
        "person_id",
        "face_cluster_id",
        "verified_by_user",
        "verified_at",
        "source",
        "confidence",
        "created_at",
        "updated_at",
    },
    "person_aliases": {
        "id",
        "person_id",
        "alias",
        "source",
        "created_at",
        "updated_at",
    },
    "line_speaker_links": {
        "id",
        "chat_id",
        "speaker_name",
        "person_id",
        "source",
        "confidence",
        "verified_by_user",
        "verified_at",
        "created_at",
        "updated_at",
    },
    "person_line_mentions": {
        "id",
        "person_id",
        "message_id",
        "mention_type",
        "source",
        "confidence",
        "created_at",
    },
    "media_people": {
        "media_id",
        "person_id",
        "source",
        "confidence",
        "face_id",
        "face_cluster_id",
        "verified_by_user",
        "created_at",
        "updated_at",
    },
    "event_people": {
        "event_id",
        "person_id",
        "source",
        "confidence",
        "evidence_count",
        "media_count",
        "line_count",
        "verified_by_user",
        "created_at",
        "updated_at",
    },
    "person_event_notes": {
        "id",
        "person_id",
        "event_id",
        "note",
        "created_at",
        "updated_at",
    },
    "privacy_actions": {
        "id",
        "action_type",
        "target_type",
        "target_id",
        "mode",
        "details_json",
        "created_at",
    },
}


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create the current schema.

    Older development DBs used a different shape. If an incompatible DB is
    empty, it is recreated automatically. If it contains rows, initialization
    stops so private data is not silently dropped.
    """

    with connection:
        _drop_incompatible_empty_tables(connection)
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        _add_column_if_missing(connection, "media_items", "caption", "TEXT")
        _add_column_if_missing(connection, "media_items", "ocr_text", "TEXT")
        _add_column_if_missing(connection, "media_items", "analysis_json", "TEXT")
        _add_column_if_missing(connection, "events", "source", "TEXT")
        _add_column_if_missing(connection, "events", "generation_method", "TEXT")
        _add_column_if_missing(connection, "events", "is_user_edited", "INTEGER NOT NULL DEFAULT 0")
        _add_column_if_missing(connection, "media_vlm", "text_cues_json", "TEXT")
        _add_column_if_missing(connection, "media_vlm", "uncertainty_notes_json", "TEXT")
        _add_column_if_missing(connection, "media_vlm", "evidence_strength", "TEXT")
        _add_column_if_missing(connection, "media_vlm_overrides", "review_note", "TEXT")
        _add_column_if_missing(connection, "location_points", "cluster_id", "TEXT")
        _add_column_if_missing(connection, "persons", "deleted_at", "TEXT")
        _add_column_if_missing(connection, "persons", "hidden", "INTEGER NOT NULL DEFAULT 0")
        _add_column_if_missing(connection, "persons", "searchable", "INTEGER NOT NULL DEFAULT 1")
        _add_column_if_missing(connection, "persons", "event_usable", "INTEGER NOT NULL DEFAULT 1")
        _add_column_if_missing(connection, "places", "hidden", "INTEGER NOT NULL DEFAULT 0")
        _add_column_if_missing(connection, "places", "searchable", "INTEGER NOT NULL DEFAULT 1")
        _add_column_if_missing(connection, "media_people", "hidden", "INTEGER NOT NULL DEFAULT 0")
        _add_column_if_missing(connection, "event_people", "hidden", "INTEGER NOT NULL DEFAULT 0")
        _add_column_if_missing(connection, "face_detections", "hidden", "INTEGER NOT NULL DEFAULT 0")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_places_privacy_flags ON places (privacy_level, hidden, searchable)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_face_detections_hidden ON face_detections (hidden)")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_persons_privacy_flags ON persons "
            "(privacy_level, hidden, searchable, event_usable, deleted_at)"
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_media_people_hidden ON media_people (hidden)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_event_people_hidden ON event_people (hidden)")


def _drop_incompatible_empty_tables(connection: sqlite3.Connection) -> None:
    incompatible = [
        table_name
        for table_name in TABLE_NAMES
        if _table_exists(connection, table_name)
        and not REQUIRED_COLUMNS[table_name].issubset(_table_columns(connection, table_name))
    ]
    if not incompatible:
        return

    row_count = sum(_count_rows(connection, table_name) for table_name in TABLE_NAMES if _table_exists(connection, table_name))
    if row_count:
        names = ", ".join(incompatible)
        raise RuntimeError(
            "Existing SQLite DB has an older incompatible schema and contains data. "
            f"Refusing to modify tables: {names}"
        )

    for table_name in reversed(TABLE_NAMES):
        connection.execute(f"DROP TABLE IF EXISTS {table_name}")


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table_name})")}


def _add_column_if_missing(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_type: str,
) -> None:
    if _table_exists(connection, table_name) and column_name not in _table_columns(connection, table_name):
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def _count_rows(connection: sqlite3.Connection, table_name: str) -> int:
    row = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    return int(row[0])
