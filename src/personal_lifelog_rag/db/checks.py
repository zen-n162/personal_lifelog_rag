"""SQLite integrity checks for the local lifelog database."""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
from typing import Any

from personal_lifelog_rag.db.repository import connect
from personal_lifelog_rag.db.schema import initialize_schema


SAMPLE_LIMIT = 10
CHAT_ID_LIMIT = 20


def run_db_check(db_path: str | Path, *, fail_on_missing_files: bool = False) -> dict[str, Any]:
    """Return privacy-conscious DB integrity diagnostics."""

    with closing(connect(db_path)) as connection:
        initialize_schema(connection)
        report = {
            "media_items": _media_item_checks(connection),
            "media_ocr": _media_ocr_checks(connection),
            "media_vlm": _media_vlm_checks(connection),
            "media_vlm_overrides": _media_vlm_override_checks(connection),
            "media_embeddings": _media_embedding_checks(connection),
            "line_messages": _line_message_checks(connection),
            "line_call_events": _line_call_event_checks(connection),
            "events": _event_checks(connection),
            "event_evidence": _event_evidence_checks(connection),
            "analysis_jobs": _analysis_job_checks(connection),
            "location_places": _location_place_checks(connection),
            "face_detections": _face_detection_checks(connection),
            "face_detection_runs": _face_detection_run_checks(connection),
            "face_embedding_clusters": _face_embedding_cluster_checks(connection),
            "persons": _person_checks(connection),
            "line_person_links": _line_person_link_checks(connection),
            "person_event_media": _person_event_media_checks(connection),
            "privacy_actions": _privacy_action_checks(connection),
        }
    report["strict"] = _strict_summary(report, fail_on_missing_files=fail_on_missing_files)
    return report


def format_db_check(report: dict[str, Any]) -> str:
    media = report["media_items"]
    media_ocr = report["media_ocr"]
    media_vlm = report["media_vlm"]
    media_vlm_overrides = report["media_vlm_overrides"]
    media_embeddings = report["media_embeddings"]
    line = report["line_messages"]
    calls = report["line_call_events"]
    events = report["events"]
    evidence = report["event_evidence"]
    analysis_jobs = report["analysis_jobs"]
    location_places = report["location_places"]
    face_detections = report["face_detections"]
    face_detection_runs = report["face_detection_runs"]
    face_embedding_clusters = report["face_embedding_clusters"]
    persons = report["persons"]
    line_person_links = report["line_person_links"]
    person_event_media = report["person_event_media"]
    privacy_actions = report["privacy_actions"]
    strict = report["strict"]

    lines = ["DB integrity check", ""]
    lines.extend(
        [
            "media_items:",
            f"- total: {media['total']}",
            f"- file_hash NULL: {media['file_hash_null']}",
            f"- unique file_hash: {media['unique_file_hash']}",
            f"- duplicate file_hash groups: {media['duplicate_file_hash_groups']}",
            f"- unique file_path: {media['unique_file_path']}",
            f"- duplicate file_path groups: {media['duplicate_file_path_groups']}",
            f"- captured_at NULL: {media['captured_at_null']}",
            f"- fallback_captured_at NULL: {media['fallback_captured_at_null']}",
            f"- GPSあり: {media['gps_present']}",
            f"- missing files: {media['missing_file_count']}",
            f"- missing thumbnails: {media['missing_thumbnail_count']}",
        ]
    )
    lines.extend(_sample_lines("duplicate file_hash sample IDs", media["duplicate_file_hash_sample_ids"]))
    lines.extend(_sample_lines("duplicate file_path sample IDs", media["duplicate_file_path_sample_ids"]))
    lines.extend(_sample_lines("missing file sample IDs", media["missing_file_sample_ids"]))
    lines.extend(_sample_lines("missing thumbnail sample IDs", media["missing_thumbnail_sample_ids"]))
    if media["missing_file_count"]:
        lines.append("- warning: missing original files are skipped by OCR/VLM/embedding jobs unless restored")

    lines.extend(
        [
            "",
            "media_ocr:",
            f"- total: {media_ocr['total']}",
            "- status counts:",
        ]
    )
    if media_ocr["status_counts"]:
        for row in media_ocr["status_counts"]:
            lines.append(f"  - {row['status']}: {row['count']}")
    else:
        lines.append("  - none")
    lines.extend(
        [
            f"- success: {media_ocr['success_count']}",
            f"- failed: {media_ocr['failed_count']}",
            f"- engine_unavailable: {media_ocr['engine_unavailable_count']}",
            f"- orphan media_id refs: {media_ocr['orphan_media_refs']}",
            f"- invalid status: {media_ocr['invalid_status_count']}",
            f"- success analyzed_at NULL: {media_ocr['success_analyzed_at_null']}",
            f"- ocr_text too long: {media_ocr['ocr_text_too_long']}",
        ]
    )
    lines.extend(_sample_lines("orphan OCR media IDs", media_ocr["orphan_media_sample_ids"]))
    lines.extend(_sample_lines("invalid OCR status media IDs", media_ocr["invalid_status_sample_ids"]))

    lines.extend(
        [
            "",
            "media_vlm:",
            f"- total: {media_vlm['total']}",
            "- status counts:",
        ]
    )
    if media_vlm["status_counts"]:
        for row in media_vlm["status_counts"]:
            lines.append(f"  - {row['status']}: {row['count']}")
    else:
        lines.append("  - none")
    lines.extend(
        [
            f"- success: {media_vlm['success_count']}",
            f"- failed: {media_vlm['failed_count']}",
            f"- engine_unavailable: {media_vlm['engine_unavailable_count']}",
            f"- orphan media_id refs: {media_vlm['orphan_media_refs']}",
            f"- invalid status: {media_vlm['invalid_status_count']}",
            f"- success caption NULL/empty: {media_vlm['success_caption_empty']}",
            f"- caption too long: {media_vlm['caption_too_long']}",
        ]
    )
    lines.extend(_sample_lines("orphan VLM media IDs", media_vlm["orphan_media_sample_ids"]))
    lines.extend(_sample_lines("invalid VLM status media IDs", media_vlm["invalid_status_sample_ids"]))

    lines.extend(
        [
            "",
            "media_vlm_overrides:",
            f"- total: {media_vlm_overrides['total']}",
            f"- orphan media_id refs: {media_vlm_overrides['orphan_media_refs']}",
            f"- hidden: {media_vlm_overrides['hidden_count']}",
            f"- wrong: {media_vlm_overrides['wrong_count']}",
            f"- not_searchable: {media_vlm_overrides['not_searchable_count']}",
            f"- not_event_usable: {media_vlm_overrides['not_event_usable_count']}",
            f"- unknown review_status: {media_vlm_overrides['unknown_status_count']}",
            f"- invalid JSON tags: {media_vlm_overrides['invalid_json_count']}",
            "- review_status counts:",
        ]
    )
    lines.extend(_count_rows_lines(media_vlm_overrides["status_counts"], "review_status"))
    lines.extend(_sample_lines("orphan VLM override media IDs", media_vlm_overrides["orphan_media_sample_ids"]))
    lines.extend(_sample_lines("invalid VLM override media IDs", media_vlm_overrides["invalid_json_sample_ids"]))

    lines.extend(
        [
            "",
            "media_embeddings:",
            f"- total: {media_embeddings['total']}",
            "- type counts:",
        ]
    )
    lines.extend(_count_rows_lines(media_embeddings["type_counts"], "embedding_type"))
    lines.append("- model counts:")
    lines.extend(_count_rows_lines(media_embeddings["model_counts"], "embedding_model"))
    lines.append("- status counts:")
    lines.extend(_count_rows_lines(media_embeddings["status_counts"], "status"))
    lines.extend(
        [
            f"- orphan media_id refs: {media_embeddings['orphan_media_refs']}",
            f"- invalid status: {media_embeddings['invalid_status_count']}",
            f"- unknown format: {media_embeddings['unknown_format_count']}",
            f"- success empty embedding: {media_embeddings['success_empty_embedding']}",
            f"- dimension mismatch: {media_embeddings['dimension_mismatch_count']}",
        ]
    )
    lines.extend(_sample_lines("orphan embedding media IDs", media_embeddings["orphan_media_sample_ids"]))
    lines.extend(_sample_lines("invalid embedding status media IDs", media_embeddings["invalid_status_sample_ids"]))
    lines.extend(_sample_lines("dimension mismatch embedding media IDs", media_embeddings["dimension_mismatch_sample_ids"]))

    lines.extend(
        [
            "",
            "line_messages:",
            f"- total: {line['total']}",
            f"- duplicate id groups: {line['duplicate_id_groups']}",
            f"- sent_at NULL: {line['sent_at_null']}",
            f"- text NULL/empty: {line['text_null_or_empty']}",
            "- chat_id counts:",
        ]
    )
    if line["chat_id_counts"]:
        for row in line["chat_id_counts"]:
            lines.append(f"  - {row['chat_id']}: {row['count']}")
    else:
        lines.append("  - none")

    lines.extend(
        [
            "",
            "line_call_events:",
            f"- total: {calls['total']}",
            "- call_status counts:",
        ]
    )
    if calls["status_counts"]:
        for row in calls["status_counts"]:
            lines.append(f"  - {row['call_status']}: {row['count']}")
    else:
        lines.append("  - none")
    lines.extend(
        [
            f"- orphan message_id refs: {calls['orphan_message_refs']}",
            f"- negative duration_sec: {calls['negative_duration_sec']}",
            f"- completed duration_sec NULL: {calls['completed_duration_null']}",
        ]
    )
    lines.extend(_sample_lines("orphan call message IDs", calls["orphan_message_sample_ids"]))

    lines.extend(
        [
            "",
            "events:",
            f"- total: {events['total']}",
            f"- date NULL: {events['date_null']}",
            f"- start_time NULL: {events['start_time_null']}",
            f"- end_time NULL: {events['end_time_null']}",
            f"- confidence min/max/avg: {_format_float(events['confidence_min'])} / "
            f"{_format_float(events['confidence_max'])} / {_format_float(events['confidence_avg'])}",
            "- event count by date:",
        ]
    )
    if events["date_counts"]:
        for row in events["date_counts"]:
            lines.append(f"  - {row['date']}: {row['count']}")
    else:
        lines.append("  - none")

    lines.extend(
        [
            "",
            "event_evidence:",
            f"- total: {evidence['total']}",
            "- evidence_type counts:",
        ]
    )
    if evidence["evidence_type_counts"]:
        for row in evidence["evidence_type_counts"]:
            lines.append(f"  - {row['evidence_type']}: {row['count']}")
    else:
        lines.append("  - none")
    lines.extend(
        [
            f"- missing photo evidence refs: {evidence['missing_photo_refs']}",
            f"- missing line evidence refs: {evidence['missing_line_refs']}",
            f"- missing VLM evidence refs: {evidence['missing_vlm_refs']}",
            f"- non-success VLM evidence refs: {evidence['non_success_vlm_refs']}",
            f"- failed VLM evidence refs: {evidence['failed_vlm_refs']}",
            f"- engine_unavailable VLM evidence refs: {evidence['engine_unavailable_vlm_refs']}",
            f"- fake VLM evidence refs: {evidence['fake_vlm_refs']}",
            f"- invalid VLM evidence refs: {evidence['invalid_vlm_refs']}",
            f"- orphan event refs: {evidence['orphan_event_refs']}",
        ]
    )
    lines.extend(_sample_lines("missing photo evidence IDs", evidence["missing_photo_sample_ids"]))
    lines.extend(_sample_lines("missing line evidence IDs", evidence["missing_line_sample_ids"]))
    lines.extend(_sample_lines("invalid VLM evidence IDs", evidence["invalid_vlm_sample_ids"]))
    lines.extend(_sample_lines("orphan event evidence IDs", evidence["orphan_event_sample_ids"]))

    lines.extend(
        [
            "",
            "analysis_jobs:",
            f"- total jobs: {analysis_jobs['total_jobs']}",
            f"- total job items: {analysis_jobs['total_items']}",
            "- job status counts:",
        ]
    )
    lines.extend(_count_rows_lines(analysis_jobs["job_status_counts"], "status"))
    lines.append("- item status counts:")
    lines.extend(_count_rows_lines(analysis_jobs["item_status_counts"], "status"))
    lines.extend(
        [
            f"- failed jobs: {analysis_jobs['failed_jobs']}",
            f"- stale running jobs: {analysis_jobs['stale_running_jobs']}",
            f"- orphan job items: {analysis_jobs['orphan_job_items']}",
            f"- invalid job status: {analysis_jobs['invalid_job_status_count']}",
            f"- invalid item status: {analysis_jobs['invalid_item_status_count']}",
            f"- item count mismatch: {analysis_jobs['item_count_mismatch']}",
        ]
    )
    lines.extend(_sample_lines("orphan job item IDs", analysis_jobs["orphan_job_item_sample_ids"]))
    lines.extend(_sample_lines("invalid job IDs", analysis_jobs["invalid_job_status_sample_ids"]))
    lines.extend(_sample_lines("invalid job item IDs", analysis_jobs["invalid_item_status_sample_ids"]))

    lines.extend(
        [
            "",
            "location/place:",
            f"- location_points total: {location_places['location_points_total']}",
            f"- location_points invalid lat/lon: {location_places['location_points_invalid_lat_lon']}",
            f"- location_points orphan media_id refs: {location_places['location_points_orphan_media_refs']}",
            f"- location_points orphan event_id refs: {location_places['location_points_orphan_event_refs']}",
            f"- location_points duplicate media_id: {location_places['location_points_duplicate_media_id']}",
            f"- location_points invalid privacy_level: {location_places['location_points_invalid_privacy_level']}",
            f"- place_clusters total: {location_places['place_clusters_total']}",
            f"- place_clusters invalid centroid: {location_places['place_clusters_invalid_centroid']}",
            f"- place_clusters point_count mismatch: {location_places['place_clusters_point_count_mismatch']}",
            f"- place_clusters invalid radius: {location_places['place_clusters_invalid_radius']}",
            f"- place_clusters invalid status: {location_places['place_clusters_invalid_status']}",
            f"- places total: {location_places['places_total']}",
            f"- places invalid privacy_level: {location_places['places_invalid_privacy_level']}",
            f"- places invalid privacy flags: {location_places['places_invalid_privacy_flags']}",
            f"- places orphan cluster_id: {location_places['places_orphan_cluster_refs']}",
            f"- places duplicate display_name: {location_places['places_duplicate_display_name']}",
            f"- event_places total: {location_places['event_places_total']}",
            f"- event_places orphan event_id refs: {location_places['event_places_orphan_event_refs']}",
            f"- event_places orphan place_id refs: {location_places['event_places_orphan_place_refs']}",
            f"- event_places invalid confidence: {location_places['event_places_invalid_confidence']}",
            f"- media_places total: {location_places['media_places_total']}",
            f"- media_places orphan media_id refs: {location_places['media_places_orphan_media_refs']}",
            f"- media_places orphan place_id refs: {location_places['media_places_orphan_place_refs']}",
            f"- media_places invalid confidence: {location_places['media_places_invalid_confidence']}",
        ]
    )
    lines.extend(_sample_lines("orphan location point media IDs", location_places["location_points_orphan_media_sample_ids"]))
    lines.extend(_sample_lines("orphan event_places IDs", location_places["event_places_orphan_sample_ids"]))
    lines.extend(_sample_lines("orphan media_places IDs", location_places["media_places_orphan_sample_ids"]))

    lines.extend(
        [
            "",
            "face_detections:",
            f"- total: {face_detections['total']}",
            "- status counts:",
        ]
    )
    lines.extend(_count_rows_lines(face_detections["status_counts"], "status"))
    lines.append("- review_status counts:")
    lines.extend(_count_rows_lines(face_detections["review_status_counts"], "review_status"))
    lines.extend(
        [
            f"- orphan media_id refs: {face_detections['orphan_media_refs']}",
            f"- invalid bbox: {face_detections['invalid_bbox']}",
            f"- missing crop files: {face_detections['missing_crop_files']}",
            f"- invalid review_status: {face_detections['invalid_review_status']}",
            f"- invalid privacy_level: {face_detections['invalid_privacy_level']}",
            f"- invalid hidden flag: {face_detections['invalid_hidden_flag']}",
        ]
    )
    lines.extend(_sample_lines("orphan face media IDs", face_detections["orphan_media_sample_ids"]))
    lines.extend(_sample_lines("invalid face bbox IDs", face_detections["invalid_bbox_sample_ids"]))
    lines.extend(_sample_lines("missing face crop IDs", face_detections["missing_crop_sample_ids"]))
    lines.extend(
        [
            "",
            "face_detection_runs:",
            f"- total: {face_detection_runs['total']}",
            "- status counts:",
        ]
    )
    lines.extend(_count_rows_lines(face_detection_runs["status_counts"], "status"))
    lines.extend(
        [
            f"- failed runs: {face_detection_runs['failed_runs']}",
            f"- stale running runs: {face_detection_runs['stale_running_runs']}",
            f"- invalid status: {face_detection_runs['invalid_status']}",
        ]
    )
    lines.extend(
        [
            "",
            "face embeddings/clusters:",
            f"- face_embeddings total: {face_embedding_clusters['face_embeddings_total']}",
            "- face_embedding status counts:",
        ]
    )
    lines.extend(_count_rows_lines(face_embedding_clusters["face_embedding_status_counts"], "status"))
    lines.extend(
        [
            f"- face_embeddings orphan face_id refs: {face_embedding_clusters['face_embeddings_orphan_face_refs']}",
            f"- face_embeddings invalid dim: {face_embedding_clusters['face_embeddings_invalid_dim']}",
            f"- face_embeddings success empty blob: {face_embedding_clusters['face_embeddings_success_empty_blob']}",
            f"- face_embeddings unknown format: {face_embedding_clusters['face_embeddings_unknown_format']}",
            f"- face_clusters total: {face_embedding_clusters['face_clusters_total']}",
            "- face_cluster status counts:",
        ]
    )
    lines.extend(_count_rows_lines(face_embedding_clusters["face_cluster_status_counts"], "status"))
    lines.extend(
        [
            f"- face_clusters invalid representative_face_id: {face_embedding_clusters['face_clusters_invalid_representative_face']}",
            f"- face_clusters invalid status: {face_embedding_clusters['face_clusters_invalid_status']}",
            f"- face_clusters invalid privacy_level: {face_embedding_clusters['face_clusters_invalid_privacy_level']}",
            f"- face_clusters empty cluster: {face_embedding_clusters['face_clusters_empty_cluster']}",
            f"- face_clusters singleton count: {face_embedding_clusters['face_clusters_singleton_count']}",
            f"- face_cluster_members orphan cluster_id refs: {face_embedding_clusters['face_cluster_members_orphan_cluster_refs']}",
            f"- face_cluster_members orphan face_id refs: {face_embedding_clusters['face_cluster_members_orphan_face_refs']}",
            f"- face_cluster_members duplicate members: {face_embedding_clusters['face_cluster_members_duplicate_member']}",
            f"- face_cluster_members invalid distance: {face_embedding_clusters['face_cluster_members_invalid_distance']}",
        ]
    )
    lines.extend(_sample_lines("orphan face embedding IDs", face_embedding_clusters["face_embeddings_orphan_sample_ids"]))
    lines.extend(_sample_lines("invalid face cluster representative IDs", face_embedding_clusters["face_clusters_invalid_representative_sample_ids"]))
    lines.extend(_sample_lines("orphan face cluster member IDs", face_embedding_clusters["face_cluster_members_orphan_sample_ids"]))

    lines.extend(
        [
            "",
            "persons:",
            f"- total: {persons['persons_total']}",
            f"- invalid privacy_level: {persons['persons_invalid_privacy_level']}",
            f"- invalid privacy flags: {persons['persons_invalid_privacy_flags']}",
            f"- hidden: {persons['persons_hidden']}",
            f"- deleted: {persons['persons_deleted']}",
            f"- deleted person links warning: {persons['persons_deleted_links_warning']}",
            f"- duplicate display_name: {persons['persons_duplicate_display_name']}",
            f"- public_alias missing public_name: {persons['persons_public_alias_missing_public_name']}",
            f"- person_face_clusters total: {persons['person_face_clusters_total']}",
            f"- person_face_clusters orphan person_id refs: {persons['person_face_clusters_orphan_person_refs']}",
            f"- person_face_clusters orphan cluster_id refs: {persons['person_face_clusters_orphan_cluster_refs']}",
            f"- person_face_clusters duplicate links: {persons['person_face_clusters_duplicate_links']}",
            f"- person_face_clusters rejected cluster links: {persons['person_face_clusters_rejected_cluster_links']}",
            f"- person_aliases total: {persons['person_aliases_total']}",
            f"- person_aliases orphan person_id refs: {persons['person_aliases_orphan_person_refs']}",
            f"- person_aliases duplicate aliases: {persons['person_aliases_duplicate_aliases']}",
            f"- person_aliases invalid source: {persons['person_aliases_invalid_source']}",
        ]
    )
    lines.extend(_sample_lines("orphan person face cluster links", persons["person_face_clusters_orphan_sample_ids"]))
    lines.extend(_sample_lines("orphan person alias IDs", persons["person_aliases_orphan_sample_ids"]))

    lines.extend(
        [
            "",
            "line speaker/person links:",
            f"- line_speaker_links total: {line_person_links['line_speaker_links_total']}",
            f"- orphan person_id refs: {line_person_links['line_speaker_links_orphan_person_refs']}",
            f"- empty speaker_name: {line_person_links['line_speaker_links_empty_speaker_name']}",
            f"- empty chat_id: {line_person_links['line_speaker_links_empty_chat_id']}",
            f"- duplicate links: {line_person_links['line_speaker_links_duplicate_links']}",
            f"- invalid confidence: {line_person_links['line_speaker_links_invalid_confidence']}",
            f"- person_line_mentions total: {line_person_links['person_line_mentions_total']}",
            f"- person_line_mentions orphan person_id refs: {line_person_links['person_line_mentions_orphan_person_refs']}",
            f"- person_line_mentions orphan message_id refs: {line_person_links['person_line_mentions_orphan_message_refs']}",
            f"- person_line_mentions invalid mention_type: {line_person_links['person_line_mentions_invalid_mention_type']}",
        ]
    )
    lines.extend(_sample_lines("orphan line speaker links", line_person_links["line_speaker_links_orphan_sample_ids"]))
    lines.extend(_sample_lines("orphan person line mentions", line_person_links["person_line_mentions_orphan_sample_ids"]))

    lines.extend(
        [
            "",
            "person media/event links:",
            f"- media_people total: {person_event_media['media_people_total']}",
            f"- media_people orphan media_id refs: {person_event_media['media_people_orphan_media_refs']}",
            f"- media_people orphan person_id refs: {person_event_media['media_people_orphan_person_refs']}",
            f"- media_people orphan face_id refs: {person_event_media['media_people_orphan_face_refs']}",
            f"- media_people orphan face_cluster_id refs: {person_event_media['media_people_orphan_cluster_refs']}",
            f"- media_people invalid source: {person_event_media['media_people_invalid_source']}",
            f"- media_people invalid confidence: {person_event_media['media_people_invalid_confidence']}",
            f"- media_people invalid hidden flag: {person_event_media['media_people_invalid_hidden_flag']}",
            f"- media_people deleted/hidden person links warning: {person_event_media['media_people_hidden_deleted_person_links']}",
            f"- media_people unverified cluster links: {person_event_media['media_people_unverified_cluster_links']}",
            f"- media_people rejected detection links: {person_event_media['media_people_rejected_detection_links']}",
            f"- event_people total: {person_event_media['event_people_total']}",
            f"- event_people orphan event_id refs: {person_event_media['event_people_orphan_event_refs']}",
            f"- event_people orphan person_id refs: {person_event_media['event_people_orphan_person_refs']}",
            f"- event_people invalid source: {person_event_media['event_people_invalid_source']}",
            f"- event_people invalid confidence: {person_event_media['event_people_invalid_confidence']}",
            f"- event_people invalid hidden flag: {person_event_media['event_people_invalid_hidden_flag']}",
            f"- event_people deleted/hidden person links warning: {person_event_media['event_people_hidden_deleted_person_links']}",
            f"- event_people duplicate links: {person_event_media['event_people_duplicate_links']}",
            f"- event_people unverified person links: {person_event_media['event_people_unverified_person_links']}",
        ]
    )
    lines.extend(_sample_lines("orphan media_people IDs", person_event_media["media_people_orphan_sample_ids"]))
    lines.extend(_sample_lines("orphan event_people IDs", person_event_media["event_people_orphan_sample_ids"]))

    lines.extend(
        [
            "",
            "privacy actions:",
            f"- total: {privacy_actions['privacy_actions_total']}",
            f"- invalid action_type: {privacy_actions['privacy_actions_invalid_action_type']}",
            f"- invalid target_type: {privacy_actions['privacy_actions_invalid_target_type']}",
            f"- invalid mode: {privacy_actions['privacy_actions_invalid_mode']}",
        ]
    )

    lines.extend(["", "strict:"])
    lines.append(f"- ok: {strict['ok']}")
    if strict["issues"]:
        for issue in strict["issues"]:
            lines.append(f"  - {issue}")
    else:
        lines.append("  - no severe issues")
    return "\n".join(lines)


def _media_item_checks(connection) -> dict[str, Any]:
    rows = connection.execute(
        """
        SELECT id, file_path, file_hash, captured_at, fallback_captured_at,
               gps_lat, gps_lon, thumbnail_path
        FROM media_items
        ORDER BY id ASC
        """
    ).fetchall()
    total = len(rows)
    missing_file_ids: list[str] = []
    missing_thumbnail_ids: list[str] = []

    for row in rows:
        file_path = row["file_path"]
        if file_path and not Path(str(file_path)).expanduser().exists():
            _append_sample(missing_file_ids, row["id"])
        thumbnail_path = row["thumbnail_path"]
        if thumbnail_path and not Path(str(thumbnail_path)).expanduser().exists():
            _append_sample(missing_thumbnail_ids, row["id"])

    duplicate_hash_rows = _duplicate_value_samples(connection, "media_items", "file_hash")
    duplicate_path_rows = _duplicate_value_samples(connection, "media_items", "file_path")

    return {
        "total": total,
        "file_hash_null": _count(connection, "SELECT COUNT(*) FROM media_items WHERE file_hash IS NULL OR TRIM(file_hash) = ''"),
        "unique_file_hash": _count(connection, "SELECT COUNT(DISTINCT file_hash) FROM media_items WHERE file_hash IS NOT NULL AND TRIM(file_hash) != ''"),
        "duplicate_file_hash_groups": len(duplicate_hash_rows),
        "duplicate_file_hash_sample_ids": _flatten_sample_ids(duplicate_hash_rows),
        "unique_file_path": _count(connection, "SELECT COUNT(DISTINCT file_path) FROM media_items WHERE file_path IS NOT NULL AND TRIM(file_path) != ''"),
        "duplicate_file_path_groups": len(duplicate_path_rows),
        "duplicate_file_path_sample_ids": _flatten_sample_ids(duplicate_path_rows),
        "captured_at_null": _count(connection, "SELECT COUNT(*) FROM media_items WHERE captured_at IS NULL OR TRIM(captured_at) = ''"),
        "fallback_captured_at_null": _count(connection, "SELECT COUNT(*) FROM media_items WHERE fallback_captured_at IS NULL OR TRIM(fallback_captured_at) = ''"),
        "gps_present": _count(connection, "SELECT COUNT(*) FROM media_items WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL"),
        "missing_file_count": _missing_path_count(rows, "file_path"),
        "missing_file_sample_ids": missing_file_ids,
        "missing_thumbnail_count": _missing_path_count([row for row in rows if row["thumbnail_path"]], "thumbnail_path"),
        "missing_thumbnail_sample_ids": missing_thumbnail_ids,
    }


def _line_message_checks(connection) -> dict[str, Any]:
    return {
        "total": _count(connection, "SELECT COUNT(*) FROM line_messages"),
        "duplicate_id_groups": len(_duplicate_value_samples(connection, "line_messages", "id", include_null=False)),
        "sent_at_null": _count(connection, "SELECT COUNT(*) FROM line_messages WHERE sent_at IS NULL OR TRIM(sent_at) = ''"),
        "text_null_or_empty": _count(connection, "SELECT COUNT(*) FROM line_messages WHERE text IS NULL OR TRIM(text) = ''"),
        "chat_id_counts": _rows(
            connection,
            """
            SELECT COALESCE(chat_id, '(null)') AS chat_id, COUNT(*) AS count
            FROM line_messages
            GROUP BY chat_id
            ORDER BY count DESC, chat_id ASC
            LIMIT ?
            """,
            [CHAT_ID_LIMIT],
        ),
    }


def _media_ocr_checks(connection) -> dict[str, Any]:
    valid_statuses = (
        "pending",
        "success",
        "skipped",
        "failed",
        "no_text",
        "no_text_detected",
        "engine_unavailable",
    )
    placeholders = ", ".join("?" for _ in valid_statuses)
    return {
        "total": _count(connection, "SELECT COUNT(*) FROM media_ocr"),
        "status_counts": _rows(
            connection,
            """
            SELECT COALESCE(status, '(null)') AS status, COUNT(*) AS count
            FROM media_ocr
            GROUP BY status
            ORDER BY count DESC, status ASC
            """,
        ),
        "success_count": _count(connection, "SELECT COUNT(*) FROM media_ocr WHERE status = 'success'"),
        "failed_count": _count(connection, "SELECT COUNT(*) FROM media_ocr WHERE status = 'failed'"),
        "engine_unavailable_count": _count(connection, "SELECT COUNT(*) FROM media_ocr WHERE status = 'engine_unavailable'"),
        "orphan_media_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM media_ocr
            LEFT JOIN media_items ON media_items.id = media_ocr.media_id
            WHERE media_items.id IS NULL
            """,
        ),
        "orphan_media_sample_ids": _sample_query(
            connection,
            """
            SELECT media_ocr.media_id AS id
            FROM media_ocr
            LEFT JOIN media_items ON media_items.id = media_ocr.media_id
            WHERE media_items.id IS NULL
            ORDER BY media_ocr.media_id ASC
            LIMIT ?
            """,
        ),
        "invalid_status_count": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM media_ocr
            WHERE status IS NULL OR status NOT IN ({placeholders})
            """,
            list(valid_statuses),
        ),
        "invalid_status_sample_ids": _sample_query_params(
            connection,
            f"""
            SELECT media_id AS id
            FROM media_ocr
            WHERE status IS NULL OR status NOT IN ({placeholders})
            ORDER BY media_id ASC
            LIMIT ?
            """,
            list(valid_statuses),
        ),
        "success_analyzed_at_null": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM media_ocr
            WHERE status = 'success'
              AND (analyzed_at IS NULL OR TRIM(analyzed_at) = '')
            """,
        ),
        "ocr_text_too_long": _count(
            connection,
            "SELECT COUNT(*) FROM media_ocr WHERE LENGTH(COALESCE(ocr_text, '')) > 20000",
        ),
    }


def _media_vlm_checks(connection) -> dict[str, Any]:
    valid_statuses = ("pending", "success", "skipped", "failed", "no_visual_content", "engine_unavailable")
    placeholders = ", ".join("?" for _ in valid_statuses)
    return {
        "total": _count(connection, "SELECT COUNT(*) FROM media_vlm"),
        "status_counts": _rows(
            connection,
            """
            SELECT COALESCE(status, '(null)') AS status, COUNT(*) AS count
            FROM media_vlm
            GROUP BY status
            ORDER BY count DESC, status ASC
            """,
        ),
        "success_count": _count(connection, "SELECT COUNT(*) FROM media_vlm WHERE status = 'success'"),
        "failed_count": _count(connection, "SELECT COUNT(*) FROM media_vlm WHERE status = 'failed'"),
        "engine_unavailable_count": _count(connection, "SELECT COUNT(*) FROM media_vlm WHERE status = 'engine_unavailable'"),
        "orphan_media_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM media_vlm
            LEFT JOIN media_items ON media_items.id = media_vlm.media_id
            WHERE media_items.id IS NULL
            """,
        ),
        "orphan_media_sample_ids": _sample_query(
            connection,
            """
            SELECT media_vlm.media_id AS id
            FROM media_vlm
            LEFT JOIN media_items ON media_items.id = media_vlm.media_id
            WHERE media_items.id IS NULL
            ORDER BY media_vlm.media_id ASC
            LIMIT ?
            """,
        ),
        "invalid_status_count": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM media_vlm
            WHERE status IS NULL OR status NOT IN ({placeholders})
            """,
            list(valid_statuses),
        ),
        "invalid_status_sample_ids": _sample_query_params(
            connection,
            f"""
            SELECT media_id AS id
            FROM media_vlm
            WHERE status IS NULL OR status NOT IN ({placeholders})
            ORDER BY media_id ASC
            LIMIT ?
            """,
            list(valid_statuses),
        ),
        "success_caption_empty": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM media_vlm
            WHERE status = 'success'
              AND (
                  (caption IS NULL OR TRIM(caption) = '')
                  OR (short_caption IS NULL OR TRIM(short_caption) = '')
              )
            """,
        ),
        "caption_too_long": _count(
            connection,
            "SELECT COUNT(*) FROM media_vlm WHERE LENGTH(COALESCE(caption, '')) > 2000",
        ),
    }


def _media_vlm_override_checks(connection) -> dict[str, Any]:
    valid_statuses = ("unreviewed", "accepted", "rejected", "needs_fix", "wrong")
    placeholders = ", ".join("?" for _ in valid_statuses)
    rows = _rows(
        connection,
        """
        SELECT media_id,
               scene_tags_override_json,
               object_tags_override_json,
               activity_tags_override_json,
               food_cues_override_json,
               location_cues_override_json
        FROM media_vlm_overrides
        ORDER BY media_id ASC
        """,
    )
    invalid_json_ids: list[str] = []
    for row in rows:
        for key in (
            "scene_tags_override_json",
            "object_tags_override_json",
            "activity_tags_override_json",
            "food_cues_override_json",
            "location_cues_override_json",
        ):
            raw = row.get(key)
            if raw and not _is_valid_json(raw):
                _append_sample(invalid_json_ids, row.get("media_id"))
                break
    return {
        "total": _count(connection, "SELECT COUNT(*) FROM media_vlm_overrides"),
        "status_counts": _rows(
            connection,
            """
            SELECT COALESCE(review_status, '(null)') AS review_status, COUNT(*) AS count
            FROM media_vlm_overrides
            GROUP BY review_status
            ORDER BY count DESC, review_status ASC
            """,
        ),
        "orphan_media_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM media_vlm_overrides
            LEFT JOIN media_items ON media_items.id = media_vlm_overrides.media_id
            WHERE media_items.id IS NULL
            """,
        ),
        "orphan_media_sample_ids": _sample_query(
            connection,
            """
            SELECT media_vlm_overrides.media_id AS id
            FROM media_vlm_overrides
            LEFT JOIN media_items ON media_items.id = media_vlm_overrides.media_id
            WHERE media_items.id IS NULL
            ORDER BY media_vlm_overrides.media_id ASC
            LIMIT ?
            """,
        ),
        "hidden_count": _count(connection, "SELECT COUNT(*) FROM media_vlm_overrides WHERE COALESCE(is_hidden, 0) = 1"),
        "wrong_count": _count(connection, "SELECT COUNT(*) FROM media_vlm_overrides WHERE COALESCE(is_wrong, 0) = 1 OR review_status = 'wrong'"),
        "not_searchable_count": _count(connection, "SELECT COUNT(*) FROM media_vlm_overrides WHERE COALESCE(is_searchable, 1) = 0"),
        "not_event_usable_count": _count(connection, "SELECT COUNT(*) FROM media_vlm_overrides WHERE COALESCE(is_event_usable, 1) = 0"),
        "unknown_status_count": _count_params(
            connection,
            f"SELECT COUNT(*) FROM media_vlm_overrides WHERE review_status IS NULL OR review_status NOT IN ({placeholders})",
            list(valid_statuses),
        ),
        "invalid_json_count": len(invalid_json_ids),
        "invalid_json_sample_ids": invalid_json_ids,
    }


def _media_embedding_checks(connection) -> dict[str, Any]:
    valid_statuses = ("pending", "success", "skipped", "failed", "engine_unavailable")
    valid_formats = ("float32_numpy", "json")
    status_placeholders = ", ".join("?" for _ in valid_statuses)
    format_placeholders = ", ".join("?" for _ in valid_formats)
    return {
        "total": _count(connection, "SELECT COUNT(*) FROM media_embeddings"),
        "type_counts": _rows(
            connection,
            """
            SELECT COALESCE(embedding_type, '(null)') AS embedding_type, COUNT(*) AS count
            FROM media_embeddings
            GROUP BY embedding_type
            ORDER BY count DESC, embedding_type ASC
            """,
        ),
        "model_counts": _rows(
            connection,
            """
            SELECT COALESCE(embedding_model, '(null)') AS embedding_model, COUNT(*) AS count
            FROM media_embeddings
            GROUP BY embedding_model
            ORDER BY count DESC, embedding_model ASC
            """,
        ),
        "status_counts": _rows(
            connection,
            """
            SELECT COALESCE(status, '(null)') AS status, COUNT(*) AS count
            FROM media_embeddings
            GROUP BY status
            ORDER BY count DESC, status ASC
            """,
        ),
        "orphan_media_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM media_embeddings
            LEFT JOIN media_items ON media_items.id = media_embeddings.media_id
            WHERE media_items.id IS NULL
            """,
        ),
        "orphan_media_sample_ids": _sample_query(
            connection,
            """
            SELECT media_embeddings.media_id AS id
            FROM media_embeddings
            LEFT JOIN media_items ON media_items.id = media_embeddings.media_id
            WHERE media_items.id IS NULL
            ORDER BY media_embeddings.media_id ASC
            LIMIT ?
            """,
        ),
        "invalid_status_count": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM media_embeddings
            WHERE status IS NULL OR status NOT IN ({status_placeholders})
            """,
            list(valid_statuses),
        ),
        "invalid_status_sample_ids": _sample_query_params(
            connection,
            f"""
            SELECT media_id AS id
            FROM media_embeddings
            WHERE status IS NULL OR status NOT IN ({status_placeholders})
            ORDER BY media_id ASC
            LIMIT ?
            """,
            list(valid_statuses),
        ),
        "unknown_format_count": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM media_embeddings
            WHERE embedding_format IS NULL OR embedding_format NOT IN ({format_placeholders})
            """,
            list(valid_formats),
        ),
        "success_empty_embedding": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM media_embeddings
            WHERE status = 'success'
              AND (embedding IS NULL OR length(embedding) = 0)
            """,
        ),
        "dimension_mismatch_count": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM media_embeddings
            WHERE status = 'success'
              AND embedding_format = 'float32_numpy'
              AND (
                  embedding_dim IS NULL
                  OR embedding_dim <= 0
                  OR embedding IS NULL
                  OR length(embedding) != embedding_dim * 4
              )
            """,
        ),
        "dimension_mismatch_sample_ids": _sample_query(
            connection,
            """
            SELECT media_id AS id
            FROM media_embeddings
            WHERE status = 'success'
              AND embedding_format = 'float32_numpy'
              AND (
                  embedding_dim IS NULL
                  OR embedding_dim <= 0
                  OR embedding IS NULL
                  OR length(embedding) != embedding_dim * 4
              )
            ORDER BY media_id ASC
            LIMIT ?
            """,
        ),
    }


def _line_call_event_checks(connection) -> dict[str, Any]:
    return {
        "total": _count(connection, "SELECT COUNT(*) FROM line_call_events"),
        "status_counts": _rows(
            connection,
            """
            SELECT COALESCE(call_status, '(null)') AS call_status, COUNT(*) AS count
            FROM line_call_events
            GROUP BY call_status
            ORDER BY count DESC, call_status ASC
            """,
        ),
        "orphan_message_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM line_call_events
            LEFT JOIN line_messages ON line_messages.id = line_call_events.message_id
            WHERE line_messages.id IS NULL
            """,
        ),
        "orphan_message_sample_ids": _sample_query(
            connection,
            """
            SELECT line_call_events.message_id AS id
            FROM line_call_events
            LEFT JOIN line_messages ON line_messages.id = line_call_events.message_id
            WHERE line_messages.id IS NULL
            ORDER BY line_call_events.message_id ASC
            LIMIT ?
            """,
        ),
        "negative_duration_sec": _count(
            connection,
            "SELECT COUNT(*) FROM line_call_events WHERE duration_sec < 0",
        ),
        "completed_duration_null": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM line_call_events
            WHERE call_status = 'completed'
              AND duration_sec IS NULL
            """,
        ),
    }


def _event_checks(connection) -> dict[str, Any]:
    stats = connection.execute(
        """
        SELECT
            MIN(confidence) AS confidence_min,
            MAX(confidence) AS confidence_max,
            AVG(confidence) AS confidence_avg
        FROM events
        WHERE confidence IS NOT NULL
        """
    ).fetchone()
    return {
        "total": _count(connection, "SELECT COUNT(*) FROM events"),
        "date_null": _count(connection, "SELECT COUNT(*) FROM events WHERE date IS NULL OR TRIM(date) = ''"),
        "start_time_null": _count(connection, "SELECT COUNT(*) FROM events WHERE start_time IS NULL OR TRIM(start_time) = ''"),
        "end_time_null": _count(connection, "SELECT COUNT(*) FROM events WHERE end_time IS NULL OR TRIM(end_time) = ''"),
        "confidence_min": stats["confidence_min"],
        "confidence_max": stats["confidence_max"],
        "confidence_avg": stats["confidence_avg"],
        "date_counts": _rows(
            connection,
            """
            SELECT COALESCE(date, '(null)') AS date, COUNT(*) AS count
            FROM events
            GROUP BY date
            ORDER BY date ASC
            LIMIT 100
            """,
        ),
    }


def _event_evidence_checks(connection) -> dict[str, Any]:
    return {
        "total": _count(connection, "SELECT COUNT(*) FROM event_evidence"),
        "evidence_type_counts": _rows(
            connection,
            """
            SELECT evidence_type, COUNT(*) AS count
            FROM event_evidence
            GROUP BY evidence_type
            ORDER BY count DESC, evidence_type ASC
            """,
        ),
        "missing_photo_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM event_evidence
            LEFT JOIN media_items ON media_items.id = event_evidence.evidence_id
            WHERE event_evidence.evidence_type = 'photo'
              AND media_items.id IS NULL
            """,
        ),
        "missing_photo_sample_ids": _sample_query(
            connection,
            """
            SELECT event_evidence.evidence_id AS id
            FROM event_evidence
            LEFT JOIN media_items ON media_items.id = event_evidence.evidence_id
            WHERE event_evidence.evidence_type = 'photo'
              AND media_items.id IS NULL
            ORDER BY event_evidence.evidence_id ASC
            LIMIT ?
            """,
        ),
        "missing_line_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM event_evidence
            LEFT JOIN line_messages ON line_messages.id = event_evidence.evidence_id
            WHERE event_evidence.evidence_type = 'line'
              AND line_messages.id IS NULL
            """,
        ),
        "missing_line_sample_ids": _sample_query(
            connection,
            """
            SELECT event_evidence.evidence_id AS id
            FROM event_evidence
            LEFT JOIN line_messages ON line_messages.id = event_evidence.evidence_id
            WHERE event_evidence.evidence_type = 'line'
              AND line_messages.id IS NULL
            ORDER BY event_evidence.evidence_id ASC
            LIMIT ?
            """,
        ),
        "missing_vlm_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM event_evidence
            LEFT JOIN media_vlm ON media_vlm.media_id = event_evidence.evidence_id
            WHERE event_evidence.evidence_type = 'vlm'
              AND media_vlm.media_id IS NULL
            """,
        ),
        "non_success_vlm_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM event_evidence
            JOIN media_vlm ON media_vlm.media_id = event_evidence.evidence_id
            WHERE event_evidence.evidence_type = 'vlm'
              AND COALESCE(media_vlm.status, '') != 'success'
            """,
        ),
        "failed_vlm_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM event_evidence
            JOIN media_vlm ON media_vlm.media_id = event_evidence.evidence_id
            WHERE event_evidence.evidence_type = 'vlm'
              AND media_vlm.status = 'failed'
            """,
        ),
        "engine_unavailable_vlm_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM event_evidence
            JOIN media_vlm ON media_vlm.media_id = event_evidence.evidence_id
            WHERE event_evidence.evidence_type = 'vlm'
              AND media_vlm.status = 'engine_unavailable'
            """,
        ),
        "fake_vlm_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM event_evidence
            JOIN media_vlm ON media_vlm.media_id = event_evidence.evidence_id
            WHERE event_evidence.evidence_type = 'vlm'
              AND (
                LOWER(COALESCE(media_vlm.vlm_engine, '')) LIKE '%fake%'
                OR LOWER(COALESCE(media_vlm.model_name, '')) LIKE '%fake%'
              )
            """,
        ),
        "invalid_vlm_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM event_evidence
            LEFT JOIN media_vlm ON media_vlm.media_id = event_evidence.evidence_id
            WHERE event_evidence.evidence_type = 'vlm'
              AND (
                media_vlm.media_id IS NULL
                OR COALESCE(media_vlm.status, '') != 'success'
                OR LOWER(COALESCE(media_vlm.vlm_engine, '')) LIKE '%fake%'
                OR LOWER(COALESCE(media_vlm.model_name, '')) LIKE '%fake%'
              )
            """,
        ),
        "invalid_vlm_sample_ids": _sample_query(
            connection,
            """
            SELECT event_evidence.event_id || ':' || event_evidence.evidence_id AS id
            FROM event_evidence
            LEFT JOIN media_vlm ON media_vlm.media_id = event_evidence.evidence_id
            WHERE event_evidence.evidence_type = 'vlm'
              AND (
                media_vlm.media_id IS NULL
                OR COALESCE(media_vlm.status, '') != 'success'
                OR LOWER(COALESCE(media_vlm.vlm_engine, '')) LIKE '%fake%'
                OR LOWER(COALESCE(media_vlm.model_name, '')) LIKE '%fake%'
              )
            ORDER BY event_evidence.event_id ASC, event_evidence.evidence_id ASC
            LIMIT ?
            """,
        ),
        "orphan_event_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM event_evidence
            LEFT JOIN events ON events.id = event_evidence.event_id
            WHERE events.id IS NULL
            """,
        ),
        "orphan_event_sample_ids": _sample_query(
            connection,
            """
            SELECT event_evidence.event_id AS id
            FROM event_evidence
            LEFT JOIN events ON events.id = event_evidence.event_id
            WHERE events.id IS NULL
            ORDER BY event_evidence.event_id ASC
            LIMIT ?
            """,
        ),
    }


def _analysis_job_checks(connection) -> dict[str, Any]:
    valid_job_statuses = ("planned", "running", "completed", "failed", "canceled", "partial")
    valid_item_statuses = ("pending", "running", "success", "failed", "skipped", "engine_unavailable")
    job_placeholders = ", ".join("?" for _ in valid_job_statuses)
    item_placeholders = ", ".join("?" for _ in valid_item_statuses)
    return {
        "total_jobs": _count(connection, "SELECT COUNT(*) FROM analysis_jobs"),
        "total_items": _count(connection, "SELECT COUNT(*) FROM analysis_job_items"),
        "job_status_counts": _rows(
            connection,
            """
            SELECT COALESCE(status, '(null)') AS status, COUNT(*) AS count
            FROM analysis_jobs
            GROUP BY status
            ORDER BY count DESC, status ASC
            """,
        ),
        "item_status_counts": _rows(
            connection,
            """
            SELECT COALESCE(status, '(null)') AS status, COUNT(*) AS count
            FROM analysis_job_items
            GROUP BY status
            ORDER BY count DESC, status ASC
            """,
        ),
        "failed_jobs": _count(connection, "SELECT COUNT(*) FROM analysis_jobs WHERE status = 'failed'"),
        "stale_running_jobs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM analysis_jobs
            WHERE status = 'running'
              AND julianday('now') - julianday(COALESCE(started_at, created_at)) > 1
            """,
        ),
        "orphan_job_items": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM analysis_job_items
            LEFT JOIN analysis_jobs ON analysis_jobs.job_id = analysis_job_items.job_id
            WHERE analysis_jobs.job_id IS NULL
            """,
        ),
        "orphan_job_item_sample_ids": _sample_query(
            connection,
            """
            SELECT analysis_job_items.job_id || ':' || analysis_job_items.item_id AS id
            FROM analysis_job_items
            LEFT JOIN analysis_jobs ON analysis_jobs.job_id = analysis_job_items.job_id
            WHERE analysis_jobs.job_id IS NULL
            ORDER BY analysis_job_items.job_id ASC, analysis_job_items.item_id ASC
            LIMIT ?
            """,
        ),
        "invalid_job_status_count": _count_params(
            connection,
            f"SELECT COUNT(*) FROM analysis_jobs WHERE status IS NULL OR status NOT IN ({job_placeholders})",
            list(valid_job_statuses),
        ),
        "invalid_job_status_sample_ids": _sample_query_params(
            connection,
            f"""
            SELECT job_id AS id
            FROM analysis_jobs
            WHERE status IS NULL OR status NOT IN ({job_placeholders})
            ORDER BY job_id ASC
            LIMIT ?
            """,
            list(valid_job_statuses),
        ),
        "invalid_item_status_count": _count_params(
            connection,
            f"SELECT COUNT(*) FROM analysis_job_items WHERE status IS NULL OR status NOT IN ({item_placeholders})",
            list(valid_item_statuses),
        ),
        "invalid_item_status_sample_ids": _sample_query_params(
            connection,
            f"""
            SELECT job_id || ':' || item_id AS id
            FROM analysis_job_items
            WHERE status IS NULL OR status NOT IN ({item_placeholders})
            ORDER BY job_id ASC, item_id ASC
            LIMIT ?
            """,
            list(valid_item_statuses),
        ),
        "item_count_mismatch": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM analysis_jobs
            LEFT JOIN (
                SELECT job_id, COUNT(*) AS actual_count
                FROM analysis_job_items
                GROUP BY job_id
            ) item_counts ON item_counts.job_id = analysis_jobs.job_id
            WHERE COALESCE(analysis_jobs.total_items, 0) != COALESCE(item_counts.actual_count, 0)
            """,
        ),
    }


def _location_place_checks(connection) -> dict[str, Any]:
    location_privacy = ("exact_private", "approximate_private", "public_hidden", "public_place_label")
    cluster_statuses = ("unreviewed", "accepted", "rejected", "merged")
    place_privacy = ("private", "public_label", "public_hidden")
    location_privacy_placeholders = ", ".join("?" for _ in location_privacy)
    cluster_status_placeholders = ", ".join("?" for _ in cluster_statuses)
    place_privacy_placeholders = ", ".join("?" for _ in place_privacy)
    return {
        "location_points_total": _count(connection, "SELECT COUNT(*) FROM location_points"),
        "location_points_invalid_lat_lon": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM location_points
            WHERE lat IS NULL OR lon IS NULL
               OR lat < -90 OR lat > 90
               OR lon < -180 OR lon > 180
            """,
        ),
        "location_points_orphan_media_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM location_points
            LEFT JOIN media_items ON media_items.id = location_points.media_id
            WHERE location_points.media_id IS NOT NULL
              AND media_items.id IS NULL
            """,
        ),
        "location_points_orphan_media_sample_ids": _sample_query(
            connection,
            """
            SELECT location_points.media_id AS id
            FROM location_points
            LEFT JOIN media_items ON media_items.id = location_points.media_id
            WHERE location_points.media_id IS NOT NULL
              AND media_items.id IS NULL
            ORDER BY location_points.media_id ASC
            LIMIT ?
            """,
        ),
        "location_points_orphan_event_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM location_points
            LEFT JOIN events ON events.id = location_points.event_id
            WHERE location_points.event_id IS NOT NULL
              AND events.id IS NULL
            """,
        ),
        "location_points_duplicate_media_id": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM (
                SELECT media_id
                FROM location_points
                WHERE media_id IS NOT NULL
                GROUP BY media_id
                HAVING COUNT(*) > 1
            )
            """,
        ),
        "location_points_invalid_privacy_level": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM location_points
            WHERE privacy_level IS NULL OR privacy_level NOT IN ({location_privacy_placeholders})
            """,
            list(location_privacy),
        ),
        "place_clusters_total": _count(connection, "SELECT COUNT(*) FROM place_clusters"),
        "place_clusters_invalid_centroid": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM place_clusters
            WHERE centroid_lat IS NULL OR centroid_lon IS NULL
               OR centroid_lat < -90 OR centroid_lat > 90
               OR centroid_lon < -180 OR centroid_lon > 180
            """,
        ),
        "place_clusters_point_count_mismatch": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM place_clusters
            LEFT JOIN (
                SELECT cluster_id, COUNT(*) AS actual_count
                FROM location_points
                WHERE cluster_id IS NOT NULL
                GROUP BY cluster_id
            ) point_counts ON point_counts.cluster_id = place_clusters.id
            WHERE COALESCE(place_clusters.point_count, 0) != COALESCE(point_counts.actual_count, 0)
            """,
        ),
        "place_clusters_invalid_radius": _count(
            connection,
            "SELECT COUNT(*) FROM place_clusters WHERE radius_m IS NULL OR radius_m < 0",
        ),
        "place_clusters_invalid_status": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM place_clusters
            WHERE status IS NULL OR status NOT IN ({cluster_status_placeholders})
            """,
            list(cluster_statuses),
        ),
        "places_total": _count(connection, "SELECT COUNT(*) FROM places"),
        "places_invalid_privacy_level": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM places
            WHERE privacy_level IS NULL OR privacy_level NOT IN ({place_privacy_placeholders})
            """,
            list(place_privacy),
        ),
        "places_invalid_privacy_flags": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM places
            WHERE COALESCE(hidden, 0) NOT IN (0, 1)
               OR COALESCE(searchable, 1) NOT IN (0, 1)
            """,
        ),
        "places_orphan_cluster_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM places
            LEFT JOIN place_clusters ON place_clusters.id = places.cluster_id
            WHERE places.cluster_id IS NOT NULL
              AND place_clusters.id IS NULL
            """,
        ),
        "places_duplicate_display_name": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM (
                SELECT display_name
                FROM places
                WHERE display_name IS NOT NULL AND TRIM(display_name) != ''
                GROUP BY display_name
                HAVING COUNT(*) > 1
            )
            """,
        ),
        "event_places_total": _count(connection, "SELECT COUNT(*) FROM event_places"),
        "event_places_orphan_event_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM event_places
            LEFT JOIN events ON events.id = event_places.event_id
            WHERE events.id IS NULL
            """,
        ),
        "event_places_orphan_place_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM event_places
            LEFT JOIN places ON places.id = event_places.place_id
            WHERE places.id IS NULL
            """,
        ),
        "event_places_orphan_sample_ids": _sample_query(
            connection,
            """
            SELECT event_places.event_id || ':' || event_places.place_id AS id
            FROM event_places
            LEFT JOIN events ON events.id = event_places.event_id
            LEFT JOIN places ON places.id = event_places.place_id
            WHERE events.id IS NULL OR places.id IS NULL
            ORDER BY event_places.event_id ASC, event_places.place_id ASC
            LIMIT ?
            """,
        ),
        "event_places_invalid_confidence": _count(
            connection,
            "SELECT COUNT(*) FROM event_places WHERE confidence IS NULL OR confidence < 0 OR confidence > 1",
        ),
        "media_places_total": _count(connection, "SELECT COUNT(*) FROM media_places"),
        "media_places_orphan_media_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM media_places
            LEFT JOIN media_items ON media_items.id = media_places.media_id
            WHERE media_items.id IS NULL
            """,
        ),
        "media_places_orphan_place_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM media_places
            LEFT JOIN places ON places.id = media_places.place_id
            WHERE places.id IS NULL
            """,
        ),
        "media_places_orphan_sample_ids": _sample_query(
            connection,
            """
            SELECT media_places.media_id || ':' || media_places.place_id AS id
            FROM media_places
            LEFT JOIN media_items ON media_items.id = media_places.media_id
            LEFT JOIN places ON places.id = media_places.place_id
            WHERE media_items.id IS NULL OR places.id IS NULL
            ORDER BY media_places.media_id ASC, media_places.place_id ASC
            LIMIT ?
            """,
        ),
        "media_places_invalid_confidence": _count(
            connection,
            "SELECT COUNT(*) FROM media_places WHERE confidence IS NULL OR confidence < 0 OR confidence > 1",
        ),
    }


def _face_detection_checks(connection) -> dict[str, Any]:
    valid_statuses = ("success", "failed", "skipped", "engine_unavailable", "no_face_detected")
    valid_review_statuses = ("unreviewed", "accepted", "rejected", "bad_detection")
    valid_privacy_levels = ("private",)
    status_placeholders = ", ".join("?" for _ in valid_statuses)
    review_placeholders = ", ".join("?" for _ in valid_review_statuses)
    privacy_placeholders = ", ".join("?" for _ in valid_privacy_levels)
    missing_crop_rows = _rows(
        connection,
        """
        SELECT id, crop_path, thumbnail_path
        FROM face_detections
        WHERE status = 'success'
          AND (
            (crop_path IS NOT NULL AND TRIM(crop_path) != '')
            OR (thumbnail_path IS NOT NULL AND TRIM(thumbnail_path) != '')
          )
        """,
    )
    missing_crop_ids = []
    for row in missing_crop_rows:
        crop_path = row.get("crop_path")
        thumbnail_path = row.get("thumbnail_path")
        if (crop_path and not Path(str(crop_path)).expanduser().exists()) or (
            thumbnail_path and not Path(str(thumbnail_path)).expanduser().exists()
        ):
            missing_crop_ids.append(str(row.get("id")))
    return {
        "total": _count(connection, "SELECT COUNT(*) FROM face_detections"),
        "status_counts": _rows(
            connection,
            "SELECT status, COUNT(*) AS count FROM face_detections GROUP BY status ORDER BY count DESC, status ASC",
        ),
        "review_status_counts": _rows(
            connection,
            "SELECT review_status, COUNT(*) AS count FROM face_detections GROUP BY review_status ORDER BY count DESC, review_status ASC",
        ),
        "orphan_media_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM face_detections
            LEFT JOIN media_items ON media_items.id = face_detections.media_id
            WHERE media_items.id IS NULL
            """,
        ),
        "orphan_media_sample_ids": _sample_query(
            connection,
            """
            SELECT face_detections.media_id AS id
            FROM face_detections
            LEFT JOIN media_items ON media_items.id = face_detections.media_id
            WHERE media_items.id IS NULL
            ORDER BY face_detections.media_id ASC
            LIMIT ?
            """,
        ),
        "invalid_bbox": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM face_detections
            WHERE status = 'success'
              AND (
                bbox_x IS NULL OR bbox_y IS NULL OR bbox_w IS NULL OR bbox_h IS NULL
                OR bbox_x < 0 OR bbox_y < 0 OR bbox_w <= 0 OR bbox_h <= 0
                OR (image_width IS NOT NULL AND bbox_x + bbox_w > image_width + 1)
                OR (image_height IS NOT NULL AND bbox_y + bbox_h > image_height + 1)
              )
            """,
        ),
        "invalid_bbox_sample_ids": _sample_query(
            connection,
            """
            SELECT id
            FROM face_detections
            WHERE status = 'success'
              AND (
                bbox_x IS NULL OR bbox_y IS NULL OR bbox_w IS NULL OR bbox_h IS NULL
                OR bbox_x < 0 OR bbox_y < 0 OR bbox_w <= 0 OR bbox_h <= 0
                OR (image_width IS NOT NULL AND bbox_x + bbox_w > image_width + 1)
                OR (image_height IS NOT NULL AND bbox_y + bbox_h > image_height + 1)
              )
            ORDER BY id ASC
            LIMIT ?
            """,
        ),
        "missing_crop_files": len(missing_crop_ids),
        "missing_crop_sample_ids": missing_crop_ids[:SAMPLE_LIMIT],
        "invalid_review_status": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM face_detections
            WHERE review_status IS NULL OR review_status NOT IN ({review_placeholders})
            """,
            list(valid_review_statuses),
        ),
        "invalid_privacy_level": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM face_detections
            WHERE privacy_level IS NULL OR privacy_level NOT IN ({privacy_placeholders})
            """,
            list(valid_privacy_levels),
        ),
        "invalid_hidden_flag": _count(
            connection,
            "SELECT COUNT(*) FROM face_detections WHERE COALESCE(hidden, 0) NOT IN (0, 1)",
        ),
        "invalid_status": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM face_detections
            WHERE status IS NULL OR status NOT IN ({status_placeholders})
            """,
            list(valid_statuses),
        ),
    }


def _face_detection_run_checks(connection) -> dict[str, Any]:
    valid_statuses = ("planned", "running", "completed", "failed", "partial", "canceled")
    status_placeholders = ", ".join("?" for _ in valid_statuses)
    return {
        "total": _count(connection, "SELECT COUNT(*) FROM face_detection_runs"),
        "status_counts": _rows(
            connection,
            "SELECT status, COUNT(*) AS count FROM face_detection_runs GROUP BY status ORDER BY count DESC, status ASC",
        ),
        "failed_runs": _count(connection, "SELECT COUNT(*) FROM face_detection_runs WHERE status = 'failed'"),
        "stale_running_runs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM face_detection_runs
            WHERE status = 'running'
              AND started_at IS NOT NULL
              AND datetime(started_at) < datetime('now', '-12 hours')
            """,
        ),
        "invalid_status": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM face_detection_runs
            WHERE status IS NULL OR status NOT IN ({status_placeholders})
            """,
            list(valid_statuses),
        ),
    }


def _face_embedding_cluster_checks(connection) -> dict[str, Any]:
    embedding_statuses = ("success", "failed", "skipped", "engine_unavailable")
    embedding_formats = ("float32_numpy",)
    cluster_statuses = ("unreviewed", "accepted", "rejected", "merged", "split")
    cluster_review_statuses = ("unreviewed", "reviewed", "bad_cluster")
    privacy_levels = ("private",)
    embedding_status_placeholders = ", ".join("?" for _ in embedding_statuses)
    format_placeholders = ", ".join("?" for _ in embedding_formats)
    cluster_status_placeholders = ", ".join("?" for _ in cluster_statuses)
    cluster_review_placeholders = ", ".join("?" for _ in cluster_review_statuses)
    privacy_placeholders = ", ".join("?" for _ in privacy_levels)
    return {
        "face_embeddings_total": _count(connection, "SELECT COUNT(*) FROM face_embeddings"),
        "face_embedding_status_counts": _rows(
            connection,
            "SELECT status, COUNT(*) AS count FROM face_embeddings GROUP BY status ORDER BY count DESC, status ASC",
        ),
        "face_embeddings_orphan_face_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM face_embeddings
            LEFT JOIN face_detections ON face_detections.id = face_embeddings.face_id
            WHERE face_detections.id IS NULL
            """,
        ),
        "face_embeddings_orphan_sample_ids": _sample_query(
            connection,
            """
            SELECT face_embeddings.face_id AS id
            FROM face_embeddings
            LEFT JOIN face_detections ON face_detections.id = face_embeddings.face_id
            WHERE face_detections.id IS NULL
            ORDER BY face_embeddings.face_id ASC
            LIMIT ?
            """,
        ),
        "face_embeddings_invalid_dim": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM face_embeddings
            WHERE status = 'success'
              AND embedding_format = 'float32_numpy'
              AND (
                embedding_dim IS NULL OR embedding_dim <= 0
                OR embedding_blob IS NULL
                OR length(embedding_blob) != embedding_dim * 4
              )
            """,
        ),
        "face_embeddings_success_empty_blob": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM face_embeddings
            WHERE status = 'success'
              AND (embedding_blob IS NULL OR length(embedding_blob) = 0)
            """,
        ),
        "face_embeddings_unknown_format": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM face_embeddings
            WHERE embedding_format IS NULL OR embedding_format NOT IN ({format_placeholders})
            """,
            list(embedding_formats),
        ),
        "face_embeddings_invalid_status": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM face_embeddings
            WHERE status IS NULL OR status NOT IN ({embedding_status_placeholders})
            """,
            list(embedding_statuses),
        ),
        "face_clusters_total": _count(connection, "SELECT COUNT(*) FROM face_clusters"),
        "face_cluster_status_counts": _rows(
            connection,
            "SELECT status, COUNT(*) AS count FROM face_clusters GROUP BY status ORDER BY count DESC, status ASC",
        ),
        "face_clusters_invalid_representative_face": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM face_clusters
            LEFT JOIN face_detections ON face_detections.id = face_clusters.representative_face_id
            WHERE face_clusters.representative_face_id IS NOT NULL
              AND face_detections.id IS NULL
            """,
        ),
        "face_clusters_invalid_representative_sample_ids": _sample_query(
            connection,
            """
            SELECT face_clusters.representative_face_id AS id
            FROM face_clusters
            LEFT JOIN face_detections ON face_detections.id = face_clusters.representative_face_id
            WHERE face_clusters.representative_face_id IS NOT NULL
              AND face_detections.id IS NULL
            ORDER BY face_clusters.representative_face_id ASC
            LIMIT ?
            """,
        ),
        "face_clusters_invalid_status": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM face_clusters
            WHERE status IS NULL OR status NOT IN ({cluster_status_placeholders})
               OR review_status IS NULL OR review_status NOT IN ({cluster_review_placeholders})
            """,
            list(cluster_statuses) + list(cluster_review_statuses),
        ),
        "face_clusters_invalid_privacy_level": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM face_clusters
            WHERE privacy_level IS NULL OR privacy_level NOT IN ({privacy_placeholders})
            """,
            list(privacy_levels),
        ),
        "face_clusters_empty_cluster": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM face_clusters
            LEFT JOIN (
                SELECT cluster_id, COUNT(*) AS actual_count
                FROM face_cluster_members
                GROUP BY cluster_id
            ) counts ON counts.cluster_id = face_clusters.id
            WHERE COALESCE(counts.actual_count, 0) = 0
               OR COALESCE(face_clusters.face_count, 0) != COALESCE(counts.actual_count, 0)
            """,
        ),
        "face_clusters_singleton_count": _count(connection, "SELECT COUNT(*) FROM face_clusters WHERE face_count = 1"),
        "face_cluster_members_orphan_cluster_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM face_cluster_members
            LEFT JOIN face_clusters ON face_clusters.id = face_cluster_members.cluster_id
            WHERE face_clusters.id IS NULL
            """,
        ),
        "face_cluster_members_orphan_face_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM face_cluster_members
            LEFT JOIN face_detections ON face_detections.id = face_cluster_members.face_id
            WHERE face_detections.id IS NULL
            """,
        ),
        "face_cluster_members_orphan_sample_ids": _sample_query(
            connection,
            """
            SELECT face_cluster_members.cluster_id || ':' || face_cluster_members.face_id AS id
            FROM face_cluster_members
            LEFT JOIN face_clusters ON face_clusters.id = face_cluster_members.cluster_id
            LEFT JOIN face_detections ON face_detections.id = face_cluster_members.face_id
            WHERE face_clusters.id IS NULL OR face_detections.id IS NULL
            ORDER BY face_cluster_members.cluster_id ASC, face_cluster_members.face_id ASC
            LIMIT ?
            """,
        ),
        "face_cluster_members_duplicate_member": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM (
                SELECT cluster_id, face_id
                FROM face_cluster_members
                GROUP BY cluster_id, face_id
                HAVING COUNT(*) > 1
            )
            """,
        ),
        "face_cluster_members_invalid_distance": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM face_cluster_members
            WHERE distance_to_centroid IS NOT NULL
              AND (distance_to_centroid < 0 OR distance_to_centroid > 2)
            """,
        ),
    }


def _person_checks(connection) -> dict[str, Any]:
    privacy_levels = ("private", "public_alias", "public_hidden")
    alias_sources = ("manual", "line_speaker", "nickname")
    privacy_placeholders = ", ".join("?" for _ in privacy_levels)
    alias_source_placeholders = ", ".join("?" for _ in alias_sources)
    return {
        "persons_total": _count(connection, "SELECT COUNT(*) FROM persons"),
        "person_privacy_counts": _rows(
            connection,
            "SELECT privacy_level, COUNT(*) AS count FROM persons GROUP BY privacy_level ORDER BY count DESC, privacy_level ASC",
        ),
        "persons_invalid_privacy_level": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM persons
            WHERE privacy_level IS NULL OR privacy_level NOT IN ({privacy_placeholders})
            """,
            list(privacy_levels),
        ),
        "persons_invalid_privacy_flags": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM persons
            WHERE COALESCE(hidden, 0) NOT IN (0, 1)
               OR COALESCE(searchable, 1) NOT IN (0, 1)
               OR COALESCE(event_usable, 1) NOT IN (0, 1)
            """,
        ),
        "persons_hidden": _count(connection, "SELECT COUNT(*) FROM persons WHERE COALESCE(hidden, 0) = 1"),
        "persons_deleted": _count(connection, "SELECT COUNT(*) FROM persons WHERE deleted_at IS NOT NULL"),
        "persons_deleted_links_warning": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM persons
            WHERE deleted_at IS NOT NULL
              AND (
                EXISTS (SELECT 1 FROM event_people WHERE event_people.person_id = persons.id AND COALESCE(event_people.hidden, 0) = 0)
                OR EXISTS (SELECT 1 FROM media_people WHERE media_people.person_id = persons.id AND COALESCE(media_people.hidden, 0) = 0)
              )
            """,
        ),
        "persons_duplicate_display_name": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM (
                SELECT display_name
                FROM persons
                WHERE deleted_at IS NULL
                  AND COALESCE(hidden, 0) = 0
                  AND COALESCE(searchable, 1) = 1
                GROUP BY display_name
                HAVING COUNT(*) > 1
            )
            """,
        ),
        "persons_public_alias_missing_public_name": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM persons
            WHERE privacy_level = 'public_alias'
              AND (public_name IS NULL OR trim(public_name) = '')
            """,
        ),
        "person_face_clusters_total": _count(connection, "SELECT COUNT(*) FROM person_face_clusters"),
        "person_face_clusters_orphan_person_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM person_face_clusters
            LEFT JOIN persons ON persons.id = person_face_clusters.person_id
            WHERE persons.id IS NULL
            """,
        ),
        "person_face_clusters_orphan_cluster_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM person_face_clusters
            LEFT JOIN face_clusters ON face_clusters.id = person_face_clusters.face_cluster_id
            WHERE face_clusters.id IS NULL
            """,
        ),
        "person_face_clusters_orphan_sample_ids": _sample_query(
            connection,
            """
            SELECT person_face_clusters.person_id || ':' || person_face_clusters.face_cluster_id AS id
            FROM person_face_clusters
            LEFT JOIN persons ON persons.id = person_face_clusters.person_id
            LEFT JOIN face_clusters ON face_clusters.id = person_face_clusters.face_cluster_id
            WHERE persons.id IS NULL OR face_clusters.id IS NULL
            ORDER BY person_face_clusters.person_id ASC, person_face_clusters.face_cluster_id ASC
            LIMIT ?
            """,
        ),
        "person_face_clusters_duplicate_links": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM (
                SELECT person_id, face_cluster_id
                FROM person_face_clusters
                GROUP BY person_id, face_cluster_id
                HAVING COUNT(*) > 1
            )
            """,
        ),
        "person_face_clusters_rejected_cluster_links": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM person_face_clusters
            JOIN face_clusters ON face_clusters.id = person_face_clusters.face_cluster_id
            WHERE face_clusters.status = 'rejected' OR face_clusters.review_status = 'bad_cluster'
            """,
        ),
        "person_aliases_total": _count(connection, "SELECT COUNT(*) FROM person_aliases"),
        "person_aliases_orphan_person_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM person_aliases
            LEFT JOIN persons ON persons.id = person_aliases.person_id
            WHERE persons.id IS NULL
            """,
        ),
        "person_aliases_orphan_sample_ids": _sample_query(
            connection,
            """
            SELECT person_aliases.id AS id
            FROM person_aliases
            LEFT JOIN persons ON persons.id = person_aliases.person_id
            WHERE persons.id IS NULL
            ORDER BY person_aliases.id ASC
            LIMIT ?
            """,
        ),
        "person_aliases_duplicate_aliases": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM (
                SELECT person_id, alias
                FROM person_aliases
                GROUP BY person_id, alias
                HAVING COUNT(*) > 1
            )
            """,
        ),
        "person_aliases_invalid_source": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM person_aliases
            WHERE source IS NULL OR source NOT IN ({alias_source_placeholders})
            """,
            list(alias_sources),
        ),
    }


def _line_person_link_checks(connection) -> dict[str, Any]:
    mention_types = ("speaker", "mentioned_in_text")
    mention_placeholders = ", ".join("?" for _ in mention_types)
    return {
        "line_speaker_links_total": _count(connection, "SELECT COUNT(*) FROM line_speaker_links"),
        "line_speaker_links_orphan_person_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM line_speaker_links
            LEFT JOIN persons ON persons.id = line_speaker_links.person_id
            WHERE persons.id IS NULL
            """,
        ),
        "line_speaker_links_orphan_sample_ids": _sample_query(
            connection,
            """
            SELECT line_speaker_links.id AS id
            FROM line_speaker_links
            LEFT JOIN persons ON persons.id = line_speaker_links.person_id
            WHERE persons.id IS NULL
            ORDER BY line_speaker_links.id ASC
            LIMIT ?
            """,
        ),
        "line_speaker_links_empty_speaker_name": _count(
            connection,
            "SELECT COUNT(*) FROM line_speaker_links WHERE speaker_name IS NULL OR trim(speaker_name) = ''",
        ),
        "line_speaker_links_empty_chat_id": _count(
            connection,
            "SELECT COUNT(*) FROM line_speaker_links WHERE chat_id IS NULL OR trim(chat_id) = ''",
        ),
        "line_speaker_links_duplicate_links": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM (
                SELECT chat_id, speaker_name, person_id
                FROM line_speaker_links
                GROUP BY chat_id, speaker_name, person_id
                HAVING COUNT(*) > 1
            )
            """,
        ),
        "line_speaker_links_invalid_confidence": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM line_speaker_links
            WHERE confidence IS NULL OR confidence < 0 OR confidence > 1
            """,
        ),
        "person_line_mentions_total": _count(connection, "SELECT COUNT(*) FROM person_line_mentions"),
        "person_line_mentions_orphan_person_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM person_line_mentions
            LEFT JOIN persons ON persons.id = person_line_mentions.person_id
            WHERE persons.id IS NULL
            """,
        ),
        "person_line_mentions_orphan_message_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM person_line_mentions
            LEFT JOIN line_messages ON line_messages.id = person_line_mentions.message_id
            WHERE line_messages.id IS NULL
            """,
        ),
        "person_line_mentions_orphan_sample_ids": _sample_query(
            connection,
            """
            SELECT person_line_mentions.id AS id
            FROM person_line_mentions
            LEFT JOIN persons ON persons.id = person_line_mentions.person_id
            LEFT JOIN line_messages ON line_messages.id = person_line_mentions.message_id
            WHERE persons.id IS NULL OR line_messages.id IS NULL
            ORDER BY person_line_mentions.id ASC
            LIMIT ?
            """,
        ),
        "person_line_mentions_invalid_mention_type": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM person_line_mentions
            WHERE mention_type IS NULL OR mention_type NOT IN ({mention_placeholders})
            """,
            list(mention_types),
        ),
    }


def _person_event_media_checks(connection) -> dict[str, Any]:
    media_sources = ("face_cluster", "manual")
    event_sources = ("face", "line_speaker", "line_mention", "manual", "combined")
    media_placeholders = ", ".join("?" for _ in media_sources)
    event_placeholders = ", ".join("?" for _ in event_sources)
    return {
        "media_people_total": _count(connection, "SELECT COUNT(*) FROM media_people"),
        "media_people_source_counts": _rows(
            connection,
            "SELECT source, COUNT(*) AS count FROM media_people GROUP BY source ORDER BY count DESC, source ASC",
        ),
        "media_people_orphan_media_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM media_people
            LEFT JOIN media_items ON media_items.id = media_people.media_id
            WHERE media_items.id IS NULL
            """,
        ),
        "media_people_orphan_person_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM media_people
            LEFT JOIN persons ON persons.id = media_people.person_id
            WHERE persons.id IS NULL
            """,
        ),
        "media_people_orphan_face_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM media_people
            LEFT JOIN face_detections ON face_detections.id = media_people.face_id
            WHERE media_people.face_id IS NOT NULL
              AND face_detections.id IS NULL
            """,
        ),
        "media_people_orphan_cluster_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM media_people
            LEFT JOIN face_clusters ON face_clusters.id = media_people.face_cluster_id
            WHERE media_people.face_cluster_id IS NOT NULL
              AND face_clusters.id IS NULL
            """,
        ),
        "media_people_orphan_sample_ids": _sample_query(
            connection,
            """
            SELECT media_people.media_id || ':' || media_people.person_id || ':' || media_people.source AS id
            FROM media_people
            LEFT JOIN media_items ON media_items.id = media_people.media_id
            LEFT JOIN persons ON persons.id = media_people.person_id
            LEFT JOIN face_detections ON face_detections.id = media_people.face_id
            LEFT JOIN face_clusters ON face_clusters.id = media_people.face_cluster_id
            WHERE media_items.id IS NULL
               OR persons.id IS NULL
               OR (media_people.face_id IS NOT NULL AND face_detections.id IS NULL)
               OR (media_people.face_cluster_id IS NOT NULL AND face_clusters.id IS NULL)
            ORDER BY media_people.media_id ASC, media_people.person_id ASC
            LIMIT ?
            """,
        ),
        "media_people_invalid_source": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM media_people
            WHERE source IS NULL OR source NOT IN ({media_placeholders})
            """,
            list(media_sources),
        ),
        "media_people_invalid_confidence": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM media_people
            WHERE confidence IS NULL OR confidence < 0 OR confidence > 1
            """,
        ),
        "media_people_invalid_hidden_flag": _count(
            connection,
            "SELECT COUNT(*) FROM media_people WHERE COALESCE(hidden, 0) NOT IN (0, 1)",
        ),
        "media_people_hidden_deleted_person_links": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM media_people
            JOIN persons ON persons.id = media_people.person_id
            WHERE COALESCE(media_people.hidden, 0) = 0
              AND (COALESCE(persons.hidden, 0) = 1 OR persons.deleted_at IS NOT NULL)
            """,
        ),
        "media_people_unverified_cluster_links": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM media_people
            LEFT JOIN face_clusters ON face_clusters.id = media_people.face_cluster_id
            LEFT JOIN person_face_clusters
              ON person_face_clusters.person_id = media_people.person_id
             AND person_face_clusters.face_cluster_id = media_people.face_cluster_id
            LEFT JOIN persons ON persons.id = media_people.person_id
            WHERE media_people.source = 'face_cluster'
              AND (
                COALESCE(media_people.verified_by_user, 0) != 1
                OR COALESCE(person_face_clusters.verified_by_user, 0) != 1
                OR COALESCE(persons.manual_verified, 0) != 1
                OR face_clusters.status != 'accepted'
                OR face_clusters.review_status = 'bad_cluster'
              )
            """,
        ),
        "media_people_rejected_detection_links": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM media_people
            JOIN face_detections ON face_detections.id = media_people.face_id
            WHERE media_people.source = 'face_cluster'
              AND (
                face_detections.status != 'success'
                OR face_detections.review_status IN ('rejected', 'bad_detection')
              )
            """,
        ),
        "event_people_total": _count(connection, "SELECT COUNT(*) FROM event_people"),
        "event_people_source_counts": _rows(
            connection,
            "SELECT source, COUNT(*) AS count FROM event_people GROUP BY source ORDER BY count DESC, source ASC",
        ),
        "event_people_orphan_event_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM event_people
            LEFT JOIN events ON events.id = event_people.event_id
            WHERE events.id IS NULL
            """,
        ),
        "event_people_orphan_person_refs": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM event_people
            LEFT JOIN persons ON persons.id = event_people.person_id
            WHERE persons.id IS NULL
            """,
        ),
        "event_people_orphan_sample_ids": _sample_query(
            connection,
            """
            SELECT event_people.event_id || ':' || event_people.person_id || ':' || event_people.source AS id
            FROM event_people
            LEFT JOIN events ON events.id = event_people.event_id
            LEFT JOIN persons ON persons.id = event_people.person_id
            WHERE events.id IS NULL OR persons.id IS NULL
            ORDER BY event_people.event_id ASC, event_people.person_id ASC
            LIMIT ?
            """,
        ),
        "event_people_invalid_source": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM event_people
            WHERE source IS NULL OR source NOT IN ({event_placeholders})
            """,
            list(event_sources),
        ),
        "event_people_invalid_confidence": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM event_people
            WHERE confidence IS NULL OR confidence < 0 OR confidence > 1
               OR evidence_count < 0 OR media_count < 0 OR line_count < 0
            """,
        ),
        "event_people_invalid_hidden_flag": _count(
            connection,
            "SELECT COUNT(*) FROM event_people WHERE COALESCE(hidden, 0) NOT IN (0, 1)",
        ),
        "event_people_hidden_deleted_person_links": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM event_people
            JOIN persons ON persons.id = event_people.person_id
            WHERE COALESCE(event_people.hidden, 0) = 0
              AND (COALESCE(persons.hidden, 0) = 1 OR persons.deleted_at IS NOT NULL)
            """,
        ),
        "event_people_duplicate_links": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM (
                SELECT event_id, person_id, source
                FROM event_people
                GROUP BY event_id, person_id, source
                HAVING COUNT(*) > 1
            )
            """,
        ),
        "event_people_unverified_person_links": _count(
            connection,
            """
            SELECT COUNT(*)
            FROM event_people
            JOIN persons ON persons.id = event_people.person_id
            WHERE COALESCE(persons.manual_verified, 0) != 1
            """,
        ),
    }


def _privacy_action_checks(connection) -> dict[str, Any]:
    action_types = (
        "hide_person",
        "unhide_person",
        "delete_person",
        "detach_person",
        "delete_face_embedding",
        "delete_face_crop",
        "hide_place",
        "hide_media",
        "export_person",
        "privacy_audit",
    )
    target_types = ("person", "face", "face_cluster", "place", "media", "event", "portfolio", "database")
    modes = ("dry_run", "executed")
    action_placeholders = ", ".join("?" for _ in action_types)
    target_placeholders = ", ".join("?" for _ in target_types)
    mode_placeholders = ", ".join("?" for _ in modes)
    return {
        "privacy_actions_total": _count(connection, "SELECT COUNT(*) FROM privacy_actions"),
        "privacy_action_type_counts": _rows(
            connection,
            "SELECT action_type, COUNT(*) AS count FROM privacy_actions GROUP BY action_type ORDER BY count DESC, action_type ASC",
        ),
        "privacy_actions_invalid_action_type": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM privacy_actions
            WHERE action_type IS NULL OR action_type NOT IN ({action_placeholders})
            """,
            list(action_types),
        ),
        "privacy_actions_invalid_target_type": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM privacy_actions
            WHERE target_type IS NULL OR target_type NOT IN ({target_placeholders})
            """,
            list(target_types),
        ),
        "privacy_actions_invalid_mode": _count_params(
            connection,
            f"""
            SELECT COUNT(*)
            FROM privacy_actions
            WHERE mode IS NULL OR mode NOT IN ({mode_placeholders})
            """,
            list(modes),
        ),
    }


def _strict_summary(report: dict[str, Any], *, fail_on_missing_files: bool = False) -> dict[str, Any]:
    media = report["media_items"]
    media_ocr = report["media_ocr"]
    media_vlm = report["media_vlm"]
    media_vlm_overrides = report["media_vlm_overrides"]
    media_embeddings = report["media_embeddings"]
    line = report["line_messages"]
    calls = report["line_call_events"]
    events = report["events"]
    evidence = report["event_evidence"]
    analysis_jobs = report["analysis_jobs"]
    location_places = report["location_places"]
    face_detections = report["face_detections"]
    face_detection_runs = report["face_detection_runs"]
    face_embedding_clusters = report["face_embedding_clusters"]
    persons = report["persons"]
    line_person_links = report["line_person_links"]
    person_event_media = report["person_event_media"]
    privacy_actions = report["privacy_actions"]
    issues: list[str] = []

    if line["duplicate_id_groups"]:
        issues.append(f"duplicate line_messages id groups: {line['duplicate_id_groups']}")
    if calls["orphan_message_refs"]:
        issues.append(f"line_call_events orphan message refs: {calls['orphan_message_refs']}")
    if calls["negative_duration_sec"]:
        issues.append(f"line_call_events negative duration_sec: {calls['negative_duration_sec']}")
    if media["duplicate_file_path_groups"]:
        issues.append(f"duplicate file_path groups: {media['duplicate_file_path_groups']}")
    if fail_on_missing_files and media["missing_file_count"]:
        issues.append(f"missing original media files: {media['missing_file_count']}")
    if media["duplicate_file_hash_groups"]:
        issues.append(f"duplicate file_hash groups: {media['duplicate_file_hash_groups']}")
    if media["file_hash_null"] >= _file_hash_null_threshold(media["total"]):
        issues.append(f"file_hash NULL/empty count is high: {media['file_hash_null']}")
    if media_ocr["orphan_media_refs"]:
        issues.append(f"media_ocr orphan media refs: {media_ocr['orphan_media_refs']}")
    if media_ocr["invalid_status_count"]:
        issues.append(f"media_ocr invalid status rows: {media_ocr['invalid_status_count']}")
    if media_ocr["success_analyzed_at_null"]:
        issues.append(f"media_ocr success analyzed_at NULL: {media_ocr['success_analyzed_at_null']}")
    if media_vlm["orphan_media_refs"]:
        issues.append(f"media_vlm orphan media refs: {media_vlm['orphan_media_refs']}")
    if media_vlm["invalid_status_count"]:
        issues.append(f"media_vlm invalid status rows: {media_vlm['invalid_status_count']}")
    if media_vlm["success_caption_empty"]:
        issues.append(f"media_vlm success caption empty: {media_vlm['success_caption_empty']}")
    if media_vlm_overrides["orphan_media_refs"]:
        issues.append(f"media_vlm_overrides orphan media refs: {media_vlm_overrides['orphan_media_refs']}")
    if media_vlm_overrides["unknown_status_count"]:
        issues.append(f"media_vlm_overrides unknown review_status rows: {media_vlm_overrides['unknown_status_count']}")
    if media_vlm_overrides["invalid_json_count"]:
        issues.append(f"media_vlm_overrides invalid JSON rows: {media_vlm_overrides['invalid_json_count']}")
    if media_embeddings["orphan_media_refs"]:
        issues.append(f"media_embeddings orphan media refs: {media_embeddings['orphan_media_refs']}")
    if media_embeddings["invalid_status_count"]:
        issues.append(f"media_embeddings invalid status rows: {media_embeddings['invalid_status_count']}")
    if media_embeddings["unknown_format_count"]:
        issues.append(f"media_embeddings unknown format rows: {media_embeddings['unknown_format_count']}")
    if media_embeddings["success_empty_embedding"]:
        issues.append(f"media_embeddings success empty embedding: {media_embeddings['success_empty_embedding']}")
    if media_embeddings["dimension_mismatch_count"]:
        issues.append(f"media_embeddings dimension mismatch: {media_embeddings['dimension_mismatch_count']}")
    if events["date_null"]:
        issues.append(f"events date NULL/empty: {events['date_null']}")
    if evidence["orphan_event_refs"]:
        issues.append(f"orphan event_evidence event refs: {evidence['orphan_event_refs']}")
    if evidence["missing_photo_refs"]:
        issues.append(f"photo evidence missing media_items refs: {evidence['missing_photo_refs']}")
    if evidence["missing_line_refs"]:
        issues.append(f"line evidence missing line_messages refs: {evidence['missing_line_refs']}")
    if evidence["invalid_vlm_refs"]:
        issues.append(f"VLM evidence invalid media_vlm refs: {evidence['invalid_vlm_refs']}")
    if analysis_jobs["orphan_job_items"]:
        issues.append(f"analysis_job_items orphan job refs: {analysis_jobs['orphan_job_items']}")
    if analysis_jobs["invalid_job_status_count"]:
        issues.append(f"analysis_jobs invalid status rows: {analysis_jobs['invalid_job_status_count']}")
    if analysis_jobs["invalid_item_status_count"]:
        issues.append(f"analysis_job_items invalid status rows: {analysis_jobs['invalid_item_status_count']}")
    if analysis_jobs["item_count_mismatch"]:
        issues.append(f"analysis_jobs item count mismatch: {analysis_jobs['item_count_mismatch']}")
    if location_places["location_points_invalid_lat_lon"]:
        issues.append(f"location_points invalid lat/lon rows: {location_places['location_points_invalid_lat_lon']}")
    if location_places["location_points_orphan_media_refs"]:
        issues.append(f"location_points orphan media refs: {location_places['location_points_orphan_media_refs']}")
    if location_places["location_points_orphan_event_refs"]:
        issues.append(f"location_points orphan event refs: {location_places['location_points_orphan_event_refs']}")
    if location_places["location_points_invalid_privacy_level"]:
        issues.append(f"location_points invalid privacy_level rows: {location_places['location_points_invalid_privacy_level']}")
    if location_places["place_clusters_invalid_centroid"]:
        issues.append(f"place_clusters invalid centroid rows: {location_places['place_clusters_invalid_centroid']}")
    if location_places["place_clusters_invalid_radius"]:
        issues.append(f"place_clusters invalid radius rows: {location_places['place_clusters_invalid_radius']}")
    if location_places["place_clusters_invalid_status"]:
        issues.append(f"place_clusters invalid status rows: {location_places['place_clusters_invalid_status']}")
    if location_places["places_invalid_privacy_level"]:
        issues.append(f"places invalid privacy_level rows: {location_places['places_invalid_privacy_level']}")
    if location_places["places_invalid_privacy_flags"]:
        issues.append(f"places invalid privacy flag rows: {location_places['places_invalid_privacy_flags']}")
    if location_places["places_orphan_cluster_refs"]:
        issues.append(f"places orphan cluster refs: {location_places['places_orphan_cluster_refs']}")
    if location_places["event_places_orphan_event_refs"]:
        issues.append(f"event_places orphan event refs: {location_places['event_places_orphan_event_refs']}")
    if location_places["event_places_orphan_place_refs"]:
        issues.append(f"event_places orphan place refs: {location_places['event_places_orphan_place_refs']}")
    if location_places["event_places_invalid_confidence"]:
        issues.append(f"event_places invalid confidence rows: {location_places['event_places_invalid_confidence']}")
    if location_places["media_places_orphan_media_refs"]:
        issues.append(f"media_places orphan media refs: {location_places['media_places_orphan_media_refs']}")
    if location_places["media_places_orphan_place_refs"]:
        issues.append(f"media_places orphan place refs: {location_places['media_places_orphan_place_refs']}")
    if location_places["media_places_invalid_confidence"]:
        issues.append(f"media_places invalid confidence rows: {location_places['media_places_invalid_confidence']}")
    if face_detections["orphan_media_refs"]:
        issues.append(f"face_detections orphan media refs: {face_detections['orphan_media_refs']}")
    if face_detections["invalid_bbox"]:
        issues.append(f"face_detections invalid bbox rows: {face_detections['invalid_bbox']}")
    if face_detections["invalid_status"]:
        issues.append(f"face_detections invalid status rows: {face_detections['invalid_status']}")
    if face_detections["invalid_review_status"]:
        issues.append(f"face_detections invalid review_status rows: {face_detections['invalid_review_status']}")
    if face_detections["invalid_privacy_level"]:
        issues.append(f"face_detections invalid privacy_level rows: {face_detections['invalid_privacy_level']}")
    if face_detections["invalid_hidden_flag"]:
        issues.append(f"face_detections invalid hidden flag rows: {face_detections['invalid_hidden_flag']}")
    if face_detection_runs["invalid_status"]:
        issues.append(f"face_detection_runs invalid status rows: {face_detection_runs['invalid_status']}")
    if face_embedding_clusters["face_embeddings_orphan_face_refs"]:
        issues.append(f"face_embeddings orphan face refs: {face_embedding_clusters['face_embeddings_orphan_face_refs']}")
    if face_embedding_clusters["face_embeddings_invalid_status"]:
        issues.append(f"face_embeddings invalid status rows: {face_embedding_clusters['face_embeddings_invalid_status']}")
    if face_embedding_clusters["face_embeddings_success_empty_blob"]:
        issues.append(f"face_embeddings success empty blob rows: {face_embedding_clusters['face_embeddings_success_empty_blob']}")
    if face_embedding_clusters["face_embeddings_invalid_dim"]:
        issues.append(f"face_embeddings invalid dim rows: {face_embedding_clusters['face_embeddings_invalid_dim']}")
    if face_embedding_clusters["face_embeddings_unknown_format"]:
        issues.append(f"face_embeddings unknown format rows: {face_embedding_clusters['face_embeddings_unknown_format']}")
    if face_embedding_clusters["face_clusters_invalid_representative_face"]:
        issues.append(
            "face_clusters invalid representative_face_id rows: "
            f"{face_embedding_clusters['face_clusters_invalid_representative_face']}"
        )
    if face_embedding_clusters["face_clusters_invalid_status"]:
        issues.append(f"face_clusters invalid status rows: {face_embedding_clusters['face_clusters_invalid_status']}")
    if face_embedding_clusters["face_clusters_invalid_privacy_level"]:
        issues.append(
            f"face_clusters invalid privacy_level rows: {face_embedding_clusters['face_clusters_invalid_privacy_level']}"
        )
    if face_embedding_clusters["face_clusters_empty_cluster"]:
        issues.append(f"face_clusters empty/mismatched rows: {face_embedding_clusters['face_clusters_empty_cluster']}")
    if face_embedding_clusters["face_cluster_members_orphan_cluster_refs"]:
        issues.append(
            f"face_cluster_members orphan cluster refs: {face_embedding_clusters['face_cluster_members_orphan_cluster_refs']}"
        )
    if face_embedding_clusters["face_cluster_members_orphan_face_refs"]:
        issues.append(
            f"face_cluster_members orphan face refs: {face_embedding_clusters['face_cluster_members_orphan_face_refs']}"
        )
    if face_embedding_clusters["face_cluster_members_invalid_distance"]:
        issues.append(
            f"face_cluster_members invalid distance rows: {face_embedding_clusters['face_cluster_members_invalid_distance']}"
        )
    if persons["persons_invalid_privacy_level"]:
        issues.append(f"persons invalid privacy_level rows: {persons['persons_invalid_privacy_level']}")
    if persons["persons_invalid_privacy_flags"]:
        issues.append(f"persons invalid privacy flag rows: {persons['persons_invalid_privacy_flags']}")
    if persons["person_face_clusters_orphan_person_refs"]:
        issues.append(
            f"person_face_clusters orphan person refs: {persons['person_face_clusters_orphan_person_refs']}"
        )
    if persons["person_face_clusters_orphan_cluster_refs"]:
        issues.append(
            f"person_face_clusters orphan cluster refs: {persons['person_face_clusters_orphan_cluster_refs']}"
        )
    if persons["person_aliases_orphan_person_refs"]:
        issues.append(f"person_aliases orphan person refs: {persons['person_aliases_orphan_person_refs']}")
    if persons["person_aliases_invalid_source"]:
        issues.append(f"person_aliases invalid source rows: {persons['person_aliases_invalid_source']}")
    if line_person_links["line_speaker_links_orphan_person_refs"]:
        issues.append(
            f"line_speaker_links orphan person refs: {line_person_links['line_speaker_links_orphan_person_refs']}"
        )
    if line_person_links["line_speaker_links_empty_speaker_name"]:
        issues.append(
            f"line_speaker_links empty speaker_name rows: {line_person_links['line_speaker_links_empty_speaker_name']}"
        )
    if line_person_links["line_speaker_links_empty_chat_id"]:
        issues.append(f"line_speaker_links empty chat_id rows: {line_person_links['line_speaker_links_empty_chat_id']}")
    if line_person_links["line_speaker_links_invalid_confidence"]:
        issues.append(
            f"line_speaker_links invalid confidence rows: {line_person_links['line_speaker_links_invalid_confidence']}"
        )
    if line_person_links["person_line_mentions_orphan_person_refs"]:
        issues.append(
            f"person_line_mentions orphan person refs: {line_person_links['person_line_mentions_orphan_person_refs']}"
        )
    if line_person_links["person_line_mentions_orphan_message_refs"]:
        issues.append(
            f"person_line_mentions orphan message refs: {line_person_links['person_line_mentions_orphan_message_refs']}"
        )
    if line_person_links["person_line_mentions_invalid_mention_type"]:
        issues.append(
            f"person_line_mentions invalid mention_type rows: {line_person_links['person_line_mentions_invalid_mention_type']}"
        )
    if person_event_media["media_people_orphan_media_refs"]:
        issues.append(f"media_people orphan media refs: {person_event_media['media_people_orphan_media_refs']}")
    if person_event_media["media_people_orphan_person_refs"]:
        issues.append(f"media_people orphan person refs: {person_event_media['media_people_orphan_person_refs']}")
    if person_event_media["media_people_orphan_face_refs"]:
        issues.append(f"media_people orphan face refs: {person_event_media['media_people_orphan_face_refs']}")
    if person_event_media["media_people_orphan_cluster_refs"]:
        issues.append(f"media_people orphan cluster refs: {person_event_media['media_people_orphan_cluster_refs']}")
    if person_event_media["media_people_invalid_source"]:
        issues.append(f"media_people invalid source rows: {person_event_media['media_people_invalid_source']}")
    if person_event_media["media_people_invalid_confidence"]:
        issues.append(f"media_people invalid confidence rows: {person_event_media['media_people_invalid_confidence']}")
    if person_event_media["media_people_invalid_hidden_flag"]:
        issues.append(f"media_people invalid hidden flag rows: {person_event_media['media_people_invalid_hidden_flag']}")
    if person_event_media["media_people_unverified_cluster_links"]:
        issues.append(
            f"media_people unverified/rejected cluster links: {person_event_media['media_people_unverified_cluster_links']}"
        )
    if person_event_media["media_people_rejected_detection_links"]:
        issues.append(
            f"media_people rejected/bad detection links: {person_event_media['media_people_rejected_detection_links']}"
        )
    if person_event_media["event_people_orphan_event_refs"]:
        issues.append(f"event_people orphan event refs: {person_event_media['event_people_orphan_event_refs']}")
    if person_event_media["event_people_orphan_person_refs"]:
        issues.append(f"event_people orphan person refs: {person_event_media['event_people_orphan_person_refs']}")
    if person_event_media["event_people_invalid_source"]:
        issues.append(f"event_people invalid source rows: {person_event_media['event_people_invalid_source']}")
    if person_event_media["event_people_invalid_confidence"]:
        issues.append(f"event_people invalid confidence rows: {person_event_media['event_people_invalid_confidence']}")
    if person_event_media["event_people_invalid_hidden_flag"]:
        issues.append(f"event_people invalid hidden flag rows: {person_event_media['event_people_invalid_hidden_flag']}")
    if privacy_actions["privacy_actions_invalid_action_type"]:
        issues.append(f"privacy_actions invalid action_type rows: {privacy_actions['privacy_actions_invalid_action_type']}")
    if privacy_actions["privacy_actions_invalid_target_type"]:
        issues.append(f"privacy_actions invalid target_type rows: {privacy_actions['privacy_actions_invalid_target_type']}")
    if privacy_actions["privacy_actions_invalid_mode"]:
        issues.append(f"privacy_actions invalid mode rows: {privacy_actions['privacy_actions_invalid_mode']}")

    return {
        "ok": not issues,
        "issues": issues,
        "file_hash_null_threshold": _file_hash_null_threshold(media["total"]),
    }


def _file_hash_null_threshold(total: int) -> int:
    if total <= 0:
        return 1
    return max(10, int(total * 0.01))


def _duplicate_value_samples(
    connection,
    table_name: str,
    column_name: str,
    *,
    include_null: bool = False,
) -> list[dict[str, Any]]:
    null_clause = "" if include_null else f"WHERE {column_name} IS NOT NULL AND TRIM({column_name}) != ''"
    rows = connection.execute(
        f"""
        SELECT {column_name} AS value, COUNT(*) AS count
        FROM {table_name}
        {null_clause}
        GROUP BY {column_name}
        HAVING COUNT(*) > 1
        ORDER BY count DESC, value ASC
        LIMIT ?
        """,
        (SAMPLE_LIMIT,),
    ).fetchall()
    samples: list[dict[str, Any]] = []
    for row in rows:
        id_rows = connection.execute(
            f"""
            SELECT id
            FROM {table_name}
            WHERE {column_name} = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (row["value"], SAMPLE_LIMIT),
        ).fetchall()
        samples.append(
            {
                "count": int(row["count"]),
                "ids": [str(id_row["id"]) for id_row in id_rows],
            }
        )
    return samples


def _flatten_sample_ids(groups: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for group in groups:
        for item_id in group["ids"]:
            _append_sample(ids, item_id)
    return ids


def _missing_path_count(rows: list[Any], column_name: str) -> int:
    count = 0
    for row in rows:
        raw_path = row[column_name]
        if raw_path and not Path(str(raw_path)).expanduser().exists():
            count += 1
    return count


def _sample_query(connection, query: str) -> list[str]:
    return [str(row["id"]) for row in connection.execute(query, (SAMPLE_LIMIT,)).fetchall()]


def _sample_query_params(connection, query: str, params: list[Any]) -> list[str]:
    return [str(row["id"]) for row in connection.execute(query, [*params, SAMPLE_LIMIT]).fetchall()]


def _count(connection, query: str) -> int:
    row = connection.execute(query).fetchone()
    return int(row[0] or 0)


def _count_params(connection, query: str, params: list[Any]) -> int:
    row = connection.execute(query, params).fetchone()
    return int(row[0] or 0)


def _rows(connection, query: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(query, params or []).fetchall()]


def _append_sample(samples: list[str], value: Any) -> None:
    if len(samples) < SAMPLE_LIMIT:
        samples.append(str(value))


def _sample_lines(label: str, samples: list[str]) -> list[str]:
    if not samples:
        return [f"- {label}: none"]
    return [f"- {label}: {', '.join(samples)}"]


def _count_rows_lines(rows: list[dict[str, Any]], key: str) -> list[str]:
    if not rows:
        return ["  - none"]
    return [f"  - {row.get(key)}: {row.get('count')}" for row in rows]


def _is_valid_json(raw: Any) -> bool:
    try:
        json.loads(str(raw))
    except json.JSONDecodeError:
        return False
    return True


def _format_float(value: Any) -> str:
    if value is None:
        return "なし"
    return f"{float(value):.3f}"
