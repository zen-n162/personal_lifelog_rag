"""UI helpers for private face detection review.

These helpers expose detection rows only for local review. They do not provide
identity labels and never surface face crops when public/private display is off.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from personal_lifelog_rag.core.privacy import redact_text
from personal_lifelog_rag.db.repository import connect
from personal_lifelog_rag.db.schema import initialize_schema
from personal_lifelog_rag.faces.embedding_service import list_face_clusters, update_face_cluster_status
from personal_lifelog_rag.faces.face_service import ensure_face_thumbnail, list_face_detections, update_face_review_status
from personal_lifelog_rag.faces.person_service import (
    add_person_alias,
    create_person,
    get_person,
    link_person_face_cluster,
    list_persons,
    public_person_name,
    unlink_person_face_cluster,
    update_person,
)
from personal_lifelog_rag.line.person_links import (
    link_line_speaker_to_person,
    list_line_speakers,
    unlink_line_speaker_from_person,
)


def face_review_queue_for_ui(
    repository,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    status: str | None = None,
    review_status: str | None = "unreviewed",
    limit: int = 50,
    show_private_crops: bool = True,
) -> dict[str, Any]:
    rows = list_face_detections(
        repository,
        date_from=date_from,
        date_to=date_to,
        status=status or None,
        review_status=review_status or None,
        limit=limit,
    )
    table = [_row_for_table(row) for row in rows]
    gallery, gallery_face_ids = _face_detection_gallery_items(
        repository,
        rows,
        show_private_crops=show_private_crops,
    )
    return {
        "rows": rows,
        "table": table,
        "face_ids": [row[0] for row in table],
        "gallery": gallery,
        "gallery_face_ids": gallery_face_ids,
    }


def adjacent_id(ids: list[str] | None, current_id: str | None, *, direction: int) -> str | None:
    """Return the previous/next id in a loaded UI queue, wrapping around."""
    item_ids = [str(item_id) for item_id in (ids or []) if item_id]
    if not item_ids:
        return None
    if current_id not in item_ids:
        return item_ids[0] if direction >= 0 else item_ids[-1]
    index = item_ids.index(str(current_id))
    return item_ids[(index + direction) % len(item_ids)]


def next_face_id(face_ids: list[str] | None, current_face_id: str | None) -> str | None:
    """Return the next face id in the currently loaded review queue."""
    return adjacent_id(face_ids, current_face_id, direction=1)


def previous_face_id(face_ids: list[str] | None, current_face_id: str | None) -> str | None:
    """Return the previous face id in the currently loaded review queue."""
    return adjacent_id(face_ids, current_face_id, direction=-1)


def person_form_for_ui(repository, person_id: str | None) -> dict[str, str]:
    if not person_id:
        return {"display_name": "", "public_name": "", "privacy_level": "private"}
    try:
        row = get_person(repository, person_id=person_id)
    except Exception:
        row = None
    if not row:
        return {"display_name": "", "public_name": "", "privacy_level": "private"}
    return {
        "display_name": str(row.get("display_name") or ""),
        "public_name": str(row.get("public_name") or ""),
        "privacy_level": str(row.get("privacy_level") or "private"),
    }


def next_person_id(person_ids: list[str] | None, current_person_id: str | None) -> str | None:
    return adjacent_id(person_ids, current_person_id, direction=1)


def previous_person_id(person_ids: list[str] | None, current_person_id: str | None) -> str | None:
    return adjacent_id(person_ids, current_person_id, direction=-1)


def next_cluster_id(cluster_ids: list[str] | None, current_cluster_id: str | None) -> str | None:
    return adjacent_id(cluster_ids, current_cluster_id, direction=1)


def previous_cluster_id(cluster_ids: list[str] | None, current_cluster_id: str | None) -> str | None:
    return adjacent_id(cluster_ids, current_cluster_id, direction=-1)


def face_detail_for_ui(repository, face_id: str | None, *, show_private_crops: bool = False) -> dict[str, Any]:
    if not face_id:
        return _empty_detail()
    rows = list_face_detections(repository, limit=1_000_000)
    row = next((item for item in rows if item.get("id") == face_id), None)
    if not row:
        return _empty_detail()
    summary = [
        f"face_id: {row.get('id')}",
        f"media_id: {row.get('media_id')}",
        f"captured_at: {row.get('captured_at') or row.get('fallback_captured_at') or ''}",
        f"status: {row.get('status') or ''}",
        f"review_status: {row.get('review_status') or ''}",
        f"engine: {row.get('engine') or ''}",
        f"score: {row.get('detection_score') if row.get('detection_score') is not None else ''}",
        f"bbox: {_bbox(row)}",
        "identity: not inferred",
        "relationship/emotion: not inferred",
    ]
    status = str(row.get("status") or "")
    if status != "success":
        return {
            "summary": "\n".join(summary + ["note: face thumbnail is only available for status=success rows with bbox."]),
            "face_thumbnail": None,
            "original_thumbnail": row.get("media_thumbnail_path") if show_private_crops else None,
            "file_name": redact_text(row.get("file_name"), max_chars=80),
            "crop_note": f"{status or 'non-success'} row has no face bbox/crop; original media thumbnail only",
            "review_status": row.get("review_status") or "",
        }
    face_thumbnail = row.get("thumbnail_path")
    if show_private_crops and (not face_thumbnail or not Path(str(face_thumbnail)).expanduser().exists()):
        repaired = ensure_face_thumbnail(repository, face_id=str(row.get("id") or ""))
        face_thumbnail = repaired.get("thumbnail_path") or face_thumbnail
    return {
        "summary": "\n".join(summary),
        "face_thumbnail": face_thumbnail if show_private_crops else None,
        "original_thumbnail": row.get("media_thumbnail_path") if show_private_crops else None,
        "file_name": redact_text(row.get("file_name"), max_chars=80),
        "crop_note": "private face crops hidden" if not show_private_crops else "private face crops shown locally",
        "review_status": row.get("review_status") or "",
    }


def update_face_review_for_ui(repository, face_id: str | None, review_status: str) -> tuple[str, dict[str, Any]]:
    if not face_id:
        return "face_id is required", _empty_detail()
    row = update_face_review_status(repository, face_id=face_id, review_status=review_status)
    return f"updated {row.get('id')} review_status={row.get('review_status')}", face_detail_for_ui(repository, str(row.get("id")), show_private_crops=True)


def face_cluster_review_for_ui(
    repository,
    *,
    status: str | None = "unreviewed",
    limit: int = 50,
    public_mode: bool = False,
    show_private_crops: bool = True,
) -> dict[str, Any]:
    rows = list_face_clusters(repository, status=None if status in (None, "all") else status, limit=limit)
    table = [_cluster_row_for_table(repository, row, public_mode=public_mode, index=i + 1) for i, row in enumerate(rows)]
    gallery, gallery_cluster_ids = _cluster_gallery_items_with_ids(
        repository,
        rows,
        public_mode=public_mode,
        show_private_crops=show_private_crops,
    )
    return {
        "rows": rows,
        "table": table,
        "cluster_ids": [row[0] for row in table],
        "gallery": gallery,
        "gallery_cluster_ids": gallery_cluster_ids,
    }


def face_cluster_detail_for_ui(
    repository,
    cluster_id: str | None,
    *,
    show_private_crops: bool = False,
    public_mode: bool = False,
) -> dict[str, Any]:
    if not cluster_id:
        return _empty_cluster_detail()
    clusters = list_face_clusters(repository, cluster_id=cluster_id, limit=1)
    if not clusters:
        return _empty_cluster_detail()
    cluster = clusters[0]
    members = _cluster_members(repository, cluster_id)
    linked_people = _linked_people_for_cluster(repository, cluster_id, public_mode=public_mode)
    summary = [
        "人物名は手動で設定してください。アプリは顔から名前や関係性を自動推定しません。",
        f"cluster_id: {cluster.get('id')}",
        f"cluster_label: {cluster.get('cluster_label') or ''}",
        f"face_count: {cluster.get('face_count') or 0}",
        f"first_seen_at: {cluster.get('first_seen_at') or ''}",
        f"last_seen_at: {cluster.get('last_seen_at') or ''}",
        f"status: {cluster.get('status') or ''}",
        f"review_status: {cluster.get('review_status') or ''}",
        f"linked_persons: {', '.join(person['display_name'] for person in linked_people) if linked_people else 'none'}",
        "identity: manual label only",
        "relationship/emotion: not inferred",
    ]
    thumbnails: list[str] = []
    thumbnail_face_ids: list[str] = []
    if show_private_crops:
        for row in members[:12]:
            thumbnail = _member_thumbnail_path(repository, row)
            if not thumbnail:
                continue
            thumbnails.append(thumbnail)
            thumbnail_face_ids.append(str(row.get("face_id") or ""))
    return {
        "summary": "\n".join(summary),
        "member_table": [_member_row_for_table(row) for row in members],
        "member_thumbnails": thumbnails[:12],
        "member_thumbnail_face_ids": thumbnail_face_ids[:12],
        "linked_people": linked_people,
        "privacy_note": "private face thumbnails hidden" if not show_private_crops else "private face thumbnails shown locally",
    }


def face_cluster_action_for_ui(repository, cluster_id: str | None, status: str) -> tuple[str, dict[str, Any]]:
    if not cluster_id:
        return "cluster_id is required", _empty_cluster_detail()
    row = update_face_cluster_status(repository, cluster_id=cluster_id, status=status)
    return f"updated face cluster {row.get('id')} status={row.get('status')}", face_cluster_detail_for_ui(repository, cluster_id, show_private_crops=True)


def label_face_cluster_for_ui(
    repository,
    cluster_id: str | None,
    name: str | None,
    public_name: str | None,
    privacy_level: str,
) -> dict[str, Any]:
    """Create or reuse a manual person label and link it to a face cluster.

    Matching is intentionally simple and explicit: the same display name means
    the same person. The app still does not infer identity from face content.
    """

    if not cluster_id:
        return _label_result(repository, "cluster_id is required", None, cluster_id)
    display_name = (name or "").strip()
    if not display_name:
        return _label_result(repository, "display_name is required", None, cluster_id)
    person = _find_person_by_display_name(repository, display_name)
    created = False
    if person is None:
        person = create_person(
            repository,
            name=display_name,
            public_name=public_name or None,
            privacy_level=privacy_level or "private",
        )
        created = True
    elif public_name or privacy_level:
        person = update_person(
            repository,
            person_id=str(person["id"]),
            public_name=public_name or None,
            privacy_level=privacy_level or None,
        )
    link_person_face_cluster(repository, person_id=str(person["id"]), cluster_id=str(cluster_id), yes=True)
    action = "created" if created else "reused"
    message = f"{action} person '{display_name}' and linked to {cluster_id}. Same display_name clusters are treated as the same person."
    return _label_result(repository, message, str(person["id"]), cluster_id)


def create_person_for_ui(
    repository,
    name: str | None,
    public_name: str | None,
    privacy_level: str,
) -> tuple[str, list[list[Any]], list[str]]:
    if not name:
        rows = list_persons(repository)
        return "name is required", _persons_table(rows), [row["id"] for row in rows]
    row = create_person(repository, name=name, public_name=public_name, privacy_level=privacy_level)
    rows = list_persons(repository)
    return f"created person {row.get('id')}", _persons_table(rows), [person["id"] for person in rows]


def update_person_for_ui(
    repository,
    person_id: str | None,
    name: str | None,
    public_name: str | None,
    privacy_level: str | None,
) -> tuple[str, list[list[Any]], list[str]]:
    if not person_id:
        rows = list_persons(repository)
        return "person_id is required", _persons_table(rows), [row["id"] for row in rows]
    row = update_person(
        repository,
        person_id=person_id,
        name=name or None,
        public_name=public_name,
        privacy_level=privacy_level or None,
    )
    rows = list_persons(repository)
    return f"updated person {row.get('id')}", _persons_table(rows), [person["id"] for person in rows]


def add_person_alias_for_ui(repository, person_id: str | None, alias: str | None) -> str:
    if not person_id:
        return "person_id is required"
    if not alias:
        return "alias is required"
    add_person_alias(repository, person_id=person_id, alias=alias)
    return f"added alias for {person_id}"


def link_person_cluster_for_ui(repository, person_id: str | None, cluster_id: str | None) -> str:
    if not person_id or not cluster_id:
        return "person_id and cluster_id are required"
    link_person_face_cluster(repository, person_id=person_id, cluster_id=cluster_id, yes=True)
    return f"linked {person_id} to {cluster_id}"


def unlink_person_cluster_for_ui(repository, person_id: str | None, cluster_id: str | None) -> str:
    if not person_id or not cluster_id:
        return "person_id and cluster_id are required"
    result = unlink_person_face_cluster(repository, person_id=person_id, cluster_id=cluster_id, yes=True)
    return f"unlinked {person_id} from {cluster_id} deleted={result.get('deleted')}"


def persons_for_ui(repository, *, public_mode: bool = False, limit: int = 200) -> dict[str, Any]:
    rows = list_persons(repository, limit=limit, public_mode=public_mode)
    return {"rows": rows, "table": _persons_table(rows), "person_ids": [row["id"] for row in rows]}


def person_cluster_overview_for_ui(
    repository,
    *,
    public_mode: bool = False,
    show_private_crops: bool = True,
    limit: int = 300,
) -> dict[str, Any]:
    """Return a person-centric view of manually linked face clusters."""
    rows = _person_cluster_rows(repository, limit=limit)
    table = [_person_cluster_row_for_table(row, public_mode=public_mode) for row in rows]
    gallery, gallery_cluster_ids, gallery_person_ids = _person_cluster_gallery_items_with_ids(
        repository,
        rows,
        public_mode=public_mode,
        show_private_crops=show_private_crops,
    )
    linked_cluster_ids = [str(row.get("cluster_id") or "") for row in rows if row.get("cluster_id")]
    linked_person_ids = [str(row.get("person_id") or "") for row in rows if row.get("person_id")]
    unlinked_count = _unlinked_cluster_count(repository)
    return {
        "rows": rows,
        "table": table,
        "gallery": gallery,
        "gallery_cluster_ids": gallery_cluster_ids,
        "gallery_person_ids": gallery_person_ids,
        "linked_cluster_ids": linked_cluster_ids,
        "linked_person_ids": sorted(set(linked_person_ids)),
        "summary": f"linked clusters={len(linked_cluster_ids)} / linked persons={len(set(linked_person_ids))} / unlabeled clusters={unlinked_count}",
    }


def link_cluster_to_person_for_ui(
    repository,
    person_id: str | None,
    cluster_id: str | None,
    *,
    public_mode: bool = False,
    show_private_crops: bool = True,
) -> dict[str, Any]:
    if not person_id or not cluster_id:
        return _organizer_result(repository, "person_id and cluster_id are required", public_mode=public_mode, show_private_crops=show_private_crops)
    link_person_face_cluster(repository, person_id=person_id, cluster_id=cluster_id, yes=True)
    return _organizer_result(
        repository,
        f"linked cluster {cluster_id} to person {person_id}",
        public_mode=public_mode,
        show_private_crops=show_private_crops,
    )


def unlink_cluster_from_person_for_ui(
    repository,
    person_id: str | None,
    cluster_id: str | None,
    *,
    public_mode: bool = False,
    show_private_crops: bool = True,
) -> dict[str, Any]:
    if not person_id or not cluster_id:
        return _organizer_result(repository, "person_id and cluster_id are required", public_mode=public_mode, show_private_crops=show_private_crops)
    result = unlink_person_face_cluster(repository, person_id=person_id, cluster_id=cluster_id, yes=True)
    return _organizer_result(
        repository,
        f"removed cluster {cluster_id} from person {person_id} deleted={result.get('deleted')}",
        public_mode=public_mode,
        show_private_crops=show_private_crops,
    )


def merge_cluster_into_target_person_for_ui(
    repository,
    source_cluster_id: str | None,
    target_cluster_id: str | None,
    *,
    public_mode: bool = False,
    show_private_crops: bool = True,
) -> dict[str, Any]:
    """Treat two clusters as the same person by linking source to target's person.

    This intentionally does not move face_cluster_members. Keeping the original
    clusters intact preserves review provenance while person_face_clusters says
    they should be treated as the same manually verified person.
    """
    if not source_cluster_id or not target_cluster_id:
        return _organizer_result(repository, "source cluster_id and target cluster_id are required", public_mode=public_mode, show_private_crops=show_private_crops)
    if source_cluster_id == target_cluster_id:
        return _organizer_result(repository, "source and target cluster_id are the same", public_mode=public_mode, show_private_crops=show_private_crops)
    target_people = _linked_people_for_cluster(repository, target_cluster_id, public_mode=False)
    if not target_people:
        return _organizer_result(
            repository,
            "target cluster has no person label. Name/link the target cluster first.",
            public_mode=public_mode,
            show_private_crops=show_private_crops,
        )
    if len(target_people) > 1:
        return _organizer_result(
            repository,
            "target cluster has multiple linked persons. Select person_id and use Link selected cluster to person.",
            public_mode=public_mode,
            show_private_crops=show_private_crops,
        )
    target_person_id = str(target_people[0]["id"])
    _replace_cluster_person_links(repository, cluster_id=source_cluster_id, person_id=target_person_id)
    link_person_face_cluster(repository, person_id=target_person_id, cluster_id=source_cluster_id, yes=True)
    return _organizer_result(
        repository,
        f"merged cluster {source_cluster_id} into target person's label from {target_cluster_id} ({target_person_id})",
        public_mode=public_mode,
        show_private_crops=show_private_crops,
    )


def line_speakers_for_ui(repository, *, limit: int = 100) -> dict[str, Any]:
    rows = list_line_speakers(repository, limit=limit)
    table = [
        [
            row.get("chat_id") or "",
            row.get("speaker_name") or "",
            row.get("message_count") or 0,
            row.get("first_seen_at") or "",
            row.get("last_seen_at") or "",
            row.get("linked_person_names") or "",
            row.get("linked_person_public_names") or "",
        ]
        for row in rows
    ]
    return {"rows": rows, "table": table}


def link_line_speaker_for_ui(
    repository,
    chat_id: str | None,
    speaker_name: str | None,
    person_id: str | None,
    add_alias: bool,
) -> str:
    if not chat_id or not speaker_name or not person_id:
        return "chat_id, speaker_name, and person_id are required"
    link_line_speaker_to_person(
        repository,
        chat_id=chat_id,
        speaker_name=speaker_name,
        person_id=person_id,
        add_alias=add_alias,
        yes=True,
    )
    return f"linked LINE speaker {speaker_name} to {person_id}"


def unlink_line_speaker_for_ui(
    repository,
    chat_id: str | None,
    speaker_name: str | None,
    person_id: str | None,
) -> str:
    if not chat_id or not speaker_name or not person_id:
        return "chat_id, speaker_name, and person_id are required"
    result = unlink_line_speaker_from_person(
        repository,
        chat_id=chat_id,
        speaker_name=speaker_name,
        person_id=person_id,
        yes=True,
    )
    return f"unlinked LINE speaker {speaker_name} from {person_id} deleted={result.get('deleted')}"


def _label_result(repository, message: str, person_id: str | None, cluster_id: str | None) -> dict[str, Any]:
    people = persons_for_ui(repository, public_mode=False)
    detail = face_cluster_detail_for_ui(repository, cluster_id, show_private_crops=True, public_mode=False)
    return {
        "message": message,
        "person_id": person_id,
        "person_table": people["table"],
        "person_ids": people["person_ids"],
        "cluster_detail": detail,
    }


def _find_person_by_display_name(repository, display_name: str) -> dict[str, Any] | None:
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        row = connection.execute(
            """
            SELECT *
            FROM persons
            WHERE display_name = ?
              AND deleted_at IS NULL
            ORDER BY created_at ASC, id ASC
            LIMIT 1
            """,
            (display_name,),
        ).fetchone()
        return dict(row) if row is not None else None


def _row_for_table(row: dict[str, Any]) -> list[Any]:
    return [
        row.get("id") or "",
        row.get("media_id") or "",
        row.get("captured_at") or row.get("fallback_captured_at") or "",
        redact_text(row.get("file_name"), max_chars=80),
        row.get("status") or "",
        _bbox(row),
        row.get("detection_score") if row.get("detection_score") is not None else "",
        row.get("review_status") or "",
        row.get("engine") or "",
    ]


def _cluster_row_for_table(repository, row: dict[str, Any], *, public_mode: bool, index: int) -> list[Any]:
    linked = _linked_people_for_cluster(repository, str(row.get("id")), public_mode=public_mode)
    return [
        row.get("id") or "",
        row.get("cluster_label") or "",
        row.get("face_count") or 0,
        row.get("first_seen_at") or "",
        row.get("last_seen_at") or "",
        row.get("status") or "",
        row.get("review_status") or "",
        ", ".join(person.get("display_name") or "" for person in linked) if linked else "",
        "manual labels only",
    ]


def _member_row_for_table(row: dict[str, Any]) -> list[Any]:
    return [
        row.get("face_id") or "",
        row.get("media_id") or "",
        row.get("captured_at") or row.get("fallback_captured_at") or "",
        redact_text(row.get("file_name"), max_chars=80),
        row.get("detection_score") if row.get("detection_score") is not None else "",
        row.get("review_status") or "",
    ]


def _face_detection_gallery_items(
    repository,
    rows: list[dict[str, Any]],
    *,
    show_private_crops: bool,
) -> tuple[list[tuple[str, str]], list[str]]:
    if not show_private_crops:
        return [], []
    items: list[tuple[str, str]] = []
    face_ids: list[str] = []
    for row in rows:
        thumbnail = _member_thumbnail_path(repository, row)
        if not thumbnail:
            continue
        face_id = str(row.get("id") or row.get("face_id") or "")
        score = row.get("detection_score")
        caption = " / ".join(
            part
            for part in [
                str(row.get("captured_at") or row.get("fallback_captured_at") or "")[:10],
                face_id,
                f"score={float(score):.2f}" if isinstance(score, (int, float)) else "",
            ]
            if part
        )
        items.append((thumbnail, caption))
        face_ids.append(face_id)
    return items, face_ids


def _cluster_gallery_items(
    repository,
    rows: list[dict[str, Any]],
    *,
    public_mode: bool,
    show_private_crops: bool,
) -> list[tuple[str, str]]:
    items, _cluster_ids = _cluster_gallery_items_with_ids(
        repository,
        rows,
        public_mode=public_mode,
        show_private_crops=show_private_crops,
    )
    return items


def _cluster_gallery_items_with_ids(
    repository,
    rows: list[dict[str, Any]],
    *,
    public_mode: bool,
    show_private_crops: bool,
) -> tuple[list[tuple[str, str]], list[str]]:
    if public_mode or not show_private_crops:
        return [], []
    items: list[tuple[str, str]] = []
    cluster_ids: list[str] = []
    for index, row in enumerate(rows, start=1):
        cluster_id = str(row.get("id") or "")
        thumbnail = _representative_cluster_thumbnail(repository, cluster_id)
        if not thumbnail:
            continue
        linked = _linked_people_for_cluster(repository, cluster_id, public_mode=public_mode)
        linked_label = ", ".join(person.get("display_name") or "" for person in linked) if linked else "unlabeled"
        caption = " / ".join(
            part
            for part in [
                str(row.get("first_seen_at") or "")[:10],
                cluster_id,
                linked_label,
                f"faces={row.get('face_count') or 0}",
            ]
            if part
        )
        items.append((thumbnail, caption))
        cluster_ids.append(cluster_id)
    return items, cluster_ids


def _representative_cluster_thumbnail(repository, cluster_id: str) -> str | None:
    for member in _cluster_members(repository, cluster_id):
        thumbnail = _member_thumbnail_path(repository, member)
        if thumbnail:
            return thumbnail
    return None


def _existing_private_image_path(path_value: Any) -> str | None:
    path_text = str(path_value or "").strip()
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if path.exists():
        return str(path.resolve())
    return None


def _member_thumbnail_path(repository, row: dict[str, Any]) -> str | None:
    """Return a local private thumbnail/crop path, repairing missing thumbnails lazily."""
    existing_thumbnail = _existing_private_image_path(row.get("thumbnail_path"))
    if existing_thumbnail:
        return existing_thumbnail
    existing_crop = _existing_private_image_path(row.get("crop_path"))
    if existing_crop:
        return existing_crop
    face_id = str(row.get("face_id") or row.get("id") or "").strip()
    if not face_id:
        return None
    repaired = ensure_face_thumbnail(repository, face_id=face_id)
    return _existing_private_image_path(repaired.get("thumbnail_path")) or _existing_private_image_path(repaired.get("crop_path"))


def _persons_table(rows: list[dict[str, Any]]) -> list[list[Any]]:
    return [
        [
            row.get("id") or "",
            row.get("display_name") or "",
            row.get("public_name") or "",
            row.get("privacy_level") or "",
            row.get("linked_clusters_count", 0),
            row.get("alias_count", 0),
        ]
        for row in rows
    ]


def _person_cluster_rows(repository, *, limit: int) -> list[dict[str, Any]]:
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        rows = connection.execute(
            """
            SELECT persons.id AS person_id,
                   persons.display_name,
                   persons.public_name,
                   persons.privacy_level,
                   face_clusters.id AS cluster_id,
                   face_clusters.cluster_label,
                   face_clusters.face_count,
                   face_clusters.first_seen_at,
                   face_clusters.last_seen_at,
                   face_clusters.status,
                   face_clusters.review_status,
                   person_face_clusters.verified_at,
                   person_face_clusters.source
            FROM person_face_clusters
            JOIN persons ON persons.id = person_face_clusters.person_id
            JOIN face_clusters ON face_clusters.id = person_face_clusters.face_cluster_id
            WHERE persons.deleted_at IS NULL
              AND COALESCE(persons.hidden, 0) = 0
            ORDER BY persons.display_name ASC, face_clusters.first_seen_at ASC, face_clusters.id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def _person_cluster_row_for_table(row: dict[str, Any], *, public_mode: bool) -> list[Any]:
    display_name = str(row.get("display_name") or "")
    if public_mode:
        display_name = public_person_name(row, index=1) or "非公開"
    return [
        row.get("person_id") or "",
        display_name,
        row.get("public_name") or "",
        row.get("privacy_level") or "",
        row.get("cluster_id") or "",
        row.get("cluster_label") or "",
        row.get("face_count") or 0,
        row.get("first_seen_at") or "",
        row.get("last_seen_at") or "",
        row.get("status") or "",
        row.get("review_status") or "",
        row.get("source") or "",
    ]


def _person_cluster_gallery_items(
    repository,
    rows: list[dict[str, Any]],
    *,
    public_mode: bool,
    show_private_crops: bool,
) -> list[tuple[str, str]]:
    items, _cluster_ids, _person_ids = _person_cluster_gallery_items_with_ids(
        repository,
        rows,
        public_mode=public_mode,
        show_private_crops=show_private_crops,
    )
    return items


def _person_cluster_gallery_items_with_ids(
    repository,
    rows: list[dict[str, Any]],
    *,
    public_mode: bool,
    show_private_crops: bool,
) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    if public_mode or not show_private_crops:
        return [], [], []
    items: list[tuple[str, str]] = []
    cluster_ids: list[str] = []
    person_ids: list[str] = []
    for row in rows:
        cluster_id = str(row.get("cluster_id") or "")
        thumbnail = _representative_cluster_thumbnail(repository, cluster_id)
        if not thumbnail:
            continue
        caption = " / ".join(
            part
            for part in [
                str(row.get("display_name") or ""),
                str(row.get("first_seen_at") or "")[:10],
                cluster_id,
                f"faces={row.get('face_count') or 0}",
            ]
            if part
        )
        items.append((thumbnail, caption))
        cluster_ids.append(cluster_id)
        person_ids.append(str(row.get("person_id") or ""))
    return items, cluster_ids, person_ids


def _unlinked_cluster_count(repository) -> int:
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        return int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM face_clusters
                LEFT JOIN person_face_clusters ON person_face_clusters.face_cluster_id = face_clusters.id
                WHERE person_face_clusters.person_id IS NULL
                """
            ).fetchone()[0]
            or 0
        )


def _replace_cluster_person_links(repository, *, cluster_id: str, person_id: str) -> None:
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        connection.execute(
            "DELETE FROM person_face_clusters WHERE face_cluster_id = ? AND person_id != ?",
            (cluster_id, person_id),
        )
        connection.commit()


def _organizer_result(repository, message: str, *, public_mode: bool, show_private_crops: bool) -> dict[str, Any]:
    overview = person_cluster_overview_for_ui(
        repository,
        public_mode=public_mode,
        show_private_crops=show_private_crops,
    )
    people = persons_for_ui(repository, public_mode=public_mode)
    return {
        "message": message,
        "overview": overview,
        "person_table": people["table"],
        "person_ids": people["person_ids"],
    }


def _cluster_members(repository, cluster_id: str) -> list[dict[str, Any]]:
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        rows = connection.execute(
            """
            SELECT face_detections.id AS face_id,
                   face_detections.media_id,
                   face_detections.thumbnail_path,
                   face_detections.crop_path,
                   face_detections.detection_score,
                   face_detections.review_status,
                   media_items.file_name,
                   media_items.captured_at,
                   media_items.fallback_captured_at
            FROM face_cluster_members
            JOIN face_detections ON face_detections.id = face_cluster_members.face_id
            JOIN media_items ON media_items.id = face_detections.media_id
            WHERE face_cluster_members.cluster_id = ?
            ORDER BY media_items.captured_at ASC, face_detections.id ASC
            """,
            (cluster_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def _linked_people_for_cluster(repository, cluster_id: str, *, public_mode: bool) -> list[dict[str, Any]]:
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        rows = connection.execute(
            """
            SELECT persons.*
            FROM person_face_clusters
            JOIN persons ON persons.id = person_face_clusters.person_id
            WHERE person_face_clusters.face_cluster_id = ?
            ORDER BY persons.display_name ASC
            """,
            (cluster_id,),
        ).fetchall()
        people: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            person = dict(row)
            if public_mode:
                person["display_name"] = public_person_name(person, index=index) or "非公開"
            people.append(person)
        return people


def _bbox(row: dict[str, Any]) -> str:
    if row.get("bbox_x") is None:
        return ""
    return (
        f"{float(row.get('bbox_x') or 0):.0f},"
        f"{float(row.get('bbox_y') or 0):.0f},"
        f"{float(row.get('bbox_w') or 0):.0f},"
        f"{float(row.get('bbox_h') or 0):.0f}"
    )


def _empty_detail() -> dict[str, Any]:
    return {
        "summary": "",
        "face_thumbnail": None,
        "original_thumbnail": None,
        "file_name": "",
        "crop_note": "",
        "review_status": "",
    }


def _empty_cluster_detail() -> dict[str, Any]:
    return {
        "summary": "",
        "member_table": [],
        "member_thumbnails": [],
        "linked_people": [],
        "privacy_note": "",
    }
