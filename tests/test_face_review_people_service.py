from __future__ import annotations

from pathlib import Path

from PIL import Image

from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.db.schema import initialize_schema
from personal_lifelog_rag.faces.person_service import create_person, link_person_face_cluster
from personal_lifelog_rag.ui.face_review_service import (
    face_cluster_detail_for_ui,
    face_cluster_review_for_ui,
    face_review_queue_for_ui,
    label_face_cluster_for_ui,
    link_cluster_to_person_for_ui,
    line_speakers_for_ui,
    link_line_speaker_for_ui,
    merge_cluster_into_target_person_for_ui,
    next_cluster_id,
    next_face_id,
    next_person_id,
    person_cluster_overview_for_ui,
    persons_for_ui,
    previous_cluster_id,
    previous_face_id,
    previous_person_id,
    unlink_cluster_from_person_for_ui,
    unlink_line_speaker_for_ui,
)


def test_face_cluster_review_service_returns_manual_person_candidates(tmp_path: Path) -> None:
    repository = _seed_cluster(tmp_path)
    person = create_person(repository, name="人物テストA", public_name="人物A", privacy_level="public_alias")
    link_person_face_cluster(repository, person_id=person["id"], cluster_id="cluster_ui", yes=True)

    queue = face_cluster_review_for_ui(repository, status="accepted", limit=10, public_mode=True)
    detail = face_cluster_detail_for_ui(repository, "cluster_ui", show_private_crops=False, public_mode=True)
    people = persons_for_ui(repository, public_mode=True)

    assert queue["cluster_ids"] == ["cluster_ui"]
    assert queue["table"][0][7] == "人物A"
    assert "人物A" in detail["summary"]
    assert detail["member_thumbnails"] == []
    assert people["table"][0][1] == "人物A"


def test_face_cluster_review_service_shows_private_cluster_gallery(tmp_path: Path) -> None:
    repository = _seed_cluster(tmp_path)

    queue = face_cluster_review_for_ui(repository, status="accepted", limit=10, public_mode=False, show_private_crops=True)
    public_queue = face_cluster_review_for_ui(repository, status="accepted", limit=10, public_mode=True, show_private_crops=True)

    assert queue["gallery"]
    assert queue["gallery_cluster_ids"] == ["cluster_ui"]
    assert queue["gallery"][0][0].endswith("face_thumb.jpg")
    assert "cluster_ui" in queue["gallery"][0][1]
    assert public_queue["gallery"] == []


def test_face_cluster_detail_uses_crop_when_thumbnail_is_missing(tmp_path: Path) -> None:
    repository = _seed_cluster(tmp_path)
    crop_path = tmp_path / "face_crop_only.jpg"
    Image.new("RGB", (48, 48), "black").save(crop_path)
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        connection.execute(
            "UPDATE face_detections SET thumbnail_path = NULL, crop_path = ? WHERE id = 'face_ui'",
            (str(crop_path),),
        )
        connection.commit()

    detail = face_cluster_detail_for_ui(repository, "cluster_ui", show_private_crops=True, public_mode=False)
    queue = face_cluster_review_for_ui(repository, status="accepted", limit=10, public_mode=False, show_private_crops=True)

    assert detail["member_thumbnails"] == [str(crop_path.resolve())]
    assert queue["gallery"][0][0] == str(crop_path.resolve())


def test_label_face_cluster_reuses_same_display_name_person(tmp_path: Path) -> None:
    repository = _seed_cluster(tmp_path)
    _seed_second_cluster(repository, tmp_path)

    first = label_face_cluster_for_ui(
        repository,
        "cluster_ui",
        "人物同名A",
        "人物A",
        "private",
    )
    second = label_face_cluster_for_ui(
        repository,
        "cluster_ui_2",
        "人物同名A",
        "人物A",
        "private",
    )

    assert first["person_id"] == second["person_id"]
    assert "reused person" in second["message"]
    people = persons_for_ui(repository)
    assert len(people["rows"]) == 1
    assert people["rows"][0]["linked_clusters_count"] == 2


def test_person_cluster_overview_and_unlink_wrong_cluster(tmp_path: Path) -> None:
    repository = _seed_cluster(tmp_path)
    person = create_person(repository, name="人物整理A", public_name="人物A", privacy_level="private")

    linked = link_cluster_to_person_for_ui(repository, person["id"], "cluster_ui")
    overview = person_cluster_overview_for_ui(repository, public_mode=False, show_private_crops=True)
    unlinked = unlink_cluster_from_person_for_ui(repository, person["id"], "cluster_ui")

    assert "linked cluster cluster_ui" in linked["message"]
    assert overview["table"][0][0] == person["id"]
    assert overview["table"][0][4] == "cluster_ui"
    assert overview["gallery"]
    assert overview["gallery_cluster_ids"] == ["cluster_ui"]
    assert overview["gallery_person_ids"] == [person["id"]]
    assert "removed cluster cluster_ui" in unlinked["message"]
    assert person_cluster_overview_for_ui(repository)["table"] == []


def test_face_detection_gallery_maps_thumbnail_to_face_id(tmp_path: Path) -> None:
    repository = _seed_cluster(tmp_path)

    queue = face_review_queue_for_ui(repository, status="success", review_status="accepted", limit=10)

    assert queue["gallery"]
    assert queue["gallery_face_ids"] == ["face_ui"]


def test_merge_cluster_into_target_person_uses_target_label(tmp_path: Path) -> None:
    repository = _seed_cluster(tmp_path)
    _seed_second_cluster(repository, tmp_path)
    person = create_person(repository, name="人物統合A", public_name="人物A", privacy_level="private")
    link_cluster_to_person_for_ui(repository, person["id"], "cluster_ui")

    result = merge_cluster_into_target_person_for_ui(repository, "cluster_ui_2", "cluster_ui")
    overview = person_cluster_overview_for_ui(repository, public_mode=False, show_private_crops=False)
    linked_clusters = sorted(row[4] for row in overview["table"])

    assert "merged cluster cluster_ui_2" in result["message"]
    assert linked_clusters == ["cluster_ui", "cluster_ui_2"]
    assert {row[0] for row in overview["table"]} == {person["id"]}


def test_face_review_ui_service_links_line_speaker_manually(tmp_path: Path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    repository.add_line_message(
        id="line_ui_speaker",
        chat_id="chat_ui",
        source_file="dummy_line.txt",
        sent_at="2025-01-01T10:00:00+09:00",
        sender="SpeakerUI",
        text="short dummy",
    )
    person = create_person(repository, name="人物テストUI", public_name="人物A", privacy_level="public_alias")

    before = line_speakers_for_ui(repository, limit=10)
    assert before["table"][0][1] == "SpeakerUI"

    status = link_line_speaker_for_ui(repository, "chat_ui", "SpeakerUI", person["id"], add_alias=True)
    assert "linked LINE speaker" in status
    linked = line_speakers_for_ui(repository, limit=10)
    assert linked["table"][0][5] == "人物テストUI"
    assert linked["table"][0][6] == "人物A"

    unlinked = unlink_line_speaker_for_ui(repository, "chat_ui", "SpeakerUI", person["id"])
    assert "deleted=1" in unlinked


def test_next_face_id_advances_with_wraparound() -> None:
    face_ids = ["face_a", "face_b", "face_c"]

    assert next_face_id(face_ids, None) == "face_a"
    assert next_face_id(face_ids, "face_a") == "face_b"
    assert next_face_id(face_ids, "face_c") == "face_a"
    assert previous_face_id(face_ids, None) == "face_c"
    assert previous_face_id(face_ids, "face_a") == "face_c"
    assert previous_face_id(face_ids, "face_c") == "face_b"
    assert next_face_id(face_ids, "unknown") == "face_a"
    assert previous_face_id(face_ids, "unknown") == "face_c"
    assert next_face_id([], "face_a") is None
    assert previous_face_id([], "face_a") is None


def test_face_review_cluster_and_person_ids_move_both_directions() -> None:
    cluster_ids = ["cluster_a", "cluster_b"]
    person_ids = ["person_a", "person_b"]

    assert next_cluster_id(cluster_ids, None) == "cluster_a"
    assert previous_cluster_id(cluster_ids, None) == "cluster_b"
    assert next_cluster_id(cluster_ids, "cluster_a") == "cluster_b"
    assert previous_cluster_id(cluster_ids, "cluster_a") == "cluster_b"

    assert next_person_id(person_ids, None) == "person_a"
    assert previous_person_id(person_ids, None) == "person_b"
    assert next_person_id(person_ids, "person_b") == "person_a"
    assert previous_person_id(person_ids, "person_a") == "person_b"


def _seed_cluster(tmp_path: Path) -> LifelogRepository:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    image_path = tmp_path / "face.jpg"
    thumb_path = tmp_path / "face_thumb.jpg"
    Image.new("RGB", (96, 96), "white").save(image_path)
    Image.new("RGB", (48, 48), "gray").save(thumb_path)
    repository.add_media_item(
        id="media_face_ui",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-face-ui",
        media_type="image",
        captured_at="2024-12-24T10:00:00+09:00",
    )
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        connection.execute(
            """
            INSERT INTO face_detections (
                id, media_id, detected_at, engine, model_name, status,
                bbox_x, bbox_y, bbox_w, bbox_h, detection_score,
                image_width, image_height, thumbnail_path, privacy_level, review_status
            )
            VALUES ('face_ui', 'media_face_ui', '2024-12-24T10:00:00+09:00',
                    'fake', 'fake', 'success', 10, 10, 32, 32, 0.9, 96, 96,
                    ?, 'private', 'accepted')
            """
            ,
            (str(thumb_path),),
        )
        connection.execute(
            """
            INSERT INTO face_clusters (
                id, cluster_label, representative_face_id, face_count,
                first_seen_at, last_seen_at, clustering_method, status,
                review_status, privacy_level
            )
            VALUES ('cluster_ui', 'person_candidate_001', 'face_ui', 1,
                    '2024-12-24T10:00:00+09:00', '2024-12-24T10:00:00+09:00',
                    'manual', 'accepted', 'reviewed', 'private')
            """
        )
        connection.execute("INSERT INTO face_cluster_members (cluster_id, face_id) VALUES ('cluster_ui', 'face_ui')")
        connection.commit()
    return repository


def _seed_second_cluster(repository: LifelogRepository, tmp_path: Path) -> None:
    image_path = tmp_path / "face2.jpg"
    thumb_path = tmp_path / "face_thumb2.jpg"
    Image.new("RGB", (96, 96), "white").save(image_path)
    Image.new("RGB", (48, 48), "gray").save(thumb_path)
    repository.add_media_item(
        id="media_face_ui_2",
        file_path=str(image_path),
        file_name=image_path.name,
        file_hash="hash-face-ui-2",
        media_type="image",
        captured_at="2024-12-25T10:00:00+09:00",
    )
    with connect(repository.db_path) as connection:
        initialize_schema(connection)
        connection.execute(
            """
            INSERT INTO face_detections (
                id, media_id, detected_at, engine, model_name, status,
                bbox_x, bbox_y, bbox_w, bbox_h, detection_score,
                image_width, image_height, thumbnail_path, privacy_level, review_status
            )
            VALUES ('face_ui_2', 'media_face_ui_2', '2024-12-25T10:00:00+09:00',
                    'fake', 'fake', 'success', 12, 12, 32, 32, 0.9, 96, 96,
                    ?, 'private', 'accepted')
            """,
            (str(thumb_path),),
        )
        connection.execute(
            """
            INSERT INTO face_clusters (
                id, cluster_label, representative_face_id, face_count,
                first_seen_at, last_seen_at, clustering_method, status,
                review_status, privacy_level
            )
            VALUES ('cluster_ui_2', 'person_candidate_002', 'face_ui_2', 1,
                    '2024-12-25T10:00:00+09:00', '2024-12-25T10:00:00+09:00',
                    'manual', 'accepted', 'reviewed', 'private')
            """
        )
        connection.execute("INSERT INTO face_cluster_members (cluster_id, face_id) VALUES ('cluster_ui_2', 'face_ui_2')")
        connection.commit()
