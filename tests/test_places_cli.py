from __future__ import annotations

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.places.location_store import build_location_points_from_media, cluster_location_points


def test_places_cli_create_link_alias_and_privacy(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_cluster(repository)

    assert main([
        "--db-path",
        str(db_path),
        "places",
        "create",
        "--id",
        "place_station",
        "--name",
        "テスト駅",
        "--public-name",
        "駅周辺",
        "--category",
        "station",
        "--privacy-level",
        "public_label",
    ]) == 0
    create_output = capsys.readouterr().out
    assert "place_station" in create_output

    assert main(["--db-path", str(db_path), "places", "list-clusters", "--limit", "5"]) == 0
    list_output = capsys.readouterr().out
    cluster_id = next(line.split(":", 1)[0].strip("- ") for line in list_output.splitlines() if line.startswith("- place_cluster_"))

    assert main(["--db-path", str(db_path), "places", "link-cluster", "--place-id", "place_station", "--cluster-id", cluster_id, "--yes"]) == 0
    assert main(["--db-path", str(db_path), "places", "add-alias", "--place-id", "place_station", "--alias", "テスト駅前"]) == 0
    assert main(["--db-path", str(db_path), "places", "set-privacy", "--place-id", "place_station", "--privacy-level", "private"]) == 0
    output = capsys.readouterr().out

    assert "privacy_level: private" in output
    assert "テスト駅前" in output


def _seed_cluster(repository: LifelogRepository) -> None:
    for index, offset in enumerate((0.0, 0.0001, 0.0002), start=1):
        repository.add_media_item(
            id=f"media_{index}",
            file_path=f"/local/fake/{index}.jpg",
            file_name=f"{index}.jpg",
            file_hash=f"hash-{index}",
            captured_at=f"2025-01-0{index}T10:00:00+09:00",
            gps_lat=35.0 + offset,
            gps_lon=139.0 + offset,
        )
    build_location_points_from_media(repository, dry_run=False)
    cluster_location_points(repository, eps_meters=100, min_samples=3, dry_run=False)
