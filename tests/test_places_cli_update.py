from __future__ import annotations

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository
from personal_lifelog_rag.places.location_store import build_location_points_from_media, cluster_location_points


def test_places_cli_show_update_and_reject_cluster(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _seed_cluster(repository)
    cluster_id = repository._fetch_all("SELECT id FROM place_clusters ORDER BY id", [])[0]["id"]

    assert main(["--db-path", str(db_path), "places", "show-cluster", "--cluster-id", cluster_id]) == 0
    show_output = capsys.readouterr().out
    assert "exact coordinates: hidden" in show_output

    assert main([
        "--db-path",
        str(db_path),
        "places",
        "create",
        "--id",
        "place_test",
        "--name",
        "古い名前",
        "--privacy-level",
        "private",
    ]) == 0
    assert main([
        "--db-path",
        str(db_path),
        "places",
        "update",
        "--place-id",
        "place_test",
        "--name",
        "新しい名前",
        "--public-name",
        "公開名",
        "--category",
        "station",
        "--privacy-level",
        "public_label",
        "--manual-verified",
    ]) == 0
    update_output = capsys.readouterr().out
    assert "新しい名前" in update_output
    assert "manual_verified: 1" in update_output

    assert main(["--db-path", str(db_path), "places", "reject-cluster", "--cluster-id", cluster_id]) == 2
    assert main(["--db-path", str(db_path), "places", "reject-cluster", "--cluster-id", cluster_id, "--yes"]) == 0
    reject_output = capsys.readouterr().out
    assert "status=rejected" in reject_output


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
