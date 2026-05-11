from __future__ import annotations

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.db.repository import LifelogRepository, connect
from personal_lifelog_rag.places.location_store import build_location_points_from_media, cluster_location_points


def test_nearby_location_points_become_cluster_and_far_point_separates(tmp_path) -> None:
    repository = LifelogRepository(tmp_path / "lifelog.sqlite")
    repository.initialize()
    _media(repository, "near_a", 35.0000, 139.0000)
    _media(repository, "near_b", 35.0002, 139.0002)
    _media(repository, "near_c", 35.0003, 139.0003)
    _media(repository, "far", 36.0000, 139.0000)
    build_location_points_from_media(repository, dry_run=False)

    report = cluster_location_points(repository, eps_meters=80, min_samples=3, dry_run=False)

    assert report.saved_clusters == 1
    assert report.assigned_points == 3
    with connect(repository.db_path) as connection:
        rows = connection.execute("SELECT id, point_count, photo_count FROM place_clusters").fetchall()
        assigned = connection.execute("SELECT COUNT(*) FROM location_points WHERE cluster_id IS NOT NULL").fetchone()[0]
    assert [(row["point_count"], row["photo_count"]) for row in rows] == [(3, 3)]
    assert assigned == 3


def test_cluster_places_cli_dry_run_uses_private_db_location_points(tmp_path, capsys) -> None:
    db_path = tmp_path / "lifelog.sqlite"
    repository = LifelogRepository(db_path)
    repository.initialize()
    _media(repository, "near_a", 35.0000, 139.0000)
    _media(repository, "near_b", 35.0002, 139.0002)
    _media(repository, "near_c", 35.0003, 139.0003)
    build_location_points_from_media(repository, dry_run=False)

    code = main([
        "--db-path",
        str(db_path),
        "cluster-places",
        "--from",
        "2025-01-01",
        "--to",
        "2025-01-31",
        "--eps-meters",
        "80",
        "--min-samples",
        "3",
        "--dry-run",
    ])
    output = capsys.readouterr().out

    assert code == 0
    assert "Place Clusters" in output
    assert "clusters: 1" in output
    assert "centroid coordinates are hidden" in output


def _media(repository: LifelogRepository, media_id: str, lat: float, lon: float) -> None:
    repository.add_media_item(
        id=media_id,
        file_path=f"/local/fake/{media_id}.jpg",
        file_name=f"{media_id}.jpg",
        file_hash=f"hash-{media_id}",
        captured_at="2025-01-02T10:00:00+09:00",
        gps_lat=lat,
        gps_lon=lon,
    )
