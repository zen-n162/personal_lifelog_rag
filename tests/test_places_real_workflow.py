from __future__ import annotations

from pathlib import Path

from personal_lifelog_rag.app.cli import main
from personal_lifelog_rag.places.clusterer import (
    cluster_place_candidates,
    place_clusters_to_yaml,
    write_place_cluster_suggestions,
)
from personal_lifelog_rag.places.place_dictionary import load_place_dictionary


def test_places_init_private_does_not_overwrite_existing_file(tmp_path, capsys) -> None:
    target = tmp_path / "private_config" / "places.yaml"

    first_exit = main(["places", "init-private", "--path", str(target), "--source", "configs/places.example.yaml"])
    first_output = capsys.readouterr().out
    target.write_text("places:\n", encoding="utf-8")
    second_exit = main(["places", "init-private", "--path", str(target), "--source", "configs/places.example.yaml"])
    second_output = capsys.readouterr().out

    assert first_exit == 0
    assert "Created private places config" in first_output
    assert second_exit == 0
    assert "not overwritten" in second_output
    assert target.read_text(encoding="utf-8") == "places:\n"


def test_cluster_place_suggestions_are_neutral_and_loadable(tmp_path) -> None:
    clusters = cluster_place_candidates(
        [
            _media("m1", 35.0, 139.0, "2024-12-24T10:00:00+09:00"),
            _media("m2", 35.0001, 139.0001, "2024-12-25T10:00:00+09:00"),
            _media("m3", 35.0002, 139.0002, "2024-12-26T10:00:00+09:00"),
        ],
        radius_m=500,
        min_points=3,
    )
    output_path = tmp_path / "private_config" / "place_suggestions.yaml"

    written = write_place_cluster_suggestions(output_path, clusters)
    yaml_text = written.read_text(encoding="utf-8")
    places = load_place_dictionary(written)

    assert "Generated local GPS cluster suggestions" in yaml_text
    assert "candidate_place_001" in yaml_text
    assert "候補地点001" in yaml_text
    assert "privacy_level: \"sensitive\"" in yaml_text
    assert "show_exact_location: false" in yaml_text
    assert len(places) == 1
    forbidden_labels = ("自宅", "大学", "職場", "恋人", "友人")
    assert not any(label in yaml_text for label in forbidden_labels)


def test_place_clusters_to_yaml_handles_empty_suggestions() -> None:
    yaml_text = place_clusters_to_yaml([])

    assert "places:" in yaml_text
    assert "自宅" not in yaml_text


def _media(media_id: str, lat: float, lon: float, captured_at: str) -> dict[str, object]:
    return {
        "id": media_id,
        "gps_lat": lat,
        "gps_lon": lon,
        "captured_at": captured_at,
    }
