from __future__ import annotations

from personal_lifelog_rag.places.place_dictionary import load_place_dictionary, validate_place_dictionary


def test_places_yaml_can_be_loaded(tmp_path) -> None:
    path = tmp_path / "places.yaml"
    path.write_text(
        """
places:
  - id: home_area
    name: "自宅周辺"
    display_name: "自宅周辺"
    lat: 10.000000
    lon: 20.000000
    radius_m: 300
    category: "private"
    privacy_level: "sensitive"
    show_exact_location: false
""".strip(),
        encoding="utf-8",
    )

    places = load_place_dictionary(path)

    assert len(places) == 1
    assert places[0].id == "home_area"
    assert places[0].privacy_level == "sensitive"
    assert places[0].show_exact_location is False


def test_invalid_places_yaml_reports_errors(tmp_path) -> None:
    path = tmp_path / "places.yaml"
    path.write_text(
        """
places:
  - id: duplicate
    name: "A"
    display_name: "A"
    lat: 10.0
    lon: 20.0
    radius_m: 300
  - id: duplicate
    name: "B"
    display_name: "B"
    lat: 91.0
    lon: 20.0
    radius_m: -1
""".strip(),
        encoding="utf-8",
    )

    validation = validate_place_dictionary(path)

    assert not validation.valid
    assert any("duplicate id" in error for error in validation.errors)

