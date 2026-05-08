from __future__ import annotations

from personal_lifelog_rag.places.matcher import match_place
from personal_lifelog_rag.places.schemas import Place


def test_match_place_returns_nearest_place_inside_radius() -> None:
    places = [
        Place(
            id="near_place",
            name="近い場所",
            display_name="近い場所",
            lat=10.0,
            lon=20.0,
            radius_m=200.0,
        ),
        Place(
            id="wide_place",
            name="広い場所",
            display_name="広い場所",
            lat=10.01,
            lon=20.01,
            radius_m=2_000.0,
        ),
    ]

    match = match_place(10.0005, 20.0005, places)

    assert match is not None
    assert match.place_id == "near_place"
    assert match.distance_m < 100.0


def test_match_place_returns_none_outside_radius() -> None:
    places = [
        Place(
            id="near_place",
            name="近い場所",
            display_name="近い場所",
            lat=10.0,
            lon=20.0,
            radius_m=100.0,
        )
    ]

    assert match_place(11.0, 20.0, places) is None

