"""Dataclasses for local place matching."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Place:
    id: str
    name: str
    display_name: str
    lat: float
    lon: float
    radius_m: float
    category: str = "custom"
    privacy_level: str = "normal"
    show_exact_location: bool = False


@dataclass(frozen=True)
class PlaceMatch:
    place_id: str
    display_name: str
    distance_m: float
    privacy_level: str
    show_exact_location: bool
    category: str = "custom"

    def to_dict(self) -> dict[str, object]:
        return {
            "place_id": self.place_id,
            "display_name": self.display_name,
            "distance_m": self.distance_m,
            "privacy_level": self.privacy_level,
            "show_exact_location": self.show_exact_location,
            "category": self.category,
        }


@dataclass(frozen=True)
class PlaceValidation:
    valid: bool
    places: list[Place]
    errors: list[str]


@dataclass(frozen=True)
class PlaceCluster:
    cluster_id: str
    photo_count: int
    center_lat: float
    center_lon: float
    radius_m: float
    date_start: str | None
    date_end: str | None
    nearest_registered_place: PlaceMatch | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.cluster_id,
            "photos": self.photo_count,
            "center_lat": self.center_lat,
            "center_lon": self.center_lon,
            "radius_m": self.radius_m,
            "date_start": self.date_start,
            "date_end": self.date_end,
            "nearest_registered_place": (
                self.nearest_registered_place.to_dict()
                if self.nearest_registered_place
                else None
            ),
        }

