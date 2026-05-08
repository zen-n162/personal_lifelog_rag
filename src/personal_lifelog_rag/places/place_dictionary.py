"""Read and validate a local places.yaml file."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from personal_lifelog_rag.places.geo import parse_lat_lon
from personal_lifelog_rag.places.schemas import Place, PlaceValidation


DEFAULT_PRIVATE_PLACES_PATH = Path("private_config/places.yaml")
ALLOWED_PRIVACY_LEVELS = {"normal", "sensitive"}
REQUIRED_FIELDS = {"id", "name", "display_name", "lat", "lon", "radius_m"}


class PlaceConfigError(ValueError):
    """Raised when a local place dictionary cannot be parsed or validated."""


def load_place_dictionary(path: str | Path | None = None, *, required: bool = True) -> list[Place]:
    """Load validated places from a small YAML subset.

    The project intentionally avoids introducing a heavy YAML dependency for
    this local-only config. The parser supports the simple `places: - key:
    value` structure documented in configs/places.example.yaml.
    """

    resolved = Path(path or DEFAULT_PRIVATE_PLACES_PATH).expanduser()
    if not resolved.exists():
        if required:
            raise PlaceConfigError(f"Places config not found: {resolved}")
        return []
    validation = validate_place_dictionary(resolved)
    if not validation.valid:
        raise PlaceConfigError("; ".join(validation.errors))
    return validation.places


def validate_place_dictionary(path: str | Path) -> PlaceValidation:
    resolved = Path(path).expanduser()
    try:
        records = _parse_places_yaml(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        return PlaceValidation(False, [], [f"Cannot read places config: {exc}"])
    except ValueError as exc:
        return PlaceValidation(False, [], [str(exc)])

    errors: list[str] = []
    places: list[Place] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        context = f"places[{index}]"
        missing = sorted(REQUIRED_FIELDS - record.keys())
        if missing:
            errors.append(f"{context}: missing required field(s): {', '.join(missing)}")
            continue
        place_id = str(record.get("id") or "").strip()
        if not place_id:
            errors.append(f"{context}: id must not be empty")
            continue
        if place_id in seen_ids:
            errors.append(f"{context}: duplicate id: {place_id}")
            continue
        seen_ids.add(place_id)

        parsed = parse_lat_lon(record.get("lat"), record.get("lon"))
        if parsed is None:
            errors.append(f"{context}: lat/lon is out of range or invalid")
            continue
        lat, lon = parsed

        try:
            radius_m = float(record["radius_m"])
        except (TypeError, ValueError):
            errors.append(f"{context}: radius_m must be a positive number")
            continue
        if radius_m <= 0:
            errors.append(f"{context}: radius_m must be positive")
            continue

        privacy_level = str(record.get("privacy_level") or "normal").strip() or "normal"
        if privacy_level not in ALLOWED_PRIVACY_LEVELS:
            errors.append(
                f"{context}: privacy_level must be one of {', '.join(sorted(ALLOWED_PRIVACY_LEVELS))}"
            )
            continue

        places.append(
            Place(
                id=place_id,
                name=str(record.get("name") or place_id),
                display_name=str(record.get("display_name") or record.get("name") or place_id),
                lat=lat,
                lon=lon,
                radius_m=radius_m,
                category=str(record.get("category") or "custom"),
                privacy_level=privacy_level,
                show_exact_location=bool(record.get("show_exact_location", False)),
            )
        )

    return PlaceValidation(valid=not errors, places=places, errors=errors)


def _parse_places_yaml(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_places = False

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped == "places:":
            in_places = True
            continue
        if not in_places:
            continue
        if stripped.startswith("- "):
            if current is not None:
                records.append(current)
            current = {}
            remainder = stripped[2:].strip()
            if remainder:
                key, value = _split_key_value(remainder)
                current[key] = _parse_scalar(value)
            continue
        if current is None:
            raise ValueError("Invalid places config: list item must start with '-'")
        key, value = _split_key_value(stripped)
        current[key] = _parse_scalar(value)

    if current is not None:
        records.append(current)
    if not in_places:
        raise ValueError("Invalid places config: missing top-level 'places:'")
    return records


def _split_key_value(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise ValueError(f"Invalid places config line: {text}")
    key, value = text.split(":", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid places config key: {text}")
    return key, value.strip()


def _parse_scalar(value: str) -> Any:
    if value == "":
        return ""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value.strip("\"'")

