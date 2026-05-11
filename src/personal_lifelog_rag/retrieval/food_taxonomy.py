"""Specific food taxonomy for conservative visual search ranking."""

from __future__ import annotations

from dataclasses import dataclass


GENERIC_FOOD_TERMS = {
    "food",
    "meal",
    "bowl",
    "dish",
    "cafe",
    "restaurant",
    "rice",
    "noodle",
    "noodles",
    "soup",
}


@dataclass(frozen=True)
class FoodTaxonomyEntry:
    key: str
    display_name: str
    triggers: tuple[str, ...]
    specific_terms: tuple[str, ...]
    generic_terms: tuple[str, ...] = tuple(sorted(GENERIC_FOOD_TERMS))


FOOD_TAXONOMY: dict[str, FoodTaxonomyEntry] = {
    "ramen": FoodTaxonomyEntry(
        key="ramen",
        display_name="ラーメン",
        triggers=("ラーメン", "らーめん", "ramen"),
        specific_terms=(
            "ramen",
            "ramen noodle",
            "ramen noodles",
            "noodle soup",
            "chashu",
            "soft-boiled egg",
            "ajitama",
            "ramen_possible",
        ),
    ),
    "soba": FoodTaxonomyEntry(
        key="soba",
        display_name="そば",
        triggers=("そば", "蕎麦", "soba"),
        specific_terms=(
            "soba",
            "soba noodle",
            "soba noodles",
            "buckwheat noodle",
            "buckwheat noodles",
            "japanese noodle",
            "japanese noodles",
            "zaru soba",
            "tempura soba",
            "cold soba",
        ),
    ),
    "udon": FoodTaxonomyEntry(
        key="udon",
        display_name="うどん",
        triggers=("うどん", "饂飩", "udon"),
        specific_terms=(
            "udon",
            "udon noodle",
            "udon noodles",
            "thick wheat noodle",
            "thick wheat noodles",
            "japanese noodle",
            "japanese noodles",
            "tempura udon",
        ),
    ),
    "omurice": FoodTaxonomyEntry(
        key="omurice",
        display_name="オムライス",
        triggers=("オムライス", "おむらいす", "omurice"),
        specific_terms=(
            "omurice",
            "omelette rice",
            "omelet rice",
            "rice omelette",
            "rice omelet",
            "ketchup rice",
            "fried rice with omelette",
            "fried rice with omelet",
        ),
    ),
    "curry": FoodTaxonomyEntry(
        key="curry",
        display_name="カレー",
        triggers=("カレー", "curry"),
        specific_terms=(
            "curry",
            "japanese curry",
            "curry rice",
            "curry sauce",
            "katsu curry",
        ),
    ),
    "sushi": FoodTaxonomyEntry(
        key="sushi",
        display_name="寿司",
        triggers=("寿司", "すし", "鮨", "sushi"),
        specific_terms=(
            "sushi",
            "nigiri",
            "sashimi",
            "maki",
            "sushi roll",
            "sushi rolls",
        ),
    ),
    "yakiniku": FoodTaxonomyEntry(
        key="yakiniku",
        display_name="焼肉",
        triggers=("焼肉", "焼き肉", "yakiniku"),
        specific_terms=(
            "yakiniku",
            "grilled meat",
            "bbq meat",
            "barbecue meat",
            "table grill",
            "grill plate",
        ),
    ),
    "pizza": FoodTaxonomyEntry(
        key="pizza",
        display_name="ピザ",
        triggers=("ピザ", "pizza"),
        specific_terms=("pizza", "pizza slice", "cheese pizza", "flatbread pizza"),
    ),
    "cafe_drink": FoodTaxonomyEntry(
        key="cafe_drink",
        display_name="カフェドリンク",
        triggers=("コーヒー", "珈琲", "カフェラテ", "ラテ", "coffee", "cafe latte"),
        specific_terms=(
            "coffee",
            "cafe latte",
            "latte",
            "espresso",
            "iced coffee",
            "glass cup",
            "mug",
        ),
    ),
    "dessert": FoodTaxonomyEntry(
        key="dessert",
        display_name="デザート",
        triggers=("デザート", "ケーキ", "スイーツ", "dessert", "cake"),
        specific_terms=("dessert", "cake", "ice cream", "parfait", "sweet", "sweets", "pastry"),
    ),
}


def detect_specific_food_query(query: str) -> FoodTaxonomyEntry | None:
    """Return a specific food entry when the query names one dish category."""

    text = str(query or "")
    for entry in FOOD_TAXONOMY.values():
        if any(trigger and trigger.lower() in text.lower() for trigger in entry.triggers):
            return entry
    return None


def specific_food_terms_for_query(query: str) -> list[str]:
    entry = detect_specific_food_query(query)
    if not entry:
        return []
    return list(dict.fromkeys([*entry.triggers, *entry.specific_terms]))


def generic_food_terms() -> list[str]:
    return list(sorted(GENERIC_FOOD_TERMS))


def all_specific_food_triggers() -> list[str]:
    terms: list[str] = []
    for entry in FOOD_TAXONOMY.values():
        terms.extend(entry.triggers)
    return list(dict.fromkeys(terms))
