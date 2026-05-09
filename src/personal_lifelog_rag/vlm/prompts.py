"""Safety-conscious prompt templates for local visual analysis."""

from __future__ import annotations

from dataclasses import dataclass


SAFE_IMAGE_ANALYSIS_PROMPT_VERSION = "lifelog_structured_tags_v1"

MANDATORY_SAFETY_RULES = """Return valid JSON only.
The first character of your response must be "{" and the last character must be "}".
Do not output <think> tags.
Do not include explanations outside JSON.
Do not include reasoning.

You are analyzing a personal photo locally for private lifelog search.

Return only observable, non-sensitive visual information.

Do not identify people.
Do not guess names.
Do not infer relationships such as girlfriend, boyfriend, lover, family, friend, coworker.
Do not infer emotions.
Do not infer age, health, disability, religion, politics, nationality, sexuality, or other sensitive traits.
If people are present, only return people_count if visually obvious, and add "people_present" to safety_flags.
Use cautious language.
Prefer "possible" tags over definitive claims.
Use possible tags instead of definitive claims.
Do not say the user definitely did an activity from the image alone.
Return valid JSON only."""


@dataclass(frozen=True)
class VlmPromptTemplate:
    name: str
    purpose: str
    output_schema: dict[str, object]
    prompt: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "output_schema": self.output_schema,
            "prompt": self.prompt,
        }


PROMPT_TEMPLATES = {
    "lifelog_safe_caption_v1": VlmPromptTemplate(
        name="lifelog_safe_caption_v1",
        purpose="Safely describe one personal photo for local lifelog search.",
        output_schema={
            "caption": "string",
            "short_caption": "string",
            "confidence": "number between 0.0 and 1.0",
            "safety_flags": "array of strings",
        },
        prompt=f"""{MANDATORY_SAFETY_RULES}

Task:
Describe the photo briefly for private local search. Keep it cautious and
non-sensitive. Do not infer what the user definitely did.

Return JSON exactly with:
{{
  "caption": "...",
  "short_caption": "...",
  "confidence": 0.0,
  "safety_flags": []
}}""",
    ),
    "lifelog_structured_tags_v1": VlmPromptTemplate(
        name="lifelog_structured_tags_v1",
        purpose="Generate cautious searchable tags and cues for local lifelog event building.",
        output_schema={
            "caption": "string",
            "short_caption": "string",
            "scene_tags": "array of strings",
            "object_tags": "array of strings",
            "activity_tags": "array of possible_* strings",
            "food_cues": "array of possible_* strings",
            "location_cues": "array of possible_* strings",
            "text_cues": "array of visible non-sensitive text cues",
            "people_count": "integer or null",
            "contains_text_hint": "boolean",
            "confidence": "number between 0.0 and 1.0",
            "uncertainty_notes": "array of strings",
            "safety_flags": "array of strings",
        },
        prompt=f"""{MANDATORY_SAFETY_RULES}

Task:
Extract only cautious visual cues useful for private local search and event
generation. Use possible-style tags such as meal_possible, cafe_possible,
station_possible, outdoor_possible, document_or_ticket_possible, or
screenshot_possible. If text is visible, summarize it only as a short,
non-sensitive cue.

Return JSON exactly with:
{{
  "caption": "...",
  "short_caption": "...",
  "scene_tags": [],
  "object_tags": [],
  "activity_tags": [],
  "food_cues": [],
  "location_cues": [],
  "text_cues": [],
  "people_count": null,
  "contains_text_hint": false,
  "confidence": 0.0,
  "uncertainty_notes": [],
  "safety_flags": []
}}""",
    ),
    "lifelog_event_cues_v1": VlmPromptTemplate(
        name="lifelog_event_cues_v1",
        purpose="Extract weak event-building cues from a photo without making claims.",
        output_schema={
            "event_cues": "object of boolean possible-cue fields",
            "supporting_visual_cues": "array of strings",
            "confidence": "number between 0.0 and 1.0",
            "caution": "string",
        },
        prompt=f"""{MANDATORY_SAFETY_RULES}

Task:
Return weak event-building cues only. These cues must not be treated as facts
unless OCR, LINE, GPS, or places support them.

Return JSON exactly with:
{{
  "event_cues": {{
    "meal_possible": false,
    "cafe_possible": false,
    "travel_possible": false,
    "station_possible": false,
    "shopping_possible": false,
    "outdoor_possible": false,
    "indoor_possible": false,
    "document_or_ticket_possible": false,
    "screenshot_possible": false
  }},
  "supporting_visual_cues": [],
  "confidence": 0.0,
  "caution": "visual-only inference"
}}""",
    ),
}

SAFE_IMAGE_ANALYSIS_PROMPT = PROMPT_TEMPLATES[SAFE_IMAGE_ANALYSIS_PROMPT_VERSION].prompt


def get_vlm_prompt_template(name: str | None = None) -> VlmPromptTemplate:
    resolved = name or SAFE_IMAGE_ANALYSIS_PROMPT_VERSION
    try:
        return PROMPT_TEMPLATES[resolved]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(f"Unknown VLM prompt template: {resolved}. Available: {available}") from exc


def format_vlm_prompt_template(name: str | None = None) -> str:
    return get_vlm_prompt_template(name).prompt
