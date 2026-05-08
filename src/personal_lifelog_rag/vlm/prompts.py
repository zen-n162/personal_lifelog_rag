"""Safety-conscious prompts for local visual analysis."""

SAFE_IMAGE_ANALYSIS_PROMPT_VERSION = "vlm_prompt_v1"

SAFE_IMAGE_ANALYSIS_PROMPT = """You are analyzing a personal photo locally for private lifelog search.

Return only observable, non-sensitive visual information.
Do not identify people.
Do not infer relationships, emotions, age, health, religion, politics, work, or other sensitive traits.
Do not guess names.
If people are present, only say "people_present" and optionally a rough count.
Prefer cautious language such as "possible" or "appears".

Return JSON with:
- caption
- short_caption
- scene_tags
- object_tags
- activity_tags
- location_cues
- food_cues
- people_count
- contains_text_hint
- confidence
- safety_flags
"""

