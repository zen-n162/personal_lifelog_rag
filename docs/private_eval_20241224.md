# Private Eval For 2024-12-24

This workflow creates a local regression suite around one well-inspected baseline
date. It is intended for private use only: keep generated files under
`private_eval/` and do not commit them.

## Why This Date

Use a date that already has a mix of LINE evidence, photos, OCR/VLM evidence,
events, and search behavior you have manually inspected. A stable baseline date
helps catch regressions in query intent, event generation, VLM safety, and
multimodal ranking.

## Date QA

The `date_qa` case checks that asking what happened on the date still returns
the target date, enough events, expected evidence types, and no overconfident
phrases such as `確実に` or `断定`.

## Place Visit

The `routed_qa` and `keyword_search` place cases check that natural-language
place questions route to `place_visit` and keep the baseline date near the top.
Put real place labels only in `private_eval/`, not committed examples.

## Food Photo Multimodal Search

The `image_search` and `multimodal_search` cases check that food-photo queries
can find VLM/event-backed candidates even when embedding results are unavailable.
These cases should cap VLM-only confidence at `中` because image analysis is a
candidate signal, not proof that an activity happened.

## VLM Quality

The `vlm_quality` case checks success counts, expected local engine name,
non-empty successful captions, forbidden terms, and allowed safety flags. For
this app, `people_present` is acceptable, while relationship, health, religion,
or political inference should fail.

## Event Quality

The `event_quality` case checks event count bounds, evidence types, orphan
evidence, and the rule that VLM-only evidence must not create high-confidence
events.

## Overclaim Detection

Use `should_not_include` for phrases that would overstate the evidence, for
example:

```yaml
should_not_include:
  - "確実に食べた"
  - "確実に新宿に行った"
  - "断定"
```

## Generate A Local Template

```bash
python -m personal_lifelog_rag.app.cli make-private-eval-template \
  --date 2024-12-24 \
  --output private_eval/questions_20241224.yaml
```

The generated YAML uses aggregate counts and evidence types. It does not include
raw LINE text, photo paths, exact GPS, or image content. Review the generated
file and adjust thresholds after confirming the current behavior.

## Run Eval

```bash
python -m personal_lifelog_rag.app.cli eval-private \
  --path private_eval/questions_20241224.yaml \
  --save-run
```

Run one case while tuning:

```bash
python -m personal_lifelog_rag.app.cli eval-private \
  --path private_eval/questions_20241224.yaml \
  --case-id date_20241224_summary
```

## Compare Runs

```bash
python -m personal_lifelog_rag.app.cli eval-compare \
  --before eval_outputs/eval_A.json \
  --after eval_outputs/eval_B.json
```

Use compare reports before and after ranking, VLM, OCR, or event-generation
changes.

## Do Not Commit Private Data

Keep these paths local:

- `private_eval/`
- `eval_outputs/`
- `private_config/`
- `data/`
- `models/`

Committed examples such as `configs/private_eval_20241224.example.yaml` must stay
anonymized and must not contain raw messages, real photo paths, exact GPS, or
private names.
