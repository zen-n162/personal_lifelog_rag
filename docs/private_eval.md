# Private Eval

Private eval is a local regression suite for checking answer quality, ranking,
privacy behavior, and safety wording without sending data to external services.
The YAML files under the local private eval folder are not intended for public
sharing.

## Case Types

Existing cases cover date QA, routed QA, monthly summaries, multimodal search,
image search, VLM quality, OCR quality, event quality, and call search.

PR73 adds these local-only case types:

- `place_qa`: evaluates questions such as "新宿に行ったのはいつ？" and checks that exact coordinates are not shown.
- `monthly_place_summary`: evaluates questions such as "2025年1月に行った場所は？".
- `person_line_qa`: evaluates manually linked LINE speaker searches.
- `person_photo_qa`: evaluates manually verified person-photo links.
- `person_place_activity_qa`: evaluates conservative person/place/activity candidate answers.
- `face_workflow_quality`: checks face workflow counts and public face crop exposure.
- `line_person_link_quality`: checks that LINE speaker links are manual and verified.
- `privacy_audit`: checks public artifacts for forbidden private patterns.
- `export_privacy`: checks public-redacted person exports.

## Skip Conditions

Person and place data may not exist on every local database. These fields make
that explicit:

- `allow_skip_if_no_person`
- `allow_skip_if_no_verified_person`
- `allow_skip_if_no_place`
- `allow_zero`

Skipped cases are reported as `skipped`, not `failed`, with a reason.

## Privacy Checks

Public-output cases can check for:

- private person display names
- exact GPS-like coordinates
- face crop paths
- face cluster identifiers
- face embedding fields
- raw chat excerpts
- local private configuration references
- local raw-data references
- local absolute paths
- `file://`

Public checks should prefer public labels, categories, or anonymized names.

## Overclaim Detection

Person-related eval cases add conservative overclaim checks for phrases such as:

- "確実に一緒にいた"
- "恋人"
- "家族"
- "親密"
- "顔認証で確定"
- "本人確定"

The expected answer style is "可能性", "候補", or "LINE上でやりとりがありました".
The app must not infer relationships, identity, emotion, or intimacy.

## Template Generation

```bash
python -m personal_lifelog_rag.app.cli make-private-eval-template \
  --include-people \
  --include-places \
  --include-privacy \
  --output private_eval/questions_people_places.yaml
```

You can also combine these flags with `--date` to include the existing date
baseline cases.

## Running

```bash
python -m personal_lifelog_rag.app.cli eval-private \
  --path private_eval/questions_people_places.yaml \
  --save-run
```

Review skipped cases before treating the run as a full regression pass.

## PR81 Person-Linked Checks

Person-linked eval cases can check that Ask uses manual `line_speaker_links`,
`media_people`, and `event_people` without overclaiming. Expected metadata
includes:

- `resolved_person_id`
- `source_counts`
- `evidence_types`
- `top_dates`
- `overclaim_flags`

If a manually verified person is not available, person cases should skip with a
reason instead of failing.
