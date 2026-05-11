# Person Media and Event Integration

PR70 adds `media_people` and `event_people`, which propagate manually verified
person labels into private media and event review surfaces.

This is not automatic identity inference. The app only uses person links that a
user has already verified manually.

## Tables

### media_people

Links a `media_items` row to a `persons` row.

Sources:

- `face_cluster`: derived from a manually linked `person_face_clusters` row
- `manual`: reserved for future direct user edits

Rules:

- The person must be manually verified.
- The face cluster must be accepted and user verified.
- Rejected or bad face clusters are excluded.
- Rejected or bad face detections are excluded.

### event_people

Links an event to a person.

Sources:

- `face`: event photo evidence has `media_people`
- `line_speaker`: event LINE evidence has a manually linked LINE speaker
- `combined`: the same event/person has both face and LINE speaker evidence
- `line_mention`: reserved for future manually reviewed mention extraction
- `manual`: reserved for future direct event edits

## Confidence

Initial confidence values are deliberately conservative:

- `media_people` from verified face cluster: `0.85`
- `event_people` from face only: `0.60`
- `event_people` from LINE speaker only: `0.70`
- `event_people` from face plus LINE speaker in the same event: `0.90`
- manual: `1.0`

Even high-confidence person links are not relationship evidence. The app must
not infer partner, family, friend, closeness, or co-presence claims from these
rows.

## CLI

Preview media-person links:

```bash
python -m personal_lifelog_rag.app.cli build-media-people \
  --from 2024-12-01 \
  --to 2025-03-31 \
  --dry-run
```

Write media-person links:

```bash
python -m personal_lifelog_rag.app.cli build-media-people \
  --from 2024-12-01 \
  --to 2025-03-31 \
  --yes
```

Preview event-person links:

```bash
python -m personal_lifelog_rag.app.cli build-event-people \
  --from 2024-12-01 \
  --to 2025-03-31 \
  --dry-run
```

Show stats:

```bash
python -m personal_lifelog_rag.app.cli people-stats
```

Show event people:

```bash
python -m personal_lifelog_rag.app.cli event-people-show --date 2024-12-24
```

Show media people:

```bash
python -m personal_lifelog_rag.app.cli media-people-show --date 2024-12-24 --limit 20
```

Use `--public` on display commands to preview anonymized labels.

After PR81, rebuilding `media_people` and `event_people` is the main refresh
step after editing person-face cluster links or LINE speaker links. Ask, Image
Search, Multimodal Search, Monthly Summary, and private eval read these derived
links.

## Private and Public Display

Private local views may show `display_name` for manually verified persons.

Public reports and portfolio HTML must not show:

- private person display names
- face crops
- face embeddings
- face cluster identifiers
- raw LINE text

Public mode should use `public_name`, a generic anonymized label, or hide the
person entirely according to `privacy_level`.

Search and QA use only manually verified people and never infer relationships.

## DB Check

`db-check` validates:

- orphan `media_people` and `event_people` references
- invalid sources
- invalid confidence values
- `media_people` rows derived from unverified or rejected clusters
- `media_people` rows derived from rejected or bad face detections
- unverified person links

Strict mode treats orphan references, invalid sources/confidence, and rejected
face-derived links as severe issues.

## Roadmap

PR70 creates the evidence graph foundation. PR71 builds on it with explicit,
opt-in person/place QA using only these manually verified links. Unverified face
clusters, unverified persons, and unlinked LINE speakers remain excluded.
