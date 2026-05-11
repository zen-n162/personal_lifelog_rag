# Person-Linked QA and Search

PR81 connects manually verified person links to Ask, image search, multimodal
search, monthly summaries, event summaries, UI services, and private eval.

## Principle

Only user-created links are used:

- `persons`
- `person_aliases`
- `person_face_clusters`
- `line_speaker_links`
- `media_people`
- `event_people`

The app does not infer a name from a face, infer a face from a LINE speaker, or
infer a relationship. Unreviewed face clusters and unverified persons are
ignored by normal QA, search, and reports.

## Ask Examples

Use anonymized names in shared docs and demos:

```bash
python -m personal_lifelog_rag.app.cli qa "人物AとLINEした日は？"
python -m personal_lifelog_rag.app.cli qa "人物Aとの通話はいつ？"
python -m personal_lifelog_rag.app.cli qa "人物Aが写っている写真はいつ？"
python -m personal_lifelog_rag.app.cli qa "人物Aとご飯を食べた日は？"
python -m personal_lifelog_rag.app.cli qa "人物Aと新宿に行ったのはいつ？"
```

The answer should say where each signal came from, such as LINE speaker links,
manual face-cluster links, `media_people`, `event_people`, place labels, or
activity evidence. It should use "可能性" or "候補" phrasing.

## Image Search

When a person name appears in an image query, the resolver checks manual person
records and adds person-aware ranking components:

- `person_score`
- `person_face_score`
- `person_line_score`
- `person_event_score`
- `related_persons`
- `person_evidence_types`

`media_people` is strong for photo queries. LINE-only evidence is treated as
same-day context and must not be described as a visual match.

## Monthly and Event Summaries

Private mode may show compact person-linked counts:

- linked LINE message counts
- linked call counts
- person-related events
- person-related media

Public mode hides this section or uses public/anonymized names. Face crops,
embeddings, exact GPS, and raw LINE text are not shown.

## UI

The Ask, Image Search, Multimodal Search, and Monthly Summary services expose
person-aware fields while preserving the same privacy rules. The Person/Face
review screens remain the place to create or adjust manual links.

## Forbidden Output

Do not output:

- automatic identity claims
- relationship labels
- "確実に一緒にいた"
- face crop paths in public mode
- face embeddings
- raw LINE excerpts in public artifacts
- exact GPS coordinates

## Rebuild Flow

After creating or editing manual person links, rebuild derived associations:

```bash
python -m personal_lifelog_rag.app.cli build-media-people \
  --from 2024-10-01 --to 2026-05-31 --replace --dry-run

python -m personal_lifelog_rag.app.cli build-event-people \
  --from 2024-10-01 --to 2026-05-31 --replace --dry-run
```

If the dry-run is reasonable, back up the DB and rerun with `--yes`.

