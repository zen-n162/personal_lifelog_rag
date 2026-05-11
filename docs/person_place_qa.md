# Person and Place QA

PR71 adds a conservative QA layer that can answer questions using manually
reviewed person and place links. It is designed for private local inspection,
not automatic identity inference.

## Supported Questions

Person-oriented examples:

```bash
python -m personal_lifelog_rag.app.cli qa "人物AとLINEした日は？"
python -m personal_lifelog_rag.app.cli qa "人物Aが写っている写真はいつ？"
python -m personal_lifelog_rag.app.cli qa "人物Aとご飯を食べた日は？"
python -m personal_lifelog_rag.app.cli qa "人物Aと新宿に行ったのはいつ？"
python -m personal_lifelog_rag.app.cli qa "人物Aと一緒だった可能性がある日は？"
```

Place-oriented examples:

```bash
python -m personal_lifelog_rag.app.cli qa "新宿に行ったのはいつ？"
python -m personal_lifelog_rag.app.cli qa "新宿の写真はいつ？"
python -m personal_lifelog_rag.app.cli qa "2025年1月に行った場所は？"
```

## Person Resolution

The resolver checks only manually verified local records:

- `persons.display_name`
- `persons.public_name`
- `person_aliases.alias`
- `persons.aliases_json`
- `line_speaker_links.speaker_name` when the speaker was manually linked to a
  person

If multiple candidates match, the answer lists the candidates and does not pick
one automatically.

After PR81, the same resolver is used by Ask, image search, multimodal search,
monthly summary helpers, UI services, and private eval metadata.

## Place Resolution

The resolver checks:

- `places.display_name`
- `places.public_name`
- `places.aliases_json`
- `places.category`

Exact GPS coordinates are never displayed in QA output. Public mode uses
`public_name`, category labels, or a hidden-place label.

## Evidence Strength

QA results include metadata for later private evaluation:

- `resolved_person_id`
- `resolved_place_id`
- `evidence_types`
- `top_dates`
- `source_counts`
- `privacy_mode`
- `overclaim_flags`

Initial evidence rules:

- Weak: line-only, face/media-only, or place-only evidence.
- Medium: person plus event, place plus event, person plus media, or place plus
  media.
- Strong: person plus place plus event, person plus line plus media, or another
  manually verified combined evidence path.

Even strong evidence is phrased as a candidate or possibility.

## Answer Style

Allowed wording:

- "LINE上でやりとりがありました"
- "写真に写っている可能性があります"
- "同じイベントに関連している可能性があります"
- "場所ラベル由来の候補です"

Forbidden wording:

- "確実に一緒にいた"
- automatic relationship labels such as romantic partner, family, or friend
- emotion or intimacy claims
- identity claims derived only from faces

## Public and Private Modes

`qa --public` uses public-safe person and place labels. Private mode may show
local display names for manually verified persons and places, but still does not
show exact coordinates, face crops, face embeddings, or raw chat excerpts.

## UI Entry Point

The Multimodal Search UI includes lightweight person, place, and activity
filters. These filters help compose the search query while preserving the same
local-only data flow. Full person/place QA remains available through the CLI QA
route.

## Limitations

- Unverified persons and face clusters are ignored.
- LINE speaker links are manual only.
- Person-place answers are candidates, not proof of being together.
- Public reports should continue to avoid private names, face images, and exact
  location data.
- If several person records share the same name, QA returns an ambiguity
  message until the user merges, renames, or links the intended person.

## Private Eval

PR73 adds private eval cases for this layer:

- `place_qa`
- `monthly_place_summary`
- `person_line_qa`
- `person_photo_qa`
- `person_place_activity_qa`

These cases record resolved person/place ids, evidence types, top dates,
source counts, privacy mode, and overclaim flags. If no manually verified
person or reviewed place exists, the case can be marked skipped with an explicit
reason instead of failing the whole run.
