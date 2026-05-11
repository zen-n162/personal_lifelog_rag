# LINE Speaker and Person Linking

PR69 adds a manual bridge between LINE speaker names and user-created `persons`.
It is a private, local-only workflow. The app does not infer who a speaker is
from a face, a face cluster, or message content.

## Principles

- LINE speaker links are created only by explicit user action.
- Face clusters and LINE speakers are never matched automatically.
- Relationship labels such as partner, family, or friend are never inferred.
- LINE text stays local and is not sent to external APIs.
- Unlinking is supported, and links can be revised at any time.
- Public reports and portfolio HTML must not show private `display_name`
  values or raw LINE text.

## Data Model

`line_speaker_links` stores manual links:

- `chat_id`
- `speaker_name`
- `person_id`
- `source`: currently `manual`
- `confidence`: `1.0` for explicit user verification
- `verified_by_user`
- `verified_at`

`person_line_mentions` is reserved for future mention tracking. PR69 only uses
speaker links.

Person aliases can optionally store a LINE speaker name with source
`line_speaker`, but this is still created only through an explicit command or UI
action.

## CLI

List speaker names found in local LINE messages:

```bash
python -m personal_lifelog_rag.app.cli line-speakers list
```

Create a manual person label:

```bash
python -m personal_lifelog_rag.app.cli persons create \
  --name "人物テストA" \
  --public-name "人物A" \
  --privacy-level private
```

Link one LINE speaker to that person:

```bash
python -m personal_lifelog_rag.app.cli line-speakers link-person \
  --chat-id CHAT_ID \
  --speaker-name "SPEAKER_NAME" \
  --person-id PERSON_ID \
  --add-alias \
  --yes
```

Remove a link:

```bash
python -m personal_lifelog_rag.app.cli line-speakers unlink-person \
  --chat-id CHAT_ID \
  --speaker-name "SPEAKER_NAME" \
  --person-id PERSON_ID \
  --yes
```

Show existing links:

```bash
python -m personal_lifelog_rag.app.cli line-speakers show-links
```

Show non-binding suggestions from existing manual person names and aliases:

```bash
python -m personal_lifelog_rag.app.cli line-speakers suggest \
  --speaker-name "SPEAKER_NAME"
```

Suggestions never create links automatically.

## UI

The Face Review tab includes a LINE speaker link section:

1. Load `LINE speakers`.
2. Select or create a manual person label.
3. Enter the `chat_id` and `speaker_name` to link.
4. Optionally add the speaker name as a person alias.
5. Unlink when the mapping is wrong or no longer desired.

The UI reminder is intentional: the app does not infer identity or
relationships from faces or conversations.

## QA Scope

PR69 adds a narrow, cautious QA path:

```bash
python -m personal_lifelog_rag.app.cli qa "人物テストAとLINEした日は？"
```

The answer uses only `line_speaker_links` that the user created manually. It
returns date-level message counts and explicitly states that relationships are
not inferred.

Photo-person search and richer person summaries are intentionally left for
later PRs.

PR70 uses manual LINE speaker links as one possible source for `event_people`.
This only says that a linked LINE speaker contributed messages that support an
event. It does not prove co-presence or relationship status.

PR81 also lets the same manual speaker links contribute to:

- person-aware image and multimodal search ranking
- refreshed `event_people` rows
- private monthly person summaries
- private event summaries

LINE-only evidence is never treated as proof that the person appears in a
photo. It is labeled as LINE context.

## Public and Private Display

Private local views may show `display_name`.

Public outputs must:

- hide private `display_name`
- avoid raw LINE text
- avoid face crops and face embeddings
- use `public_name` or a generic anonymized label only when an explicit public
  display path is enabled

## Safety Notes

Do not use LINE speaker names as proof of identity. A speaker name can be a
nickname, a device label, or an exported chat label. Treat every link as a
manual local annotation that can be corrected or removed.
