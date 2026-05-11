# Privacy and Safety

## Local-Only Design

This project is designed for private lifelog data. The core operating rule is:
photos, chat exports, GPS metadata, OCR text, VLM outputs, embeddings, and
evaluation artifacts stay on the local machine.

The system does not require:

- OpenAI API
- cloud OCR
- cloud VLM
- cloud embedding API
- reverse geocoding API

## Files That Must Not Be Published

Do not publish:

- local photo or chat exports
- local SQLite databases
- generated thumbnails
- model weights
- private configuration files
- private eval files
- generated reports that include private details
- backups
- evaluation outputs

The public repository should contain source code, dummy examples, tests with
synthetic data, and documentation only.

## Location Privacy

GPS coordinates are sensitive. The app avoids exposing exact coordinates in
answers and UI. Place names can be manually assigned through a private place
dictionary, but the app does not use external reverse geocoding.

Sensitive places should be displayed as broad names such as "near a registered
place" or "sensitive place" rather than exact coordinates.

The DB-backed place workflow keeps exact location points and cluster centroids
as private local records. Public reports and portfolio HTML should use only a
reviewed public place label, a broad category, or a generic private-place
placeholder. Home, lab, and similarly sensitive categories should not expose a
specific display name in public mode.

Place Review lets the user assign labels manually. Manual labels take priority
over GPS-only inference, but the app does not infer sensitive labels such as
home or lab automatically. Public mode must use `public_name`, category, or
`非公開の場所`; exact latitude/longitude and private display names stay local.

## Face Detection Privacy

Face detection is local bbox review only. The app may store detection boxes and
optional private crops, but it does not identify people, infer relationships,
infer emotions, or answer identity questions. Unreviewed face detections are not
normal QA/search/report evidence. Public reports and portfolio HTML must not
include face crops or face thumbnails.

Face embeddings and candidate face clusters are more sensitive than bbox
detections because they can support person matching. They are stored only in the
local private DB, are never exported to public reports, and are never used for
normal search, QA, or event generation until a future explicit human-review
workflow allows it. Candidate clusters use labels such as `person_candidate_001`
rather than names.

Manual person labels are user-entered only. The app does not guess names from
faces, does not link face clusters to LINE speakers automatically, and does not
infer relationships. Public mode must use `public_name`, an anonymized alias, or
hide the person entirely; private `display_name` values stay out of public
reports and portfolio HTML.

LINE speaker to person links are also manual-only. A LINE sender name can be
linked to a person only through the `line-speakers link-person` CLI or the local
Face Review UI. The app does not infer identity from speaker names, face
clusters, or message content, and the LINE speaker QA path reports only
date-level message counts with a no-relationship-inference disclaimer.

`media_people` and `event_people` are private integration tables. They may use
accepted, manually verified face clusters and manually linked LINE speakers, but
they are not relationship evidence. Public reports and portfolio HTML must keep
private person names, face crops, face embeddings, and face cluster identifiers
out of the output.

Person-linked QA and search use only manual links from `persons`,
`person_face_clusters`, and `line_speaker_links`. Public mode hides private
display names and never publishes face crops, face embeddings, or raw chat
content. Answers must avoid relationship inference and must phrase face/LINE
evidence as candidates or context.

## VLM Safety Filter

Qwen3-VL output is filtered before use. The prompt and safety layer prohibit:

- identifying people
- guessing names
- inferring relationships such as partner, family, friend, or coworker
- inferring emotions
- inferring age, health, disability, religion, politics, nationality,
  sexuality, or other sensitive traits
- claiming that an image proves an activity

If people are present, the output should only record a generic safety flag such
as `people_present` and, when visually obvious, a rough count.

## Evidence Strength

The ranking system distinguishes weak, medium, and strong evidence:

- VLM-only or embedding-only evidence is weak.
- VLM plus event, OCR, or LINE evidence can be medium.
- Multiple independent evidence types, or human verification, can be strong.

Even strong evidence is phrased cautiously in user-facing answers.

## OCR Redaction

OCR can capture private text such as phone numbers, email addresses, URLs,
long numeric IDs, postal-code-like strings, and address-like fragments. CLI/UI
preview paths redact or shorten OCR text. Raw OCR text is a local DB artifact and
should not be copied into public material.

## Report Modes

Public report mode should:

- avoid raw LINE text
- avoid exact GPS
- avoid file paths
- avoid real names
- avoid photos or thumbnails by default
- anonymize example queries and evidence descriptions

Private report mode can include more operational detail for local inspection,
but should still avoid bulk raw text and exact coordinates.

## Human Review

VLM Review lets the user mark image analysis results as:

- accepted
- rejected
- wrong
- hidden
- not searchable
- not usable for event generation

Rejected, wrong, hidden, and not-searchable results are excluded from normal
search. Not-event-usable results are excluded from event generation.

## Person and Place QA

Person/place QA uses only manual links:

- manually verified persons
- manually linked LINE speakers
- manually accepted face clusters propagated into `media_people`
- manually reviewed place labels

The system does not infer relationships, identity, or emotions. Public mode
hides private person names and private place labels, and QA output never prints
exact coordinates.

## Privacy Controls

PR72 adds local controls for export, hide, detach, and soft delete:

- `privacy-audit --public` checks generated public artifacts.
- `person-export` supports private and public-redacted exports.
- `person-detach` removes face, LINE, media, and event links while preserving
  the person row.
- `person-delete --soft` hides a person from normal retrieval without physical
  deletion.
- `face-delete-data` removes local face crops and/or embeddings.
- `places hide` removes a place label from search and public output.

Executed operations are recorded in `privacy_actions`. Use `--dry-run` before
changing the DB and back up the DB before any executed privacy operation.

## Publication Checklist

Before using the project in a portfolio:

1. Use public report mode.
2. Check generated Markdown manually.
3. Remove raw chat, exact GPS, file paths, and real names.
4. Do not include screenshots that show private images or message text.
5. Describe results as aggregate metrics and anonymized examples.

## Face Embedding and Cluster Privacy

Face embeddings and face clusters are private diagnostic/review artifacts:

- embeddings remain SQLite BLOBs and are never exported to public reports
- face crops and thumbnails remain under local private storage
- unreviewed face clusters are not used as person evidence in normal QA/search
- public reports and portfolio HTML must not include face crops, face IDs,
  cluster IDs, embeddings, or inferred names
- manual person labels are required before any person-related workflow can use
  a face cluster

PR75 rebuilt full-range YuNet/SFace embeddings and unreviewed clusters, but did
not promote any cluster into a person identity.
