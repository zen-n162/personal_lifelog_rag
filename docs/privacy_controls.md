# Privacy Controls

This app stores sensitive local lifelog data, so privacy controls are designed
around reversible operations first.

## Principles

- External APIs are not used.
- Face crops, face embeddings, exact GPS, raw chat text, and local file paths stay local.
- Hide, detach, and soft delete are preferred over physical deletion.
- Hard deletion is intentionally narrow and requires explicit confirmation.
- Every executed privacy operation writes a row to `privacy_actions`.

## Hide, Detach, Delete

Hide:
removes an entity from search, QA, reports, and public display while keeping the
underlying row for local audit.

Detach:
removes links from a person to face clusters, LINE speakers, media, and events.
The person row remains.

Soft delete:
sets the person as hidden, non-searchable, not event-usable, and records
`deleted_at`. Related `media_people` and `event_people` links are hidden.

## CLI

```bash
python -m personal_lifelog_rag.app.cli privacy-audit --public
python -m personal_lifelog_rag.app.cli person-export --person-id PERSON_ID --mode public_redacted --dry-run
python -m personal_lifelog_rag.app.cli person-detach --person-id PERSON_ID --dry-run
python -m personal_lifelog_rag.app.cli person-delete --person-id PERSON_ID --soft --dry-run
python -m personal_lifelog_rag.app.cli face-delete-data --face-id FACE_ID --delete-crop --delete-embedding --dry-run
python -m personal_lifelog_rag.app.cli places hide --place-id PLACE_ID --yes
```

Run a DB backup before executing non-dry-run operations.

## Public Export

`public_redacted` person export uses public labels only. It does not include face
crops, face embeddings, exact coordinates, raw chat text, or original file paths.

## Public Report Safety

Public reports and portfolio HTML must not include:

- real person display names
- face crops or face embeddings
- exact coordinates
- raw chat text
- private local paths

After editing docs, reports, or README, regenerate the portfolio HTML and run
the public safety check.

## Private Eval Coverage

PR73 adds privacy-focused private eval cases:

- `privacy_audit` scans selected public artifacts for blocked patterns.
- `export_privacy` checks that public-redacted person exports omit private
  display names, face embedding fields, crop paths, and exact coordinates.
- `face_workflow_quality` verifies that face crops are not exposed as public
  artifacts.
- `line_person_link_quality` verifies that LINE speaker links are manual and
  user-verified.

These checks complement the standalone `privacy-audit --public` command and
the portfolio HTML safety checker.
