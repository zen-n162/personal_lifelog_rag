# Person Delete And Export

Person controls are manual and local. The app never infers identity from a face,
never links LINE speakers automatically, and never infers relationships.

## Export

Private export is for local inspection. Public-redacted export is for sharing
safe aggregate information.

```bash
python -m personal_lifelog_rag.app.cli person-export \
  --person-id PERSON_ID \
  --output reports/person_export_PERSON_ID.json \
  --mode public_redacted \
  --dry-run
```

Public-redacted export includes:

- a public person label
- counts of linked LINE speakers, face clusters, media links, and event links
- date ranges and counts for LINE activity

It excludes:

- private display names
- raw LINE text
- face crop paths
- face embeddings
- exact coordinates
- original media file paths

## Detach

Detach removes person links while preserving the person row:

```bash
python -m personal_lifelog_rag.app.cli person-detach --person-id PERSON_ID --dry-run
python -m personal_lifelog_rag.app.cli person-detach --person-id PERSON_ID --yes
```

Targets:

- face cluster links
- person aliases
- LINE speaker links
- media_people
- event_people
- person line mentions

## Soft Delete

Soft delete hides the person from normal retrieval and public outputs:

```bash
python -m personal_lifelog_rag.app.cli person-delete --person-id PERSON_ID --soft --dry-run
python -m personal_lifelog_rag.app.cli person-delete --person-id PERSON_ID --soft --yes
```

The row remains for audit, but the person is not searchable, not event-usable,
hidden, and marked with a deletion timestamp.

## Face Data Removal

Face data can be removed independently:

```bash
python -m personal_lifelog_rag.app.cli face-delete-data \
  --face-id FACE_ID \
  --delete-crop \
  --delete-embedding \
  --dry-run
```

This is for local private data only. Face crops and embeddings are not public
portfolio artifacts.
