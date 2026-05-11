# Location Points and Place Clustering

This app can use detailed GPS metadata inside the local SQLite database while
keeping public output limited to safe place labels. No reverse geocoding,
online map API, or external lookup is used.

## Data Model

`location_points` stores private position points derived from local records.
The first supported source is EXIF GPS from `media_items`. Exact coordinates
stay in the private DB and are not printed in public reports.

`place_clusters` stores local clusters of nearby location points. The current
clustering method is a small local distance-based algorithm using Haversine
meters. It is intentionally dependency-light and offline.

`places` stores user-reviewed labels. Each place can have:

- `display_name` for local/private use
- `public_name` for safe public output
- `category` such as station, cafe, restaurant, travel, shop, event venue, or
  other
- `privacy_level`: private, public_label, or public_hidden
- optional aliases
- optional cluster link

`media_places` links photos to reviewed places. `event_places` links events to
reviewed places.

## Privacy Levels

Location points and clusters default to `exact_private`. They are technical
records for local analysis only.

Places use:

- `private`: usable locally, but not a public label.
- `public_label`: may use `public_name` in public reports.
- `public_hidden`: shown as a generic private place in public reports.

Home, lab, or other sensitive categories should use private or public_hidden.

## CLI Workflow

Build private location points from GPS-tagged media:

```bash
python -m personal_lifelog_rag.app.cli build-location-points \
  --from 2024-12-01 \
  --to 2025-03-31 \
  --dry-run

python -m personal_lifelog_rag.app.cli build-location-points \
  --from 2024-12-01 \
  --to 2025-03-31 \
  --yes
```

Cluster existing location points:

```bash
python -m personal_lifelog_rag.app.cli cluster-places \
  --from 2024-12-01 \
  --to 2025-03-31 \
  --eps-meters 100 \
  --min-samples 3 \
  --dry-run
```

Review clusters and add labels:

```bash
python -m personal_lifelog_rag.app.cli places list-clusters --status unreviewed --limit 20
python -m personal_lifelog_rag.app.cli places show-cluster --cluster-id CLUSTER_ID

python -m personal_lifelog_rag.app.cli places create \
  --name "駅周辺" \
  --public-name "駅周辺" \
  --category station \
  --privacy-level public_label

python -m personal_lifelog_rag.app.cli places link-cluster \
  --place-id PLACE_ID \
  --cluster-id CLUSTER_ID \
  --yes

python -m personal_lifelog_rag.app.cli places update \
  --place-id PLACE_ID \
  --name "駅周辺" \
  --category station \
  --privacy-level public_label

python -m personal_lifelog_rag.app.cli places add-alias \
  --place-id PLACE_ID \
  --alias "駅名"

python -m personal_lifelog_rag.app.cli places reject-cluster \
  --cluster-id CLUSTER_ID \
  --yes
```

Assign reviewed places to photos and events:

```bash
python -m personal_lifelog_rag.app.cli assign-places \
  --from 2024-12-01 \
  --to 2025-03-31 \
  --dry-run

python -m personal_lifelog_rag.app.cli assign-places \
  --from 2024-12-01 \
  --to 2025-03-31 \
  --yes
```

## Existing YAML Dictionary Compatibility

The older local YAML dictionary workflow remains available. Use
`assign-places --path ...` when you want to match events directly against a
local place dictionary file instead of the DB-backed place tables.

## Search, QA, and Reports

Reviewed labels can improve event location names, search, QA, monthly summary,
and reports. Public report mode should only show `public_name`, category-level
labels, or a generic hidden-place label. It must not show exact coordinates or
sensitive private place names.

PR71 adds place-aware QA for reviewed labels:

```bash
python -m personal_lifelog_rag.app.cli qa "新宿に行ったのはいつ？"
python -m personal_lifelog_rag.app.cli qa "新宿の写真はいつ？"
python -m personal_lifelog_rag.app.cli qa "2025年1月に行った場所は？"
```

The resolver uses display names, public names, aliases, and categories. Ambiguous
matches are shown as candidates instead of being auto-selected.

The Gradio `Place Review / 場所レビュー` tab provides the same workflow with
cluster filters, representative thumbnails, related events, place creation,
cluster linking, alias editing, status updates, and place reassignment. See
`docs/place_review_ui.md`.

## Integrity Checks

`db-check` validates:

- invalid location coordinates
- orphan media/event/place references
- duplicate media location points
- invalid privacy levels
- cluster centroid and radius sanity
- event/media place link confidence

Run:

```bash
python -m personal_lifelog_rag.app.cli db-check --strict
```

## Notes

- Do not use reverse geocoding APIs.
- Do not publish raw GPS coordinates.
- Review clusters manually before assigning meaningful labels.
- Treat inferred places as candidates unless manually verified.
