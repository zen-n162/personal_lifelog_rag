# Place Review UI

The Place Review tab is a local-only workflow for turning GPS clusters into a
small reviewed place dictionary. It never calls geocoding services and does not
send coordinates to external map APIs.

## Launch

```bash
python -m personal_lifelog_rag.app.cli ui
```

The UI must stay on `127.0.0.1`; do not use `share=True`.

## Review Flow

1. Open `Place Review / 場所レビュー`.
2. Filter clusters by date range, status, privacy, category, text, or minimum
   point count.
3. Select a `cluster_id`.
4. Inspect representative thumbnails, related media, related events, and dates.
5. Create or update a place label.
6. Link the cluster to the place.
7. Mark the cluster accepted or rejected.
8. Reassign events/media after label changes.

Exact latitude and longitude are hidden by default. The private detail checkbox
is only for local inspection and should not be used for public screenshots.

## Place Fields

- `display_name`: local/private label.
- `public_name`: safe label for public reports.
- `category`: broad type such as station, cafe, restaurant, travel, shop, or
  event_venue.
- `privacy_level`: `private`, `public_label`, or `public_hidden`.
- `aliases`: local search synonyms.
- `manual_verified`: human-confirmed place label.

Manual place labels take priority over GPS-only inference. The app does not
automatically infer sensitive labels such as home or lab.

## CLI Equivalents

```bash
python -m personal_lifelog_rag.app.cli places list-clusters --status unreviewed --limit 20
python -m personal_lifelog_rag.app.cli places show-cluster --cluster-id CLUSTER_ID
python -m personal_lifelog_rag.app.cli places create --name "駅周辺" --public-name "駅周辺" --category station --privacy-level public_label
python -m personal_lifelog_rag.app.cli places link-cluster --place-id PLACE_ID --cluster-id CLUSTER_ID --yes
python -m personal_lifelog_rag.app.cli places update --place-id PLACE_ID --name "駅周辺" --privacy-level public_label
python -m personal_lifelog_rag.app.cli places add-alias --place-id PLACE_ID --alias "駅名"
python -m personal_lifelog_rag.app.cli places reject-cluster --cluster-id CLUSTER_ID --yes
python -m personal_lifelog_rag.app.cli assign-places --from 2025-01-01 --to 2025-01-31 --yes
```

## Public/Private Display

Private reports may show `display_name` for local inspection, but still avoid
exact coordinates by default.

Public reports and portfolio HTML must use:

- `public_name`
- broad category
- `非公開の場所`

They must not show exact GPS, private display names, raw photo paths, or private
configuration.
