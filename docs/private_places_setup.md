# Private Places Setup

This app never uses online reverse geocoding. GPS coordinates from photos can
identify private homes, schools, workplaces, and other sensitive places, so
place labels are created and edited locally by the user.

## Why No Reverse Geocoding

Reverse geocoding APIs send coordinates to an external service. That conflicts
with the local-first privacy policy of this project. Keep GPS matching fully
local by using `private_config/places.yaml`.

## Create A Private Places File

```bash
python -m personal_lifelog_rag.app.cli places init-private
```

This creates `private_config/places.yaml` from the dummy example when it does
not already exist. It does not overwrite an existing file. The example contains
dummy coordinates only; edit it manually before using it with real data.

Do not commit:

- `private_config/places.yaml`
- `private_config/place_suggestions.yaml`
- `private_config/`

## Generate Local GPS Cluster Suggestions

```bash
python -m personal_lifelog_rag.app.cli cluster-places --all --radius-m 500 --min-points 5 --output private_config/place_suggestions.yaml
```

The generated suggestions use neutral labels such as `candidate_place_001` and
`候補地点001`. The app does not infer labels such as home, school, workplace, or
friend's house. Open the suggestion file locally, decide safe display names
yourself, then copy selected entries into `private_config/places.yaml`.

Suggestion entries default to:

- `privacy_level: "sensitive"`
- `show_exact_location: false`
- `category: "unknown"`

## Edit Safe Display Names

For sensitive places, prefer broad labels:

- `自宅周辺`
- `大学周辺`
- `職場周辺`
- `よく行く駅周辺`

Keep `show_exact_location: false` for homes, schools, workplaces, and private
addresses. The app will then show the display name and hide exact coordinates.

## Validate And Preview Redaction

```bash
python -m personal_lifelog_rag.app.cli places validate --path private_config/places.yaml
python -m personal_lifelog_rag.app.cli places redact-preview --path private_config/places.yaml
```

`redact-preview` shows whether exact coordinate display is hidden. Sensitive
places should show `exact coordinate display: hidden`.

## Assign Places To Events

Preview first:

```bash
python -m personal_lifelog_rag.app.cli assign-places --all --path private_config/places.yaml --dry-run
```

Apply after checking the preview:

```bash
python -m personal_lifelog_rag.app.cli assign-places --all --path private_config/places.yaml
```

`assign-places` updates `events.location_name` only. It does not modify
`media_items` or `line_messages`, and it does not overwrite
`event_overrides.location_name_override`.

## Check Results

```bash
python -m personal_lifelog_rag.app.cli place-stats
python -m personal_lifelog_rag.app.cli list-events --date 2024-12-24
python -m personal_lifelog_rag.app.cli qa "新宿に行ったのはいつ？"
```

Answers, event lists, search, and the UI prefer `location_name` display labels
over raw coordinates. Exact coordinates are not needed for normal review.
