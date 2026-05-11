# Face Embedding and Clustering

PR67 adds local face embedding and candidate face clustering. This is not face
recognition and it does not assign names.

## Scope

- Read reviewed or unreviewed rows from `face_detections`.
- Create a local embedding for each detected face crop.
- Store embeddings in `face_embeddings` as local SQLite BLOBs.
- Group similar embeddings into `face_clusters` and `face_cluster_members`.
- Treat every cluster as a `person_candidate_###` until a human reviews it.

The app does not infer identity, relationships, emotions, family status, or
LINE speaker links from face embeddings.

## Privacy Rules

- No external API, cloud model, or automatic model download is used.
- Face crops, thumbnails, and embeddings stay under local private storage and
  the local SQLite DB.
- `data/faces/`, including `data/faces/embeddings/`, is ignored by Git.
- Unreviewed clusters are not normal search, QA, event, report, or portfolio
  evidence.
- Public reports and portfolio HTML must not include face crops, embeddings,
  cluster IDs, names, or person labels.

## Database Tables

`face_embeddings` stores one embedding per `face_detections.id`.

- `embedding_model`
- `embedding_dim`
- `embedding_blob`
- `embedding_format`, currently `float32_numpy`
- `normalized`
- `status`
- `error_message`

`face_clusters` stores candidate groups only.

- `cluster_label`, such as `person_candidate_001`
- `representative_face_id`
- `face_count`
- `first_seen_at` / `last_seen_at`
- `clustering_method`
- `distance_threshold`
- `status` and `review_status`
- `privacy_level`, currently private

`face_cluster_members` links candidate clusters to face IDs.

## Engines

`opencv_sface` is the intended local real engine. It requires OpenCV support and
a locally configured model file. If the model file is not present, the engine
returns `engine_unavailable` and does not download anything.

`fake` is deterministic test-only infrastructure. It is useful for tests and
schema smoke checks, but fake embeddings should not be mixed into production
analysis.

## CLI

```bash
python -m personal_lifelog_rag.app.cli face-embedding-diagnostics \
  --config private_config/model_runtime.yaml

python -m personal_lifelog_rag.app.cli face-embed \
  --from 2024-12-01 \
  --to 2025-03-31 \
  --config private_config/model_runtime.yaml \
  --limit 100 \
  --dry-run

python -m personal_lifelog_rag.app.cli face-cluster \
  --from 2024-12-01 \
  --to 2025-03-31 \
  --distance-threshold 0.45 \
  --min-samples 2 \
  --dry-run

python -m personal_lifelog_rag.app.cli face-cluster-stats
python -m personal_lifelog_rag.app.cli face-cluster-show --status unreviewed --limit 20

python -m personal_lifelog_rag.app.cli update-face-cluster \
  --cluster-id CLUSTER_ID \
  --status rejected
```

Real clustering writes require `--yes`. Use `--replace` only when intentionally
discarding previous candidate clusters.

## Clustering

The implementation uses cosine distance over normalized embeddings. The
`dbscan_cosine` CLI option is implemented as a conservative connected-components
pass over a local radius-neighbor graph:

- embeddings with distance at or below the threshold are connected
- connected groups with at least `min_samples` become candidate clusters
- singleton faces remain unclustered
- cluster labels are generated as person candidates, not names

This keeps the behavior local and auditable. In PR75, common thresholds such as
`0.45` produced giant connected components, so the full-range YuNet/SFace
rollout selected a conservative `distance_threshold=0.10` and
`min_samples=3`. See `docs/pr75_yunet_face_embedding_cluster_full_range.md`.

For large full-range rebuilds:

```bash
python -m personal_lifelog_rag.app.cli face-embed \
  --from 2024-10-01 \
  --to 2026-05-31 \
  --config private_config/model_runtime.yaml \
  --engine opencv_sface \
  --detections-engine opencv_yunet \
  --status success \
  --only-existing-files \
  --batch-size 500 \
  --replace \
  --save-report

python -m personal_lifelog_rag.app.cli face-cluster \
  --from 2024-10-01 \
  --to 2026-05-31 \
  --method dbscan_cosine \
  --distance-threshold 0.10 \
  --min-samples 3 \
  --scope yunet_202410_202605 \
  --replace \
  --yes \
  --save-report
```

Run dry-runs and backups before writing to the DB.

## Review Boundary

Candidate clusters are not people by themselves. Manual person labels are
documented in `docs/face_review_people.md`; they require the user to type the
label and explicitly link it to a cluster. Automatic identity inference remains
out of scope.
