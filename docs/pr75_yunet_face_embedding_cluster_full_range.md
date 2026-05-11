# PR75 YuNet Face Embedding and Clustering Full Range

## Summary

PR75 rebuilt private face embeddings from YuNet face detections for
2024-10-01 through 2026-05-31, then regenerated unreviewed face cluster
candidates.

This remains a private review workflow. The app does not infer names,
relationships, emotions, or LINE speaker links from faces.

## Inputs

- Date range: 2024-10-01 to 2026-05-31
- Detection engine filter: `opencv_yunet`
- Detection status filter: `success`
- Embedding engine: `opencv_sface`
- Local-only model config: enabled
- External API usage: none
- Model auto-download: none

## Preflight

- DB process check: no blocking Gradio or app process was found.
- `lsof data/db/lifelog.sqlite`: no active holder was reported.
- Backup before embedding:
  `backups/lifelog_before_yunet_face_embedding_cluster_full_202410_202605_20260511_124454.sqlite`
- Backup before clustering:
  `backups/lifelog_before_yunet_face_cluster_full_202410_202605_20260511_133728.sqlite`

## Embedding Results

- YuNet success detections: 13,914
- Existing embeddings in selected rows before replace: 48
- Deleted selected old embeddings: 48
- Processed detections: 13,914
- Embedding success: 13,914
- Failed: 0
- Skipped: 0
- Engine unavailable: 0
- Report: `reports/face_embed_20260511_133257.json`

Note: only 4,149 detections had persisted crop paths. To cover all YuNet
success detections, the run allowed `face-embed` to recreate temporary crops
from the original local image and bbox where a persisted crop was absent.

## Clustering Dry-Run Comparison

| distance_threshold | min_samples | cluster candidates | largest cluster | singleton/outlier count | judgment |
|---:|---:|---:|---:|---:|---|
| 0.45 | 2 | 180 | 11,978 | 1,456 | rejected, giant cluster |
| 0.40 | 2 | 358 | 9,980 | 2,941 | rejected, giant cluster |
| 0.45 | 3 | 63 | 11,978 | 1,690 | rejected, giant cluster |
| 0.30 | 2 | 705 | 5,515 | 6,331 | rejected, giant cluster |
| 0.25 | 2 | 790 | 3,432 | 8,064 | rejected, giant cluster |
| 0.20 | 2 | 826 | 1,246 | 9,712 | rejected, still too large |
| 0.15 | 2 | 692 | 506 | 11,275 | rejected, still too large |
| 0.15 | 3 | 208 | 506 | 12,243 | rejected, still too large |
| 0.10 | 2 | 450 | 22 | 12,651 | acceptable but more small clusters |
| 0.10 | 3 | 130 | 22 | 13,291 | selected |
| 0.70 | 2 | not executed | expected larger than 0.45 | n/a | unsafe after giant-cluster evidence |
| 0.75 | 2 | not executed | expected larger than 0.45 | n/a | unsafe after giant-cluster evidence |
| 0.75 | 3 | not executed | expected larger than 0.45 | n/a | unsafe after giant-cluster evidence |

The selected setting is intentionally conservative:

```text
method=dbscan_cosine
distance_threshold=0.10
min_samples=3
scope=yunet_202410_202605
```

## Cluster Results

- Selected embeddings: 13,914
- Cluster candidates written: 130
- Cluster members written: 623
- Singleton/outlier faces: 13,291
- Largest cluster size: 22
- Cluster report: `reports/face_cluster_20260511_133737.json`

All generated clusters are `unreviewed`. They are not used by normal QA,
search, event generation, public reports, or portfolio HTML until a human
reviews and links them.

## Validation

- `db-check --strict`: PASS
- Face embedding orphan refs: 0
- Empty success embedding blobs: 0
- Invalid embedding dimensions: 0
- Face cluster orphan refs: 0
- Duplicate cluster members: 0
- Invalid cluster member distance: 0
- `privacy-audit --public`: PASS

## QA and Privacy Boundary

Unreviewed face clusters were not promoted into person QA. A query for a
specific public placeholder person without a manual person link returned that no
manual verified person or LINE speaker link was found.

Generic image-search-style queries such as "who is in the photo" can still route
through multimodal image search, but they do not use face clusters to identify a
person.

## Known Limitations

- Many YuNet detections do not currently have persisted crop thumbnails, even
  though embeddings can be created from original image plus bbox.
- SFace clustering produced giant connected components at common thresholds
  such as 0.45, so PR75 selected a conservative threshold.
- Cluster quality still requires human review in the Face Review UI.

## Next Steps

- In Face Review, inspect the largest and highest-quality unreviewed clusters.
- Mark bad clusters or bad detections where needed.
- Manually create/link person labels only after visual confirmation.
- Link LINE speakers manually after person labels are confirmed.
- Then run `build-media-people` and `build-event-people` if person links are
  ready.
- Consider a later crop backfill job so all successful detections have review
  thumbnails.
