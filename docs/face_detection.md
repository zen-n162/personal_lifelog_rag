# Face Detection

PR66 adds local face detection storage and review. This is not face recognition.

## Scope

- Detect face bounding boxes in local photos.
- Store bbox, landmarks when available, detection score, and local crop paths.
- Save optional crops under `data/faces/crops/` and thumbnails under `data/faces/thumbnails/`.
- Review detections in CLI or the localhost UI.

The app does not automatically name people, link faces to LINE speakers, infer
relationships, infer emotions, or answer "who is this" questions.

## Privacy Rules

- No external API or geocoding service is used.
- Face crops and thumbnails are private local artifacts.
- Unreviewed detections are not used by normal search, QA, event generation, or
  public reports.
- Public reports and portfolio HTML must not include face crops.
- `data/faces/` is ignored by Git.

## CLI

```bash
python -m personal_lifelog_rag.app.cli face-diagnostics
python -m personal_lifelog_rag.app.cli face-diagnostics \
  --config private_config/model_runtime.yaml

python -m personal_lifelog_rag.app.cli face-detect \
  --date 2024-12-24 \
  --limit 20 \
  --engine opencv_haar \
  --dry-run

python -m personal_lifelog_rag.app.cli face-detect \
  --date 2024-12-24 \
  --limit 20 \
  --engine opencv_yunet \
  --config private_config/model_runtime.yaml \
  --dry-run

python -m personal_lifelog_rag.app.cli face-stats
python -m personal_lifelog_rag.app.cli face-show --status success --limit 20
python -m personal_lifelog_rag.app.cli face-review-queue --limit 20

python -m personal_lifelog_rag.app.cli update-face-detection \
  --face-id FACE_ID \
  --review-status bad_detection
```

Use `--save-crops` only when you want local private review crops. Missing
original files are skipped.

## Engines

- `opencv_yunet`: Uses OpenCV `FaceDetectorYN` with a local YuNet ONNX model.
  This is the preferred detector when a model file is available because it is a
  small DNN detector and is much less likely than Haar to mistake round objects
  or clothing for faces. The model path must be configured manually; the app
  never downloads it.
- `opencv_haar`: Uses an installed OpenCV Haar cascade. It is useful as a
  fallback because OpenCV often bundles the cascade XML, but accuracy is lower.
  The implementation applies a conservative eye-feature filter to reduce false
  positives, which can also miss small or side/back-facing faces.
- `opencv_sface`: Used by `face-embed`, not by `face-detect`. It uses OpenCV
  `FaceRecognizerSF` with a local SFace ONNX model to create private face
  embeddings after detection.
- `scrfd` / InsightFace-style detectors: not enabled in this local-first PR.
  They can be considered later as optional local engines, but only with manually
  provided model files and no automatic download.
- `fake`: Deterministic test-only detector.
- `noop`: Intentionally unavailable.

If OpenCV, a local cascade, or a configured YuNet/SFace model is missing,
diagnostics explain the reason and write paths stay local. `face-detect` records
`engine_unavailable` without crashing when a real run is requested.

Example local runtime config:

```yaml
face_detection:
  engine: "opencv_yunet"
  model_path: "models/face/face_detection_yunet.onnx"
  local_only: true
  score_threshold: 0.85
  nms_threshold: 0.3
  top_k: 5000
  max_input_size: 1280

face_embedding:
  engine: "opencv_sface"
  model_path: "models/face/sface.onnx"
  embedding_dim: 128
  local_only: true
  normalize: true
```

## UI

The Gradio UI includes a Face Review tab. It shows detection rows, bbox details,
optional private thumbnails, and review buttons:

- Accept
- Bad detection
- Reject
- Hide from face workflow

The tab is for local review only. It does not assign names.

## Next Work

Face embedding and candidate clustering are documented in
`docs/face_embedding_clustering.md`. Those clusters are still not names and are
not used by normal QA/search/report paths until reviewed in a future manual
labeling workflow. Automatic identity inference remains out of scope.

## PR75 Full-Range YuNet Follow-Up

YuNet detections from 2024-10-01 through 2026-05-31 were used to rebuild local
SFace embeddings and unreviewed face cluster candidates. Some successful
detections do not have persisted crop thumbnails, so `face-embed` can recreate a
temporary crop from the original local image and bbox for embedding generation.
For easier review, a later crop-backfill job can persist missing review
thumbnails without changing the privacy boundary.
