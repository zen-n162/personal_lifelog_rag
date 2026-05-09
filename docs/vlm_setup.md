# Local VLM Setup

This app only supports local image analysis. Do not use cloud VLM services or
any API that uploads private photos, screenshots, GPS-linked images, receipts,
tickets, or LINE-related content.

## What VLM Adds

VLM analysis stores cautious visual hints for imported photos:

- caption and short caption
- scene tags
- object tags
- activity tags
- location cues
- food cues
- rough people count and `people_present` flag
- safety flags
- engine/model metadata, confidence, status, and analysis version

Results are saved in `media_vlm`. Successful captions are mirrored to
`media_items.caption`, and compact tags are mirrored to
`media_items.analysis_json` so search and event generation can use them without
loading a model.

VLM output is treated as noisy evidence. Search and `qa` should say things like
"画像解析では、料理の可能性がある写真が見つかりました" rather than stating that
the activity definitely happened.

## Safety Rules

VLM prompts and post-processing are designed to keep only observable,
non-sensitive information.

The app must not identify people or infer:

- relationships such as lover, family, or friend
- emotions
- age, health, disability, religion, politics, occupation, or other sensitive
  traits
- names or account identities

If people are present, the stored signal should be limited to
`people_present` and a rough count when available.

## Engines

Built-in adapters:

- `noop`: safe no-op
- `fake`: deterministic test/development engine
- `ollama`: localhost-only adapter skeleton
- `transformers`: local-file-only Hugging Face image-to-text adapter
- `llama_cpp`: localhost-only placeholder boundary

Optional dependencies:

```bash
pip install -e ".[vlm]"
pip install -e ".[vlm-ollama]"
```

The project does not download models automatically. For Transformers, point the
model setting at a local model path. For Ollama, the URL must be localhost
only, for example `http://127.0.0.1:11434`.

## Commands

Check coverage:

```bash
python -m personal_lifelog_rag.app.cli vlm-stats
python -m personal_lifelog_rag.app.cli vlm-stats --from 2024-12-01 --to 2024-12-31
```

Preview a run without writing to SQLite:

```bash
python -m personal_lifelog_rag.app.cli analyze-images --date 2024-12-24 --dry-run --limit 5
```

Run with the fake engine for a local smoke test:

```bash
python -m personal_lifelog_rag.app.cli analyze-images --date 2024-12-24 --engine fake --limit 5 --force
```

Run with local Ollama:

```bash
export PERSONAL_LIFELOG_RAG_VLM_MODEL=llava
export PERSONAL_LIFELOG_RAG_OLLAMA_URL=http://127.0.0.1:11434
python -m personal_lifelog_rag.app.cli analyze-images --date 2024-12-24 --engine ollama --limit 10
```

Run with a local Transformers model path:

```bash
python -m personal_lifelog_rag.app.cli analyze-images --date 2024-12-24 --engine transformers --model /local/path/to/model --limit 10
```

Inspect compact results:

```bash
python -m personal_lifelog_rag.app.cli vlm-show --date 2024-12-24 --limit 10
python -m personal_lifelog_rag.app.cli vlm-show MEDIA_ID --full
```

Search image content:

```bash
python -m personal_lifelog_rag.app.cli image-search "ラーメン"
python -m personal_lifelog_rag.app.cli image-search "カフェ" --from 2024-12-01 --to 2024-12-31
python -m personal_lifelog_rag.app.cli qa "ラーメンを食べた写真はいつ？"
```

For Qwen3-VL / Qwen3-VL-Embedding model selection and benchmarking, see
`docs/vlm_model_selection.md`. The benchmark keeps caption generation and
text-to-image retrieval as separate roles:

```bash
python -m personal_lifelog_rag.app.cli vlm-model-info --config private_config/model_runtime.yaml
python -m personal_lifelog_rag.app.cli benchmark-qwen-multimodal --cases configs/vlm_benchmark.example.yaml --engine fake --save
```

For prompt templates and safety filtering, see
`docs/vlm_prompting_and_safety.md`:

```bash
python -m personal_lifelog_rag.app.cli vlm-prompt --template lifelog_structured_tags_v1
python -m personal_lifelog_rag.app.cli vlm-safety-check --text "彼女と楽しそうにご飯を食べている写真です"
```

## Search and Events

`search`, `qa`, and `image-search` include VLM captions and tags when
`media_vlm` rows exist. VLM evidence is shown as short photo-linked previews.

Event generation can use mirrored VLM tags as weak photo text. VLM alone should
not force a location or activity; it is strongest when it agrees with LINE, OCR,
GPS, or a place dictionary match.

## UI

The Gradio UI includes local VLM controls in the ingest/analysis area and an
image search tab. Event detail can show compact VLM evidence beside OCR and
photo evidence. Public sharing is disabled; keep the server bound to
`127.0.0.1`.

## Privacy Notes

- Do not commit `data/`, `models/`, `vlm_outputs/`, `eval_outputs/`,
  `private_eval/`, or `private_config/`.
- VLM captions may contain mistaken guesses. Review important results in the UI
  and correct events with overrides.
- Do not use remote endpoints. The Ollama and llama.cpp boundaries reject
  non-localhost URLs.
- CLI and UI output should stay compact; avoid dumping large caption JSON in
  logs.
