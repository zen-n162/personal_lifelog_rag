# Multimodal Search

`personal_lifelog_rag` keeps image search local. It does not call OpenAI,
cloud OCR, cloud VLM, reverse geocoding, or cloud embedding APIs.

## Roles

- Qwen3-VL: turns local photos into cautious captions, tags, food cues,
  location cues, and event hints stored in `media_vlm`.
- Qwen3-VL-Embedding: maps local images and text queries into a shared search
  space. Embeddings are stored locally in `media_embeddings`.

Qwen3-VL is for explanation. Qwen3-VL-Embedding is for retrieval. They are
complementary, not substitutes.

## Table

`media_embeddings` stores local vectors:

- `media_id`
- `embedding_type`: `image`, `caption`, `ocr`, or `combined_text`
- `embedding_model`
- `embedding_dim`
- `embedding` as a local BLOB
- `embedding_format`: currently `float32_numpy` or `json`
- `source_text` for text-derived rows
- `status`: `pending`, `success`, `skipped`, `failed`, or `engine_unavailable`

The current repository layer is SQLite-only. It is deliberately separated so a
future FAISS, Chroma, or Qdrant backend can be added behind the same service
boundary.

## Build

Use the fake engine for tests and smoke checks:

```bash
python -m personal_lifelog_rag.app.cli build-image-embeddings --date 2024-12-24 --engine fake --limit 10 --force
python -m personal_lifelog_rag.app.cli build-text-embeddings --date 2024-12-24 --engine fake --type combined_text --force
python -m personal_lifelog_rag.app.cli embedding-stats
```

For real Qwen3-VL-Embedding, create `private_config/model_runtime.yaml`
manually. The app never downloads models automatically.

## Search

```bash
python -m personal_lifelog_rag.app.cli multimodal-search "ご飯を食べた写真" --backend hybrid --limit 10
python -m personal_lifelog_rag.app.cli image-search "カフェ" --backend hybrid --limit 10
python -m personal_lifelog_rag.app.cli qa "ご飯を食べた写真はいつ？"
```

Backends:

- `sql`: OCR, VLM text/tags, filenames, events, and LINE text.
- `embedding`: local embedding similarity only.
- `hybrid`: embedding candidates plus SQL/OCR/VLM/LINE/events/places reranking.

## Ranking

Hybrid results include score components:

- `embedding_score`
- `vlm_text_score`
- `ocr_score`
- `line_score`
- `event_score`
- `place_score`
- `override_boost`
- `safety_penalty`
- `final_score`

Evidence strength is conservative:

- `weak`: embedding-only, VLM-only, OCR-only with weak context, or one LINE hit.
- `medium`: embedding+VLM, embedding+OCR, VLM+LINE, OCR+LINE, or an event.
- `strong`: multiple independent sources such as embedding+VLM+OCR+LINE, or a
  verified/pinned event.

Even strong results should be worded as candidates. The app avoids phrases such
as "確実に食べた" or "確実に行った" when the claim is based on image analysis or
embedding similarity.

## Private Eval

`multimodal_search` cases can check expected dates, evidence types, overclaim
phrases, and embedding-only rank limits:

```yaml
cases:
  - id: mm_search_food_001
    type: multimodal_search
    query: "ご飯を食べた写真"
    expected_evidence_types_any:
      - "embedding"
      - "vlm"
      - "ocr"
      - "event"
    should_not_include:
      - "確実に食べた"
```

Keep real eval files under ignored `private_eval/` or `eval_outputs/`.

## UI

The Gradio UI has a `Multimodal Search` tab with query, backend, date range, and
limit controls. Results show thumbnails, OCR/caption snippets, score, evidence
types, and evidence strength without exposing exact GPS.

