# VLM Model Selection

This app uses Qwen3-VL and Qwen3-VL-Embedding for different local-only jobs.
They are complementary, not interchangeable.

## Role Split

- Qwen3-VL is the caption and visual-understanding model.
  It should produce cautious captions, scene tags, object tags, activity
  candidates, food cues, and location cues for `media_vlm`.
- Qwen3-VL-Embedding is the retrieval/index model.
  It should map images and text queries into the same vector space so text can
  retrieve likely relevant photos quickly.

Qwen3-VL can explain an image, but it is too expensive to run across every
query over a large photo library. Qwen3-VL-Embedding can retrieve candidates,
but it does not replace caption generation or event explanation.

## Local Execution Policy

- No external API calls.
- No cloud VLM or cloud embedding API.
- No network upload of photos, OCR text, LINE text, GPS, captions, or vectors.
- No automatic model downloads.
- Real models must be configured through `private_config/model_runtime.yaml`
  or explicit local paths.
- Benchmark images and outputs live under ignored paths such as
  `private_eval/` or `eval_outputs/`.

The committed files `configs/model_runtime.example.yaml` and
`configs/vlm_benchmark.example.yaml` contain dummy configuration only.

## Benchmark Workflow

1. Create private benchmark images:

   ```text
   private_eval/vlm_benchmark/images/
   ```

2. Create private cases:

   ```bash
   cp configs/vlm_benchmark.example.yaml private_eval/vlm_benchmark/benchmark_cases.yaml
   ```

3. Create private model runtime settings:

   ```bash
   cp configs/model_runtime.example.yaml private_config/model_runtime.yaml
   ```

4. Check model configuration without loading or downloading models:

   ```bash
   python -m personal_lifelog_rag.app.cli vlm-model-info --config private_config/model_runtime.yaml
   ```

5. Smoke-test the benchmark with fake engines:

   ```bash
   python -m personal_lifelog_rag.app.cli benchmark-qwen-multimodal \
     --cases configs/vlm_benchmark.example.yaml \
     --engine fake \
     --save
   ```

6. Run the real local benchmark only after models are already present locally:

   ```bash
   python -m personal_lifelog_rag.app.cli benchmark-qwen-multimodal \
     --cases private_eval/vlm_benchmark/benchmark_cases.yaml \
     --config private_config/model_runtime.yaml \
     --save
   ```

## Metrics

Qwen3-VL benchmark metrics:

- execution status and failure rate
- latency per image
- caption and short caption availability
- scene/object/activity/food/location cue coverage
- expected tag matches
- forbidden term violations
- schema validity

Qwen3-VL-Embedding benchmark metrics:

- image embedding success
- text embedding success
- embedding dimension
- query-to-image target rank
- top1 accuracy
- recall@3 and recall@5

## Recommended Integration

- Store Qwen3-VL outputs in `media_vlm`.
- Store future Qwen3-VL-Embedding vectors in a dedicated media embedding table.
- Use embeddings to fetch candidate photos first.
- Rerank candidates with VLM captions, OCR, LINE evidence, GPS, places, and
  event overrides.
- Keep VLM-only claims conservative. Use "可能性があります" and ask the user
  to verify important photos in the UI.

## Safety

Benchmark prompts and filters must not infer identity, relationships, emotions,
health, religion, politics, occupation, or other sensitive traits. If people are
present, only store a coarse `people_present` signal and rough count. Forbidden
terms in benchmark cases should include words that would indicate unsafe
relationship or sensitive-attribute inference.

See `docs/vlm_prompting_and_safety.md` for the prompt templates, JSON schema,
safety flags, and evidence-strength rules used before saving Qwen3-VL output.
