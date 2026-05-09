# Evaluation Summary

This page summarizes the local evaluation snapshot used for portfolio
explanation. Values are aggregate-only and should be refreshed before public
presentation.

Snapshot date: 2026-05-09

## Automated Tests

```text
pytest: 420 passed
```

The test suite covers ingestion, query intent, routing, event generation,
private eval, OCR/VLM/embedding services, multimodal search, UI service helpers,
monthly rollout planning, and safety filters with dummy data only.

## DB Integrity

```text
db-check --strict: ok
```

Strict checks include:

- media item integrity
- OCR/VLM orphan checks
- embedding dimension checks
- event evidence references
- fake/failed VLM evidence exclusion
- analysis job metadata integrity

## Private Eval

Latest local run:

```text
cases: 17
passed: 16
failed: 1
skipped: 0
top1 accuracy: 0.8
expected date recall@5: 0.8
forbidden phrase violations: 0
overclaim violations: 0
```

The single failure was a VLM quality case that did not yet allow the
`json_repaired` safety flag. This is a known evaluation-spec mismatch rather
than a privacy leak or database integrity issue.

Representative case types:

- date QA
- routed place QA
- keyword search
- image search
- multimodal search
- VLM quality
- event quality
- call search
- monthly summary

## Qwen3-VL Coverage

For the inspected 2024-12 to 2025-01 rollout window:

```text
media_vlm total: 400
success: 394
failed: 6
engine_unavailable: 0
engine: qwen3_vl_transformers
```

Common high-level tags included indoor scenes, cafe-like scenes, performance or
stage cues, food/meal cues, and vehicle interior cues. These are used as search
candidates, not final facts.

## Qwen3-VL-Embedding Coverage

For the same rollout window:

```text
media_embeddings total: 471
success: 471
embedding_dim: 4096
types:
- image: 223
- combined_text: 248
```

Embeddings are used for candidate retrieval and then reranked with VLM, OCR,
LINE, event, place, and human-review signals.

## OCR Coverage

OCR is intentionally optional and still limited:

```text
media_ocr total: 11
success: 5
engine_unavailable: 6
```

OCR is useful for signs, receipts, tickets, menus, screenshots, and station or
shop text, but the project treats it as noisy evidence.

## Known Limitations

- OCR coverage is still small.
- Some VLM rows need JSON repair or retry.
- Visual model output can over-describe; safety filtering and human review are
  required.
- Embedding similarity alone cannot prove an event happened.
- LINE mention versus actual action remains a hard ranking problem.
- Place dictionaries require manual private setup.
- Event split/merge editing is not yet implemented.

## Evaluation Takeaway

The project has a working local multimodal search and QA loop with strict DB
checks and regression tests. The strongest portfolio point is not perfect
accuracy; it is the full privacy-preserving engineering loop:

1. local data ingestion
2. local vision and embedding models
3. evidence-linked event construction
4. human review
5. private eval
6. redacted reporting

