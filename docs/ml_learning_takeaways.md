# ML Learning Takeaways

- VLM output should not be treated as ground truth.
- Embedding similarity is excellent for candidate generation but weak as a
  standalone explanation.
- Search quality improves when visual evidence is combined with events, OCR,
  and message context.
- Private eval is essential because small ranking changes can regress real
  user questions.
- Local model operations require careful diagnostics for dependency versions,
  model loading, and GPU memory.
- JSON output from thinking-style models needs repair and validation paths.
- OCR is useful but noisy; it should be redacted and treated as candidate
  evidence.
- Human review is not an afterthought. It is part of the safety and ranking
  loop.
- Public reporting needs its own privacy checks, separate from private eval.
