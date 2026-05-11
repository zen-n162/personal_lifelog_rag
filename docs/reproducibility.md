# Reproducibility

This project is designed to be reproducible without sending private data to a
cloud service.

## Environment

Use the dedicated Conda environment:

```bash
conda run -n personal_lifelog_rag pytest
```

Local model runtime settings live in a private runtime configuration file. Keep
local model paths out of public reports and documentation.

## Freeze Workflow

1. Back up the SQLite database.
2. Run the test suite.
3. Run strict DB checks.
4. Run private eval.
5. Generate public and private reports.
6. Build the public portfolio HTML.
7. Run the public privacy checker.
8. Create the release manifest.

## Release Manifest

Use:

```bash
conda run -n personal_lifelog_rag python -m personal_lifelog_rag.app.cli release-check --version v0.1 --save-manifest
```

The manifest stores model names and aggregate counts, but not local absolute
model paths.

## Notes

- Generated eval and report files are local artifacts.
- Heavy month rollout should be done one month at a time.
- Failed rows should be retried with dry-run first.
