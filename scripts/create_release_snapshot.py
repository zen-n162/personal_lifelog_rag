#!/usr/bin/env python3
"""Create a public-safe v0.1 release manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    from personal_lifelog_rag.db.repository import LifelogRepository, resolve_db_path
    from personal_lifelog_rag.reporting.release_snapshot import (
        DEFAULT_RELEASE_MANIFEST,
        build_release_manifest,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from personal_lifelog_rag.db.repository import LifelogRepository, resolve_db_path
    from personal_lifelog_rag.reporting.release_snapshot import (
        DEFAULT_RELEASE_MANIFEST,
        build_release_manifest,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--version", default="v0.1")
    parser.add_argument("--eval-path", type=Path, default=Path("private_eval/questions_20241224.yaml"))
    parser.add_argument("--output", type=Path, default=DEFAULT_RELEASE_MANIFEST)
    parser.add_argument("--run-pytest", action="store_true")
    args = parser.parse_args()
    repository = LifelogRepository(resolve_db_path(args.db_path))
    manifest = build_release_manifest(
        repository,
        version=args.version,
        eval_path=args.eval_path,
        save_manifest=True,
        output=args.output,
        run_pytest=args.run_pytest,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
