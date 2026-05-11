#!/usr/bin/env python3
"""Build the public single-file portfolio HTML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    from personal_lifelog_rag.reporting.portfolio_html import PortfolioHtmlOptions, build_portfolio_html
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from personal_lifelog_rag.reporting.portfolio_html import PortfolioHtmlOptions, build_portfolio_html


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/portfolio_public.html"))
    parser.add_argument("--mode", choices=["public"], default="public")
    parser.add_argument("--source-report", type=Path, default=None)
    parser.add_argument("--check-privacy", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    payload = build_portfolio_html(
        PortfolioHtmlOptions(
            output_html=args.output,
            mode=args.mode,
            source_report=args.source_report,
            check_privacy=args.check_privacy,
            force=args.force,
        )
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
