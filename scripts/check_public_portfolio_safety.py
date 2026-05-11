#!/usr/bin/env python3
"""Check public portfolio HTML for obvious private-data leaks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from personal_lifelog_rag.reporting.portfolio_html import check_public_portfolio_path, format_safety_report
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from personal_lifelog_rag.reporting.portfolio_html import check_public_portfolio_path, format_safety_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    report = check_public_portfolio_path(args.path)
    print(format_safety_report(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
