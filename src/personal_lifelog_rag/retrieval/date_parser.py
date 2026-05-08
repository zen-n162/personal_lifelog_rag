"""Small Japanese/ISO date parser for timeline questions."""

from __future__ import annotations

import calendar
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta

DEFAULT_YEAR_ENV_VAR = "PERSONAL_LIFELOG_RAG_DEFAULT_YEAR"


@dataclass(frozen=True)
class DateRange:
    start_date: date
    end_date: date
    label: str
    year_was_inferred: bool = False
    ambiguous: bool = False

    @property
    def start_iso(self) -> str:
        return self.start_date.isoformat()

    @property
    def end_iso(self) -> str:
        return self.end_date.isoformat()


def parse_date_query(
    query: str,
    *,
    today: date | None = None,
    default_year: int | None = None,
) -> DateRange | None:
    base_date = today or date.today()
    inferred_year = default_year or int(os.getenv(DEFAULT_YEAR_ENV_VAR, "2024"))

    exact = _match_exact_date(query)
    if exact:
        return DateRange(exact, exact, exact.isoformat())

    month = _match_month(query)
    if month:
        year, month_number = month
        last_day = calendar.monthrange(year, month_number)[1]
        return DateRange(
            date(year, month_number, 1),
            date(year, month_number, last_day),
            f"{year}-{month_number:02d}",
        )

    md = re.search(r"(?<!\d)(\d{1,2})月(\d{1,2})日", query)
    if md:
        try:
            parsed = date(inferred_year, int(md.group(1)), int(md.group(2)))
        except ValueError:
            return None
        return DateRange(
            parsed,
            parsed,
            parsed.isoformat(),
            year_was_inferred=True,
            ambiguous=True,
        )

    if "昨日" in query:
        parsed = base_date - timedelta(days=1)
        return DateRange(parsed, parsed, "yesterday")
    if "今日" in query:
        return DateRange(base_date, base_date, "today")
    if "今月" in query:
        last_day = calendar.monthrange(base_date.year, base_date.month)[1]
        return DateRange(date(base_date.year, base_date.month, 1), date(base_date.year, base_date.month, last_day), "this month")
    if "先月" in query:
        first_this_month = date(base_date.year, base_date.month, 1)
        last_prev_month = first_this_month - timedelta(days=1)
        first_prev_month = date(last_prev_month.year, last_prev_month.month, 1)
        return DateRange(first_prev_month, last_prev_month, "last month")
    if "最近" in query or "この1か月" in query or "この一か月" in query:
        return DateRange(base_date - timedelta(days=30), base_date, "recent 30 days")
    if "去年の夏" in query:
        year = base_date.year - 1
        return DateRange(date(year, 6, 1), date(year, 8, 31), f"{year} summer")
    last_year_month = re.search(r"去年(?:の)?(\d{1,2})月", query)
    if last_year_month:
        year = base_date.year - 1
        month_number = int(last_year_month.group(1))
        try:
            last_day = calendar.monthrange(year, month_number)[1]
            return DateRange(date(year, month_number, 1), date(year, month_number, last_day), f"{year}-{month_number:02d}")
        except calendar.IllegalMonthError:
            return None
    if "去年" in query:
        year = base_date.year - 1
        return DateRange(date(year, 1, 1), date(year, 12, 31), str(year))
    if "今年" in query:
        return DateRange(date(base_date.year, 1, 1), date(base_date.year, 12, 31), str(base_date.year))

    return None


def _match_exact_date(query: str) -> date | None:
    patterns = [
        re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日"),
        re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})"),
    ]
    for pattern in patterns:
        match = pattern.search(query)
        if match:
            year, month, day = (int(part) for part in match.groups())
            try:
                return date(year, month, day)
            except ValueError:
                return None
    return None


def _match_month(query: str) -> tuple[int, int] | None:
    match = re.search(r"(\d{4})年(\d{1,2})月(?!\d{1,2}日)", query)
    if not match:
        match = re.search(r"(\d{4})-(\d{1,2})(?!-\d{1,2})", query)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))
