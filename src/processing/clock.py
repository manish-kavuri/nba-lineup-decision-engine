"""Parse NBA PlayByPlay ``clock`` strings (ISO-8601 duration, period time remaining)."""

from __future__ import annotations

import re
from typing import Optional

_CLOCK_RE = re.compile(
    r"^PT"
    r"(?:(?P<hours>\d+)H)?"
    r"(?:(?P<minutes>\d+)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?"
    r"$",
    re.IGNORECASE,
)


def parse_clock_seconds_remaining(clock: Optional[object]) -> float:
    """Seconds remaining in the current period."""
    if clock is None:
        return float("nan")
    s = str(clock).strip()
    if not s or s.lower() == "nan":
        return float("nan")

    m = _CLOCK_RE.match(s)
    if not m:
        raise ValueError(f"Unparseable clock: {clock!r}")

    hours = float(m.group("hours") or 0)
    minutes = float(m.group("minutes") or 0)
    seconds = float(m.group("seconds") or 0)
    return hours * 3600 + minutes * 60 + seconds


def period_length_seconds(period: int) -> float:
    """Regulation quarters 12 min; overtime periods 5 min."""
    return 300.0 if period >= 5 else 720.0
