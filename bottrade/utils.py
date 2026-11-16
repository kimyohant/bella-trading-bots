"""Shared helpers. Standard library only, no imports from the rest of the package."""

from __future__ import annotations

import math
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence


# --------------------------------------------------------------------------- time

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_ts(value: Any) -> datetime | None:
    """Best-effort parse of an ISO-8601 string, epoch seconds/ms, or datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 1e11:  # milliseconds
            seconds /= 1000.0
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            return parse_ts(int(text))
        cleaned = text.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(cleaned)
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S"):
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def iso(dt: datetime | None) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds") if dt else ""


def hours_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 3600.0


# --------------------------------------------------------------------------- math

def clamp(value: float, low: float, high: float) -> float:
    if value != value:  # NaN
        return low
    return low if value < low else high if value > high else value


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    try:
        if denominator == 0:
            return default
        result = numerator / denominator
        return default if result != result else result
    except (TypeError, ZeroDivisionError):
        return default


def sigmoid(x: float) -> float:
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def mean(values: Sequence[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / (len(values) - 1))


def max_drawdown(equity: Iterable[float]) -> float:
    """Largest peak-to-trough decline, returned as a positive fraction."""
    peak = -math.inf
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst


# --------------------------------------------------------------------------- format

def fmt_usd(value: float, digits: int = 2) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000:
        return f"{sign}${value / 1_000_000:,.{digits}f}M"
    if value >= 1_000:
        return f"{sign}${value / 1_000:,.{digits}f}K"
    return f"{sign}${value:,.{digits}f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:+.{digits}f}%"


def fmt_price(value: float) -> str:
    if value <= 0:
        return "0"
    if value >= 1:
        return f"{value:,.2f}"
    if value >= 0.01:
        return f"{value:.4f}"
    return f"{value:.3e}"


def bar(value: float, width: int = 20) -> str:
    """Tiny ASCII bar for terminal sparklines."""
    filled = int(clamp(value, 0.0, 1.0) * width)
    return "█" * filled + "·" * (width - filled)


# --------------------------------------------------------------------------- text

_STOPWORDS = {
    "will", "the", "a", "an", "be", "by", "in", "on", "at", "to", "of", "for",
    "before", "after", "this", "that", "is", "are", "and", "or", "if", "any",
    "who", "what", "when", "which", "how", "does", "do", "it", "than", "more",
    "less", "there", "their", "its", "as", "from", "with", "not", "no", "yes",
}


def keywords(text: str, limit: int = 5) -> list[str]:
    """Crude keyword extraction used to build news search queries."""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9'\-]{2,}", text or "")
    seen: list[str] = []
    for token in tokens:
        low = token.lower()
        if low in _STOPWORDS or low in seen:
            continue
        seen.append(low)
        if len(seen) >= limit:
            break
    return seen


def shorten(text: str, width: int = 64) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


def now_ms() -> int:
    return int(time.time() * 1000)
