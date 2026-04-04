"""
Persistent disk cache for Binance kline data.

Raw klines for any historical (symbol, date, interval) tuple are immutable —
they never need to be re-fetched from the API.  This module transparently
stores them as JSON files on disk so that every script (backtest_month.py,
simulate.py, rank_today.py) reuses the same data without extra API calls.

Cache structure:
    kline_disk_cache/
        {SYMBOL}_{market}/
            {YYYY-MM-DD}_4h.json   ← first 4-hour candle data  (NY date)
            {YYYY-MM-DD}_5m.json   ← full-day 5-min candle data (NY date)

Only non-empty API responses are written to disk, so transient errors will
be retried on the next run.

To clear the cache for a specific symbol/date simply delete the relevant file.
To clear everything:  rm -rf kline_disk_cache/
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from main import (
    get_klines as _get_klines,
    NY_TZ,
    Client,
)

# All cache files live under this directory (sibling of this script).
CACHE_DIR = Path(__file__).with_name("kline_disk_cache")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _cache_path(symbol: str, market: str, interval: str, date_midnight_ny) -> Path:
    """Return the path for a single (symbol, market, interval, ny_date) entry."""
    date_key = date_midnight_ny.strftime("%Y-%m-%d")
    # Normalise interval to lowercase for consistent filenames (5m, 4h, 1h …)
    iv = str(interval).lower().replace(" ", "")
    subdir = CACHE_DIR / f"{symbol}_{market}"
    return subdir / f"{date_key}_{iv}.json"


# ── Public API ────────────────────────────────────────────────────────────────

def get_klines_cached(
    symbol: str,
    market: str,
    interval: str,
    start_str: str,
    end_str: str,
    date_midnight_ny=None,
):
    """
    Drop-in replacement for main.get_klines that caches responses to disk.

    If *date_midnight_ny* is supplied (a timezone-aware datetime at NY midnight)
    the result is stored/loaded from the cache directory.  When it is None the
    call is forwarded to the live API without caching (e.g. intra-day usage).
    """
    if date_midnight_ny is None:
        return _get_klines(symbol, market, interval, start_str, end_str)

    path = _cache_path(symbol, market, interval, date_midnight_ny)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            # Corrupted file — delete and re-fetch.
            path.unlink(missing_ok=True)

    data = _get_klines(symbol, market, interval, start_str, end_str)

    # Only persist non-empty responses so transient API failures are retried.
    if data:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    return data


def get_first_4h_candle_levels_cached(symbol: str, market: str, date_midnight_ny):
    """
    Cached version of main.get_first_4h_candle_levels.

    Returns (open_level, high_level) i.e. the low and high of the first closed
    4-hour candle of *date_midnight_ny* (NY timezone).
    The underlying 4H klines are persisted so subsequent calls hit disk only.
    """
    next_day_ny = date_midnight_ny + timedelta(days=1)
    today_midnight_utc = date_midnight_ny.astimezone(timezone.utc)
    start_str = today_midnight_utc.strftime("%d %b %Y %H:%M:%S")
    end_str = next_day_ny.astimezone(timezone.utc).strftime("%d %b %Y %H:%M:%S")

    klines_4h = get_klines_cached(
        symbol,
        market,
        Client.KLINE_INTERVAL_4HOUR,
        start_str,
        end_str,
        date_midnight_ny,
    )

    if not klines_4h:
        return None, None

    now_ny = datetime.now(NY_TZ)
    if date_midnight_ny.date() < now_ny.date():
        ref_ms = int(next_day_ny.astimezone(timezone.utc).timestamp() * 1000)
    else:
        ref_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    closed = [k for k in klines_4h if k[6] < ref_ms]  # k[6] = candle close time
    if not closed:
        return None, None

    first = closed[0]
    return float(first[3]), float(first[2])  # low, high
