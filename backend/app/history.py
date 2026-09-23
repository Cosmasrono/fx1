"""Paginated multi-year candle download with an on-disk cache.

``market.fetch_candles`` is capped by the provider at 5000 rows per request,
which is only about seven weeks of 15-minute candles -- far too short to
validate a strategy. This module walks backwards through ``end_date`` windows
until it reaches the requested start, then caches the result as CSV so a
research run never spends quota twice on the same range.

The cache is intentionally plain CSV: it stays diffable and needs no extra
dependency, and a few hundred thousand candles is only tens of megabytes.
"""
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd

from .config import settings
from .market import interval_minutes

CACHE_DIR = Path(__file__).resolve().parents[1] / "data"
MAX_ROWS_PER_REQUEST = 5000
# A throttled window clears within a minute; retry rather than losing progress.
MAX_RETRIES = 5
RETRY_SECONDS = 65.0
COLUMNS = ["datetime", "open", "high", "low", "close"]


def cache_path(symbol: str, interval: str) -> Path:
    slug = symbol.replace("/", "").lower()
    return CACHE_DIR / f"{slug}_{interval}.csv"


def load_cache(symbol: str, interval: str) -> pd.DataFrame | None:
    path = cache_path(symbol, interval)
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["datetime"])
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    return frame.sort_values("datetime").reset_index(drop=True)


def save_cache(frame: pd.DataFrame, symbol: str, interval: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(symbol, interval)
    frame.to_csv(path, index=False)
    return path


def merge(*frames: pd.DataFrame) -> pd.DataFrame:
    """Combine downloads and drop duplicate timestamps, keeping the newest copy."""
    usable = [f for f in frames if f is not None and not f.empty]
    if not usable:
        return pd.DataFrame(columns=COLUMNS)
    combined = pd.concat(usable, ignore_index=True)
    combined = combined.drop_duplicates(subset="datetime", keep="last")
    return combined.sort_values("datetime").reset_index(drop=True)


async def _request_window(client: httpx.AsyncClient, symbol: str, interval: str,
                          end: datetime) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "outputsize": MAX_ROWS_PER_REQUEST,
              "timezone": "UTC", "order": "DESC", "apikey": settings.twelve_data_api_key,
              "end_date": end.strftime("%Y-%m-%d %H:%M:%S")}
    for attempt in range(MAX_RETRIES):
        response = await client.get("https://api.twelvedata.com/time_series", params=params)
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        # The per-minute credit window is shared with any other call made from
        # this key, so a download can be throttled even when its own pacing is
        # correct. Waiting out the window is the documented remedy.
        throttled = response.status_code == 429 or (
            isinstance(payload, dict) and payload.get("code") == 429)
        if not throttled:
            break
        if attempt == MAX_RETRIES - 1:
            raise RuntimeError("Twelve Data rate limit: still throttled after "
                               f"{MAX_RETRIES} attempts; lower requests_per_minute.")
        await asyncio.sleep(RETRY_SECONDS)
    if response.is_error:
        # As in market.py: never surface httpx's message, it embeds the API key.
        message = payload.get("message") if isinstance(payload, dict) else None
        raise RuntimeError(f"Twelve Data HTTP {response.status_code}: "
                           f"{message or 'history request failed'}")
    if isinstance(payload, dict) and payload.get("status") == "error":
        raise RuntimeError(f"Twelve Data error: {payload.get('message', 'unknown')}")
    if "values" not in payload:
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.DataFrame(payload["values"])
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column])
    return frame[COLUMNS].sort_values("datetime").reset_index(drop=True)


async def download(symbol: str | None = None, interval: str | None = None,
                   years: float = 3.0, requests_per_minute: int = 6,
                   use_cache: bool = True, progress=None) -> pd.DataFrame:
    """Download ``years`` of candles, resuming from cache where possible.

    ``requests_per_minute`` throttles to the provider's plan limit. The free
    Basic plan allows 8/minute; the default of 6 leaves headroom for other
    calls sharing the key, and 429s are retried rather than aborting the run.

    Each window is written to the cache as it arrives, so an interrupted or
    failed download resumes instead of re-spending quota on the same range.
    Set ``use_cache=False`` to force a clean re-download.
    """
    symbol = symbol or settings.symbol
    interval = interval or settings.interval
    if not settings.twelve_data_api_key:
        raise RuntimeError("TWELVE_DATA_API_KEY is missing; multi-year history needs a real key.")

    cached = load_cache(symbol, interval) if use_cache else None
    now = datetime.now(timezone.utc)
    target_start = now - timedelta(days=365.25 * years)
    if cached is not None and not cached.empty and cached.datetime.min() <= pd.Timestamp(target_start):
        if progress:
            progress(f"cache already covers {cached.datetime.min().date()} "
                     f"-> {cached.datetime.max().date()} ({len(cached)} rows)")
        return cached

    step = timedelta(minutes=interval_minutes(interval))
    delay = 60.0 / max(requests_per_minute, 1)
    cursor = now
    collected: list[pd.DataFrame] = [cached] if cached is not None else []
    async with httpx.AsyncClient(timeout=30) as client:
        while cursor > target_start:
            batch = await _request_window(client, symbol, interval, cursor)
            if batch.empty:
                break
            collected.append(batch)
            oldest = batch.datetime.min().to_pydatetime()
            # Persist after every window. A multi-year pull is dozens of rate
            # limited requests; losing all of it to one late failure would mean
            # spending the quota again for nothing.
            save_cache(merge(*collected), symbol, interval)
            if progress:
                progress(f"  {oldest.date()} .. {batch.datetime.max().date()}  "
                         f"(+{len(batch)} rows)")
            # Step one interval past the oldest row so the next window does not
            # re-request a candle we already hold.
            next_cursor = oldest - step
            if next_cursor >= cursor or len(batch) < 2:
                break  # provider returned no new ground; stop rather than loop
            cursor = next_cursor
            if cursor > target_start:
                await asyncio.sleep(delay)

    frame = merge(*collected)
    frame = frame[frame.datetime >= pd.Timestamp(target_start)].reset_index(drop=True)
    if not frame.empty:
        save_cache(frame, symbol, interval)
    return frame
