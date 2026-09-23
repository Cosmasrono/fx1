from datetime import datetime, timedelta, timezone
import asyncio
import time
import httpx
import numpy as np
import pandas as pd
from .config import settings


INTERVAL_MINUTES = {"1min": 1, "5min": 5, "15min": 15, "30min": 30, "45min": 45,
                    "1h": 60, "2h": 120, "4h": 240, "8h": 480, "1day": 1440,
                    "1week": 10080, "1month": 43200}


def interval_minutes(interval: str | None = None) -> int:
    timeframe = interval or settings.interval
    if timeframe not in INTERVAL_MINUTES:
        raise RuntimeError(f"Unsupported interval {timeframe!r}. Use one of {', '.join(INTERVAL_MINUTES)}.")
    return INTERVAL_MINUTES[timeframe]


_candle_cache: dict = {}
_candle_locks: dict = {}


async def fetch_candles(outputsize: int = 240, interval: str | None = None, symbol: str | None = None) -> tuple[pd.DataFrame, str]:
    """Share fresh snapshots within this process; callers receive independent frames.

    A larger cached window can satisfy smaller requests. Expired snapshots are
    never returned when the provider fails, and concurrent refreshes coalesce.
    """
    timeframe = interval or settings.interval
    key = (symbol or settings.symbol, timeframe, settings.use_synthetic_data, settings.twelve_data_api_key)
    loop = asyncio.get_running_loop()
    lock_key = (loop, key)
    lock = _candle_locks.setdefault(lock_key, asyncio.Lock())
    async with lock:
        now = time.monotonic()
        cached = _candle_cache.get(key)
        if cached is not None:
            expires, capacity, frame, feed = cached
            if now < expires and capacity >= outputsize:
                return frame.tail(outputsize).copy(deep=True).reset_index(drop=True), feed
        capacity = max(240, outputsize)
        # Expire at candle boundaries too, so a pre-close snapshot does not
        # conceal the next candle from the engine's first scan after close.
        step = interval_minutes(timeframe) * 60
        ttl = min(60.0, step - time.time() % step)
        expires = time.monotonic() + ttl
        frame, feed = (await _fetch_candles_uncached(capacity, timeframe, symbol=symbol)
                       if symbol is not None else await _fetch_candles_uncached(capacity, timeframe))
        _candle_cache[key] = (expires, capacity, frame.copy(deep=True), feed)
        return frame.tail(outputsize).copy(deep=True).reset_index(drop=True), feed


async def _fetch_candles_uncached(outputsize: int = 240, interval: str | None = None, symbol: str | None = None) -> tuple[pd.DataFrame, str]:
    timeframe = interval or settings.interval
    if settings.use_synthetic_data:
        if symbol is not None and symbol != "EUR/USD":
            raise RuntimeError("Synthetic preview is only available for EUR/USD.")
        return synthetic_candles(outputsize, timeframe), "synthetic"
    if not settings.twelve_data_api_key:
        raise RuntimeError("TWELVE_DATA_API_KEY is missing. Add it to backend/.env or enable synthetic data.")
    params = {"symbol": symbol or settings.symbol, "interval": timeframe, "outputsize": outputsize,
              "timezone": "UTC", "apikey": settings.twelve_data_api_key}
    # A full 5000-row page for training can take close to 20 seconds on its
    # own; keep the short timeout for the engine's small, frequent polls.
    async with httpx.AsyncClient(timeout=20 if outputsize <= 1000 else 60) as client:
        response = await client.get("https://api.twelvedata.com/time_series", params=params)
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if response.is_error:
            # Do not propagate httpx's exception string: it contains the full
            # request URL, including the API key query parameter.
            message = payload.get("message") if isinstance(payload, dict) else None
            raise RuntimeError(f"Twelve Data HTTP {response.status_code}: {message or 'market data request failed'}")
    if "values" not in payload:
        raise RuntimeError(payload.get("message", "Twelve Data did not return candles"))
    frame = pd.DataFrame(payload["values"])
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column])
    frame = frame.sort_values("datetime").reset_index(drop=True)
    validate_candles(frame)
    return frame, "twelve_data"


def validate_candles(frame: pd.DataFrame) -> None:
    prices = frame[["open", "high", "low", "close"]]
    if frame.empty or frame.datetime.isna().any() or frame.datetime.duplicated().any():
        raise ValueError("Market data is empty or contains invalid/duplicate timestamps.")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Market data contains invalid prices.")
    if ((frame.high < prices.max(axis=1)) | (frame.low > prices.min(axis=1))).any():
        raise ValueError("Market data contains inconsistent candle ranges.")


def validate_entry_data(frame: pd.DataFrame, now=None) -> dict:
    """Block stale or incomplete entries; valid older candles can still close positions."""
    validate_candles(frame)
    if len(frame) < 100 or not frame.datetime.is_monotonic_increasing:
        return {"allowed": False, "reason": "Entry blocked: insufficient or unordered candle history."}
    current = pd.to_datetime(now, utc=True) if now is not None else pd.Timestamp.now(tz="UTC")
    step = pd.Timedelta(minutes=interval_minutes())
    latest = pd.to_datetime(frame.datetime.iloc[-1], utc=True)
    if latest > current or current - latest > step * settings.market_max_age_bars:
        return {"allowed": False, "reason": "Entry blocked: market data is stale or future-dated."}
    recent = pd.to_datetime(frame.datetime.tail(4), utc=True).tolist()
    for before, after in zip(recent, recent[1:]):
        crosses_weekend = any(day.dayofweek >= 5 for day in pd.date_range(before.normalize(), after.normalize(), freq="D"))
        if after - before > step * 1.5 and not crosses_weekend:
            return {"allowed": False, "reason": "Entry blocked: recent market candles are missing."}
    return {"allowed": True, "reason": "Market data checks passed.", "latest_at": latest.isoformat()}


def synthetic_candles(count: int = 240, interval: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    step = interval_minutes(interval)
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    end -= timedelta(minutes=(end.hour * 60 + end.minute) % step)
    dates = pd.date_range(end=end, periods=count, freq=f"{step}min", tz="UTC")
    # Per-bar drift scales with time, dispersion with its square root, so shorter
    # bars stay realistic instead of carrying 15min-sized moves.
    scale = step / 15
    close = 1.1520 + np.cumsum(rng.normal(0.00002 * scale, 0.00045 * np.sqrt(scale), count))
    open_ = np.r_[close[0], close[:-1]]
    spread = rng.uniform(0.00015, 0.00065, count) * np.sqrt(scale)
    return pd.DataFrame({"datetime": dates, "open": open_, "high": np.maximum(open_, close)+spread,
                         "low": np.minimum(open_, close)-spread, "close": close})
