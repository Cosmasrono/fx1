"""Higher-timeframe direction filter for short-term entries."""
import asyncio
from datetime import datetime, timedelta, timezone

from .config import settings
from .market import fetch_candles
from .strategy import enrich


_cached_context: dict | None = None
_cached_at: datetime | None = None


async def higher_timeframe_context() -> dict:
    global _cached_context, _cached_at
    now = datetime.now(timezone.utc)
    if (_cached_context is not None and _cached_at is not None and
            now - _cached_at < timedelta(minutes=settings.higher_timeframe_cache_minutes)):
        return _cached_context
    requests = (fetch_candles(120, settings.weekly_interval),
                fetch_candles(120, settings.monthly_interval))
    weekly, monthly = await asyncio.gather(*requests, return_exceptions=True)
    levels = {}
    for name, result in (("weekly", weekly), ("monthly", monthly)):
        if isinstance(result, Exception):
            levels[name] = {"trend": "UNAVAILABLE"}
            continue
        candles, _ = result
        data = enrich(candles)
        row = data.iloc[-2] if len(data) > 2 else data.iloc[-1]
        levels[name] = {
            "trend": "BULLISH" if row.ema20 > row.ema50 else "BEARISH",
            "candle_time": row.datetime.isoformat(),
        }
    _cached_context, _cached_at = levels, now
    return levels


def entry_allowed(action: str, context: dict) -> bool:
    """Require both higher-timeframe data sets and direction agreement."""
    expected = "BULLISH" if action == "BUY" else "BEARISH"
    trends = [context[name]["trend"] for name in ("weekly", "monthly")]
    return all(trend == expected for trend in trends)
