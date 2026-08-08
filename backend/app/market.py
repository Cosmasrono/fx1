from datetime import datetime, timedelta, timezone
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


async def fetch_candles(outputsize: int = 240, interval: str | None = None) -> tuple[pd.DataFrame, str]:
    timeframe = interval or settings.interval
    if settings.use_synthetic_data:
        return synthetic_candles(outputsize, timeframe), "synthetic"
    if not settings.twelve_data_api_key:
        raise RuntimeError("TWELVE_DATA_API_KEY is missing. Add it to backend/.env or enable synthetic data.")
    params = {"symbol": settings.symbol, "interval": timeframe, "outputsize": outputsize,
              "timezone": "UTC", "apikey": settings.twelve_data_api_key}
    async with httpx.AsyncClient(timeout=20) as client:
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
    return frame.sort_values("datetime").reset_index(drop=True), "twelve_data"


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
