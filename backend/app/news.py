"""High-impact EUR/USD economic-event entry guard.

News is used as a risk blackout, never as a directional prediction. That keeps
the technical strategy testable and avoids chasing the first post-release wick.
"""
from datetime import datetime, timedelta, timezone
import httpx

from .config import settings


class NewsGuard:
    def __init__(self):
        self._events: list[dict] = []
        self._fetched_at: datetime | None = None
        self._last_status = self._disabled_status()

    async def evaluate(self, now: datetime | None = None) -> dict:
        now = _utc(now or datetime.now(timezone.utc))
        if not settings.news_filter_enabled:
            self._last_status = self._disabled_status(now)
            return self._last_status
        if not settings.fmp_api_key:
            self._last_status = {
                "enabled": True,
                "configured": False,
                "blocking": settings.news_fail_closed,
                "status": "UNCONFIGURED",
                "event": None,
                "checked_at": now.isoformat(),
            }
            return self._last_status
        try:
            if self._cache_expired(now):
                self._events = await self._fetch_events(now)
                self._fetched_at = now
            self._last_status = decision_from_events(self._events, now)
        except Exception as exc:
            self._last_status = {
                "enabled": True,
                "configured": True,
                "blocking": settings.news_fail_closed,
                "status": "UNAVAILABLE",
                "event": None,
                "error": str(exc),
                "checked_at": now.isoformat(),
            }
        return self._last_status

    def status(self) -> dict:
        return self._last_status

    def _cache_expired(self, now: datetime) -> bool:
        return self._fetched_at is None or now - self._fetched_at >= timedelta(minutes=settings.news_cache_minutes)

    async def _fetch_events(self, now: datetime) -> list[dict]:
        start = (now - timedelta(days=1)).date().isoformat()
        end = (now + timedelta(days=1)).date().isoformat()
        url = "https://financialmodelingprep.com/stable/economic-calendar"
        params = {"from": start, "to": end, "apikey": settings.fmp_api_key}
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(url, params=params)
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if response.is_error:
                # Avoid exposing the key from the request URL in an httpx error.
                message = payload.get("Error Message") if isinstance(payload, dict) else None
                raise RuntimeError(f"FMP calendar HTTP {response.status_code}: {message or 'request failed'}")
        if not isinstance(payload, list):
            raise RuntimeError("Economic calendar returned an unexpected response")
        normalized = []
        for item in payload:
            currency = str(item.get("currency") or "").upper()
            country = str(item.get("country") or "")
            # Prefer the affected-currency field. Country names/codes are a
            # fallback for provider rows that omit currency.
            relevant_country = country.lower() in {
                "us", "united states", "eu", "euro area", "euro zone",
                "germany", "france", "italy", "spain",
            }
            if currency not in {"USD", "EUR"} and not relevant_country:
                continue
            impact = str(item.get("impact") or "").lower()
            importance = {"high": 3, "medium": 2, "low": 1}.get(impact, 0)
            normalized.append({
                "Date": item.get("date"),
                "Country": country or currency,
                "Event": item.get("event"),
                "Importance": importance,
            })
        return normalized

    @staticmethod
    def _disabled_status(now: datetime | None = None) -> dict:
        return {
            "enabled": False,
            "configured": False,
            "blocking": False,
            "status": "DISABLED",
            "event": None,
            "checked_at": (now or datetime.now(timezone.utc)).isoformat(),
        }


def decision_from_events(events: list[dict], now: datetime) -> dict:
    now = _utc(now)
    before = timedelta(minutes=settings.news_blackout_minutes_before)
    after = timedelta(minutes=settings.news_blackout_minutes_after)
    relevant = []
    for event in events:
        if int(event.get("Importance") or 0) < 3 or not event.get("Date"):
            continue
        event_time = _parse_event_time(event["Date"])
        relevant.append((event_time, event))
    relevant.sort(key=lambda item: item[0])
    blocking = [(event_time, event) for event_time, event in relevant if event_time - before <= now <= event_time + after]
    chosen = min(blocking, key=lambda item: abs((item[0] - now).total_seconds())) if blocking else None
    upcoming = next(((event_time, event) for event_time, event in relevant if event_time > now), None)
    display = chosen or upcoming
    event_summary = None
    if display:
        event_time, event = display
        event_summary = {
            "name": event.get("Event") or event.get("Category") or "High-impact event",
            "country": event.get("Country"),
            "time": event_time.isoformat(),
            "importance": int(event.get("Importance") or 0),
        }
    return {
        "enabled": True,
        "configured": True,
        "blocking": bool(chosen),
        "status": "BLACKOUT" if chosen else "CLEAR",
        "event": event_summary,
        "checked_at": now.isoformat(),
    }


def _parse_event_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _utc(parsed)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


news_guard = NewsGuard()
