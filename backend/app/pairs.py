"""Read-only multi-pair monitoring; never switches the engine's global settings."""
import asyncio
import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from . import database
from .config import settings
from .market import fetch_candles, validate_entry_data, interval_minutes
from .strategy import enrich

PAIRS = (("EUR/USD", .0001, 5), ("GBP/USD", .0001, 5), ("USD/JPY", .01, 3))


def journal_by_symbol():
    result = {}
    if not database.DB_PATH.exists():
        return result
    with sqlite3.connect(database.DB_PATH.as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        for row in db.execute("SELECT status,pnl,signal_snapshot FROM trades"):
            try:
                context = json.loads(row["signal_snapshot"]).get("training_context", {})
                symbol = context.get("symbol")
                if not symbol or context.get("feed") != "twelve_data":
                    continue
            except (ValueError, TypeError, AttributeError):
                continue
            totals = result.setdefault(symbol, {"closed": 0, "wins": 0, "open": 0, "realized_pnl": 0.0})
            if row["status"] == "CLOSED":
                totals["closed"] += 1
                totals["wins"] += int(row["pnl"] > 0)
                totals["realized_pnl"] += row["pnl"]
            elif row["status"] == "OPEN":
                totals["open"] += 1
    for totals in result.values():
        totals["realized_pnl"] = round(totals["realized_pnl"], 2)
    return result


async def overview(active_signal=None):
    async def one(symbol, pip, digits):
        base = {"symbol": symbol, "digits": digits, "active": symbol == settings.symbol,
                "status": "unavailable", "decision": None}
        try:
            candles, feed = await fetch_candles(symbol=symbol)
            quality = validate_entry_data(candles)
            latest = candles.iloc[-1]
            age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(latest.datetime)).total_seconds()
            if not 0 <= age < interval_minutes() * 60:
                quality = {"allowed": False, "reason": "The latest price is from an older candle."}
            data = enrich(candles)
            if len(data) < 2:
                raise ValueError("Not enough candles for trend indicators")
            closed = data.iloc[-2]
            day = pd.Timestamp(latest.datetime).normalize()
            today = candles[candles.datetime >= day]
            opening = float(today.iloc[0].open)
            quote = float(latest.close)
            change = quote - opening
            decision = None
            if (symbol == settings.symbol and active_signal and quality["allowed"]
                    and pd.Timestamp(active_signal.get("candle_time")) == pd.Timestamp(closed.datetime)):
                decision = {k: active_signal.get(k) for k in ("action", "reason", "candle_time")}
            return {**base, "status": "fresh" if quality["allowed"] else "stale",
                    "message": quality["reason"], "feed": feed, "price": quote,
                    "candle_time": latest.datetime.isoformat(), "day_start": today.iloc[0].datetime.isoformat(),
                    "change_percent": change / opening * 100, "change_pips": change / pip,
                    "trend": "BULLISH" if closed.ema20 > closed.ema50 else "BEARISH" if closed.ema20 < closed.ema50 else "NEUTRAL",
                    "atr_pips": float(closed.atr) / pip, "rsi": float(closed.rsi), "decision": decision,
                    "series": [{"time": row.datetime.isoformat(), "price": float(row.close)}
                               for row in candles.tail(96).itertuples()]}
        except Exception:
            # Provider/network exceptions can contain the API key in their URL.
            return {**base, "message": "Price data unavailable. Check the provider connection, symbol access or API quota."}
    rows = await asyncio.gather(*(one(*pair) for pair in PAIRS))
    try:
        journal = journal_by_symbol()
        journal_available = True
    except sqlite3.Error:
        journal, journal_available = {}, False
    for row in rows:
        row["paper"] = journal.get(row["symbol"])
    return {"as_of": datetime.now(timezone.utc).isoformat(), "interval": settings.interval,
            "strategy": settings.strategy, "pairs": rows, "journal_available": journal_available}
