import asyncio
import json
import sqlite3

import pandas as pd
import pytest

from app import pairs, market, database
from app.config import settings


def test_pair_cache_isolated_and_global_settings_unchanged(monkeypatch):
    market._candle_cache.clear()
    market._candle_locks.clear()
    original = settings.symbol
    calls = []
    async def provider(size, interval, symbol=None):
        calls.append(symbol)
        frame = market.synthetic_candles(size, interval)
        frame[["open", "high", "low", "close"]] *= 100 if symbol == "USD/JPY" else 1
        return frame, "test"
    monkeypatch.setattr(market, "_fetch_candles_uncached", provider)
    async def scenario():
        euro, yen = await asyncio.gather(market.fetch_candles(symbol="EUR/USD"), market.fetch_candles(symbol="USD/JPY"))
        assert yen[0].close.iloc[-1] == pytest.approx(euro[0].close.iloc[-1] * 100)
        await market.fetch_candles(symbol="USD/JPY")
    try:
        asyncio.run(scenario())
        assert sorted(calls) == ["EUR/USD", "USD/JPY"]
        assert settings.symbol == original
    finally:
        market._candle_cache.clear()
        market._candle_locks.clear()


def test_monitor_partial_failure_stale_decision_and_yen_pips(monkeypatch):
    frame = market.synthetic_candles(240)
    async def provider(symbol):
        if symbol == "GBP/USD":
            raise RuntimeError("https://provider.invalid?apikey=secret")
        data = frame.copy()
        if symbol == "USD/JPY":
            data[["open", "high", "low", "close"]] *= 100
        return data, "twelve_data"
    monkeypatch.setattr(pairs, "fetch_candles", provider)
    monkeypatch.setattr(pairs, "journal_by_symbol", lambda: {})
    monkeypatch.setattr(settings, "symbol", "EUR/USD")
    stale = {"candle_time": (frame.datetime.iloc[-2] - pd.Timedelta(days=1)).isoformat(), "action": "BUY"}
    result = asyncio.run(pairs.overview(stale))
    euro, pound, yen = result["pairs"]
    assert euro["status"] == yen["status"] == "fresh"
    assert pound["status"] == "unavailable" and "secret" not in json.dumps(result)
    assert euro["decision"] is None
    assert euro["change_pips"] == pytest.approx(yen["change_pips"])
    assert yen["digits"] == 3 and not yen["active"]
    assert all(row["paper"] is None for row in result["pairs"])


def test_older_quotes_are_labelled_stale(monkeypatch):
    frame = market.synthetic_candles(240)
    frame["datetime"] -= pd.Timedelta(minutes=15)
    async def provider(symbol):
        return frame, "twelve_data"
    monkeypatch.setattr(pairs, "fetch_candles", provider)
    monkeypatch.setattr(pairs, "journal_by_symbol", lambda: {})
    result = asyncio.run(pairs.overview())
    assert all(row["status"] == "stale" for row in result["pairs"])


def test_paper_results_require_matching_symbol_provenance(tmp_path, monkeypatch):
    path = tmp_path / "paper.db"
    monkeypatch.setattr(database, "DB_PATH", path)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE trades(status TEXT,pnl REAL,signal_snapshot TEXT)")
        for symbol, pnl, feed in [("EUR/USD", 25, "twelve_data"), ("GBP/USD", -10, "twelve_data"),
                                   ("EUR/USD", 999, "synthetic"), (None, 500, "twelve_data")]:
            context = {"training_context": {"symbol": symbol, "feed": feed}}
            db.execute("INSERT INTO trades VALUES('CLOSED',?,?)", (pnl, json.dumps(context)))
    records = pairs.journal_by_symbol()
    assert records["EUR/USD"]["realized_pnl"] == 25
    assert records["GBP/USD"]["realized_pnl"] == -10
    assert "USD/JPY" not in records
