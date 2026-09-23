import asyncio
import copy
import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from app import database, engine, smc, prediction, backtest
from app.config import settings
from app.execution import account_risk_gate
from app.market import synthetic_candles
from app.strategy import enrich, order_levels
from test_prediction import smc_signal
from test_smc import _trending_frame


@pytest.fixture
def paper(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "journal.db")
    for name, value in {"strategy": "smc", "higher_timeframe_filter": False,
                        "prediction_model_enabled": False, "prediction_auto_retrain": False,
                        "news_filter_enabled": False}.items():
        monkeypatch.setattr(settings, name, value)
    database.initialize(10000)
    return tmp_path


def insert_open(opened_at):
    with database.connection() as db:
        db.execute("""INSERT INTO trades
            (side,status,entry,stop_loss,take_profit,units,risk_amount,opened_at,signal_snapshot,entry_candle_time)
            VALUES('BUY','OPEN',1.10,1.09,1.12,10000,100,?,?,'2026-09-15T11:00:00Z')""",
            (opened_at, json.dumps(smc_signal())))
        return dict(db.execute("SELECT * FROM trades").fetchone())


def test_exit_cannot_use_a_bar_from_before_the_actual_entry(paper):
    trade = insert_open("2026-09-15T11:30:14Z")
    with database.connection() as db:
        trader = engine.PaperEngine()
        assert not trader._manage_open(db, trade, 1.13, 1.08, 1.10, "2026-09-15T11:15:00Z")
        # Even the entry candle's eventual range contains unknown pre-entry wicks.
        assert not trader._manage_open(db, trade, 1.13, 1.08, 1.10, "2026-09-15T11:30:00Z")
        assert db.execute("SELECT status FROM trades").fetchone()[0] == "OPEN"
        assert trader._manage_open(db, trade, 1.09, 1.09, 1.09, "2026-09-15T11:31:00Z", sampled=True)
        assert db.execute("SELECT closed_at FROM trades").fetchone()[0] == "2026-09-15T11:31:00+00:00"


def test_completed_exit_time_is_when_the_bar_becomes_known(paper):
    trade = insert_open("2026-09-15T11:30:00Z")
    with database.connection() as db:
        assert engine.PaperEngine()._manage_open(db, trade, 1.13, 1.10, 1.12, "2026-09-15T11:30:00Z")
        assert db.execute("SELECT closed_at FROM trades").fetchone()[0] == "2026-09-15T11:45:00+00:00"


def test_scan_uses_current_price_and_deduplicates_a_pool_after_restart(paper, monkeypatch):
    candles = synthetic_candles(240)
    signal = smc_signal()
    price = float(candles.close.iloc[-1])
    signal.update(price=price - .001, candle_time=candles.datetime.iloc[-2].isoformat(),
                  confidence=.9, reason="Complete setup", setup_id="smc:BUY:pool-1")
    signal["levels_hint"] = {"stop": price - .01, "target": price + .02}
    async def feed():
        return candles, "twelve_data"
    monkeypatch.setattr(engine, "fetch_candles", feed)
    monkeypatch.setattr(engine, "signal_from", lambda data: copy.deepcopy(signal))
    asyncio.run(engine.PaperEngine().scan())
    with database.connection() as db:
        trade = dict(db.execute("SELECT * FROM trades").fetchone())
        cost = (settings.spread_pips / 2 + settings.slippage_pips) * settings.pip_size
        assert trade["entry"] == pytest.approx(price + cost, abs=1e-6)
        assert trade["setup_id"] == signal["setup_id"]
        # A closed pool must remain consumed, even if the same candle is retried.
        db.execute("UPDATE trades SET status='CLOSED',pnl=10,closed_at=?", (datetime.now(timezone.utc).isoformat(),))
        db.execute("UPDATE account SET last_candle=NULL")
    result = asyncio.run(engine.PaperEngine().scan())
    assert result["action"] == "HOLD" and "already been traded" in result["reason"]
    with database.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 1


def test_stale_latest_candle_cannot_fill_an_entry(paper, monkeypatch):
    candles = synthetic_candles(240)
    candles["datetime"] -= pd.Timedelta(minutes=15)
    async def feed():
        return candles, "twelve_data"
    monkeypatch.setattr(engine, "fetch_candles", feed)
    signal = smc_signal()
    signal.update(candle_time=candles.datetime.iloc[-2].isoformat(), confidence=.9, reason="Setup")
    monkeypatch.setattr(engine, "signal_from", lambda data: copy.deepcopy(signal))
    result = asyncio.run(engine.PaperEngine().scan())
    assert result["action"] == "HOLD" and "current candle" in result["reason"]
    with database.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0


def test_open_positions_and_proposed_order_reserve_daily_budget(monkeypatch):
    monkeypatch.setattr(settings, "max_daily_loss_percent", 2)
    monkeypatch.setattr(settings, "max_weekly_loss_percent", 5)
    opened = [{"entry": 1.1, "stop_loss": 1.09, "units": 10000, "risk_amount": 100}]
    assert account_risk_gate([], 10000, open_trades=opened, new_risk=100)["allowed"]
    gate = account_risk_gate([], 10000, open_trades=opened * 2, new_risk=100)
    assert not gate["allowed"] and "daily loss budget" in gate["reason"]
    closed = [{"closed_at": "2026-09-15T09:00:00Z", "pnl": -100}]
    gate = account_risk_gate(closed, 9900, "2026-09-15T10:00:00Z", opened, 99)
    assert not gate["allowed"]


@pytest.mark.parametrize("side,stop,target", [("BUY", 1.11, 1.13), ("SELL", 1.09, 1.07)])
def test_repriced_entry_rejects_invalid_structural_levels(side, stop, target):
    signal = {"action": side, "price": 1.10, "execution_price": 1.10,
              "levels_hint": {"stop": stop, "target": target}}
    with pytest.raises(ValueError, match="outside"):
        order_levels(signal, 10000, 1, 1, 1.5, .00005)


def test_high_score_cannot_replace_the_required_smc_sequence(monkeypatch):
    candidate = {"kind": "BULLISH", "score": 100, "breakdown": {}, "price": 1.1,
                 "stop_hint": 1.09, "target_hint": 1.12, "killzone": "LONDON_OPEN",
                 "swept_level": 1.095, "structure": None, "poi": "FVG", "equilibrium": 1.11,
                 "complete": False}
    monkeypatch.setattr(smc, "evaluate", lambda frame, kind, *args: candidate if kind == "BULLISH" else None)
    result = smc.signal_from(enrich(_trending_frame()))
    assert result["action"] == "HOLD" and "later retracement" in result["reason"]


def test_every_emitted_smc_entry_has_all_four_conditions():
    data = enrich(synthetic_candles(1000))
    entries = 0
    for i in range(100, len(data) - 1):
        signal = smc.signal_from(data.iloc[max(0, i - 200):i + 2])
        if signal["action"] in ("BUY", "SELL"):
            entries += 1
            assert signal["setup_id"]
            assert all(signal["score_breakdown"][name] > 0 for name in
                       ("liquidity_sweep", "structure_shift", "displacement", "point_of_interest"))
    assert entries > 0, "The fixture must exercise qualifying entries as well as HOLDs"


def test_revision_separates_old_journal_and_model():
    assert prediction.VERSION == 3
    assert prediction.MODEL_PATH.name == "prediction_model_v3.json"


@pytest.mark.parametrize("missing", ["structure", "displacement", "zone", "later_bar", "invalidated"])
def test_smc_evaluation_requires_each_step_and_an_intact_stop(monkeypatch, missing):
    frame = enrich(_trending_frame())
    frame["low"], frame["high"] = .9, 1.3
    cutoff = len(frame) - 2
    pool = smc.Swing(10, 1.0, "LOW")
    swings = [pool, smc.Swing(12, 1.2, "HIGH")]
    monkeypatch.setattr(smc, "_window_context", lambda *args: (swings, []))
    monkeypatch.setattr(smc, "liquidity_sweep", lambda *args: (pool, cutoff - 4))
    monkeypatch.setattr(smc, "structure_break", lambda *args:
                        None if missing == "structure" else (cutoff if missing == "later_bar" else cutoff - 2, "CHOCH"))
    monkeypatch.setattr(smc, "displacement_index", lambda *args:
                        None if missing == "displacement" else cutoff - 3)
    monkeypatch.setattr(smc, "order_block", lambda *args:
                        None if missing == "zone" else smc.Zone(cutoff - 3, 1.3, .9, "BULLISH", "ORDER_BLOCK"))
    if missing == "invalidated":
        frame.loc[cutoff - 1, "low"] = .5
    assert not smc.evaluate(frame, "BULLISH")["complete"]


def test_backtest_prices_the_next_open_and_consumes_each_pool_once(monkeypatch):
    candles = synthetic_candles(300)
    candles["open"] = candles["close"] + .0001
    monkeypatch.setattr(settings, "strategy", "smc")
    seen = []
    def candidate(data, index):
        signal = smc_signal()
        signal.update(price=float(data.close.iloc[index]), candle_time=data.datetime.iloc[index].isoformat(),
                      setup_id="pool-once")
        signal["levels_hint"] = {"stop": .5, "target": 2.0}
        return signal
    original = backtest.order_levels
    def capture(signal, *args):
        seen.append((signal["price"], signal["execution_price"]))
        return original(signal, *args)
    monkeypatch.setattr(backtest, "_signal_at", candidate)
    monkeypatch.setattr(backtest, "order_levels", capture)
    result = backtest.run_backtest(candles, backtest.BacktestSettings(10000, 1, 1, 1.5))
    assert result["metrics"]["open_trades"] == 1
    assert len(seen) == 1
    data = enrich(candles)
    assert seen[0] == (float(data.close.iloc[2]), float(data.open.iloc[3]))
