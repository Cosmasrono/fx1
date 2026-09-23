import asyncio
import copy
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app import database, engine, journal, prediction, backtest
from app.config import settings
from app.execution import account_risk_gate, filter_setup
from app.learning import assess_examples, outcome_features
from app.market import synthetic_candles, validate_candles, validate_entry_data
from test_prediction import smc_signal


@pytest.fixture
def isolated_journal(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "paper.db")
    monkeypatch.setattr(prediction, "MODEL_PATH", tmp_path / "model.json")
    monkeypatch.setattr(prediction, "TRAINING_STATE_PATH", tmp_path / "training.json")
    monkeypatch.setattr(settings, "strategy", "smc")
    monkeypatch.setattr(settings, "use_synthetic_data", False)
    monkeypatch.setattr(settings, "higher_timeframe_filter", False)
    monkeypatch.setattr(settings, "prediction_auto_retrain", False)
    database.initialize(10000)
    return tmp_path


def insert_trade(signal, pnl=-100, reason="STOP_LOSS", status="CLOSED", closed=None):
    stamp = pd.Timestamp(signal["candle_time"])
    closed = closed or (stamp + pd.Timedelta(hours=1)).isoformat()
    with database.connection() as db:
        cursor = db.execute("""INSERT INTO trades
            (side,status,entry,stop_loss,take_profit,units,risk_amount,opened_at,closed_at,pnl,reason,signal_snapshot,entry_candle_time)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (signal["action"], status, signal["price"], 1.103, 1.109, 50000, 100,
             stamp.isoformat(), closed, pnl, reason, json.dumps(signal), stamp.isoformat()))
        return cursor.lastrowid


def test_closed_trade_feedback_replaces_replayed_outcome_and_excludes_open_trades(isolated_journal):
    signal = smc_signal()
    signal["training_context"] = prediction.entry_context(signal, "twelve_data")
    insert_trade(signal)
    insert_trade(signal, status="OPEN")
    paper, report = journal.examples(None)
    assert report["used"] == 1 and report["losses"] == 1 and report["closed_trades"] == 1
    history_row = {"time": signal["candle_time"].replace("T", " "), **prediction.features(signal),
                   "gate_score": 90, "win": 1, "r": 1.5}
    combined = journal.merge_examples(pd.DataFrame([history_row]), paper)
    assert len(combined) == 1
    assert combined.iloc[0]["source"] == "paper" and combined.iloc[0].win == 0
    assert combined.iloc[0].r == -1


def test_feedback_rejects_other_settings_and_unresolved_horizons(isolated_journal):
    signal = smc_signal()
    signal["training_context"] = prediction.entry_context(signal, "synthetic")
    insert_trade(signal)
    signal["training_context"] = prediction.entry_context(signal, "twelve_data")
    insert_trade(signal, closed="2026-01-10T08:15:00Z")
    paper, report = journal.examples(None)
    assert paper.empty and sum(report["skipped"].values()) == 2


def test_legacy_feedback_requires_matching_historical_price(isolated_journal):
    signal = smc_signal()
    insert_trade(signal)
    assert journal.examples(None)[1]["used"] == 0
    candles = pd.DataFrame({"datetime": [pd.Timestamp(signal["candle_time"])], "close": [signal["price"]]})
    assert journal.examples(candles)[1]["legacy_verified"] == 1
    candles["close"] += .01
    assert journal.examples(candles)[1]["used"] == 0


def test_loss_pause_recovers_and_timestamp_formats_do_not_bypass_daily_limit(monkeypatch):
    monkeypatch.setattr(settings, "loss_cooldown_hours", 24)
    trades = [{"id": i, "closed_at": "2026-08-12 09:00:00+00:00", "pnl": -50} for i in range(3)]
    paused = account_risk_gate(trades, 9850, "2026-08-12T12:00:00Z")
    assert not paused["allowed"] and paused["resume_at"] == "2026-08-13T09:00:00+00:00"
    assert account_risk_gate(trades, 9850, "2026-08-13T10:00:00Z")["allowed"]
    trades[0]["pnl"] = -110
    result = account_risk_gate(trades, 9790, "2026-08-12T12:00:00Z")
    assert result["daily_pnl"] == -210 and "daily loss" in result["reason"]


def test_pattern_learning_uses_recent_matching_strategy_and_past_outcomes_only():
    signal = smc_signal()
    row = {"side": "BUY", "features": outcome_features(signal), "outcome": "LOSS", "pnl": -100,
           "closed_at": "2026-01-04T00:00:00Z"}
    examples = [copy.deepcopy(row) for _ in range(3)]
    assert assess_examples(examples, signal, 3, .85, "2026-01-05T00:00:00Z")["blocked"]
    examples[0]["closed_at"] = "2026-01-06T00:00:00Z"
    examples[1]["features"]["_strategy"] = "judas"
    examples[2]["closed_at"] = "2025-01-01T00:00:00Z"
    assert assess_examples(examples, signal, 3, .85, "2026-01-05T00:00:00Z")["similar_matches"] == 0


def test_auto_retrain_batches_new_outcomes_and_respects_persisted_attempt(isolated_journal, monkeypatch):
    monkeypatch.setattr(settings, "prediction_auto_retrain", True)
    monkeypatch.setattr(settings, "prediction_model_enabled", True)
    worker = prediction.TrainingJob()
    monkeypatch.setattr(prediction, "job", worker)
    starts = []
    monkeypatch.setattr(worker, "start", lambda data: starts.append(data) or True)
    for _ in range(4):
        insert_trade(smc_signal())
    assert not prediction.maybe_retrain(now="2026-09-11T12:00:00Z")
    insert_trade(smc_signal())
    assert prediction.maybe_retrain(now="2026-09-11T12:00:00Z") and len(starts) == 1
    worker._state = {"status": "error", "started_at": "2026-09-11T11:00:00Z"}
    worker._persist()
    monkeypatch.setattr(prediction, "job", prediction.TrainingJob())
    assert not prediction.maybe_retrain(now="2026-09-11T12:00:00Z")


def test_freshness_and_invalid_candles_block_entries():
    candles = synthetic_candles(240)
    now = candles.datetime.iloc[-1] + pd.Timedelta(minutes=1)
    assert validate_entry_data(candles, now)["allowed"]
    assert not validate_entry_data(candles, now + pd.Timedelta(hours=2))["allowed"]
    assert not validate_entry_data(candles.iloc[:30], now)["allowed"]
    corrupt = candles.copy()
    corrupt.loc[3, "high"] = -1
    with pytest.raises(ValueError):
        validate_candles(corrupt)
    with pytest.raises(ValueError):
        validate_candles(pd.concat([candles, candles.tail(1)]))


def test_structural_scores_do_not_exceed_100_with_trend_bonus(monkeypatch):
    monkeypatch.setattr(settings, "strategy", "smc")
    signal = smc_signal()
    signal["setup_score"] = 90
    signal["score_breakdown"].pop("higher_timeframes")
    filter_setup(signal, {"weekly": {"trend": "BULLISH"}, "monthly": {"trend": "BULLISH"}}, True)
    assert signal["action"] == "BUY" and signal["setup_score"] == 90


def test_scan_restart_keeps_one_entry_per_candle_and_records_closed_feedback(isolated_journal, monkeypatch):
    candles = synthetic_candles(240)
    monkeypatch.setattr(settings, "prediction_model_enabled", False)
    monkeypatch.setattr(settings, "news_filter_enabled", False)
    async def feed():
        return candles, "twelve_data"
    monkeypatch.setattr(engine, "fetch_candles", feed)
    signal = smc_signal()
    signal.update(candle_time=candles.datetime.iloc[-2].isoformat(), price=float(candles.close.iloc[-2]))
    signal.update(confidence=.9, reason="Test setup")
    signal["levels_hint"] = {"stop": signal["price"] - .003, "target": signal["price"] + .003}
    monkeypatch.setattr(engine, "signal_from", lambda data: copy.deepcopy(signal))
    asyncio.run(engine.PaperEngine().scan())
    asyncio.run(engine.PaperEngine().scan())
    original_time = signal["candle_time"]
    signal["candle_time"] = (pd.Timestamp(original_time) - pd.Timedelta(minutes=15)).isoformat()
    asyncio.run(engine.PaperEngine().scan())
    signal["candle_time"] = original_time
    with database.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 1
        trade = dict(db.execute("SELECT * FROM trades").fetchone())
        snapshot = json.loads(trade["signal_snapshot"])
        assert snapshot["training_context"]["features"]
        closed_at = datetime.now(timezone.utc).isoformat()
        monkeypatch.setattr(engine.PaperEngine, "now", staticmethod(lambda: closed_at))
        engine.PaperEngine()._manage_open(db, trade, trade["stop_loss"], trade["stop_loss"],
                                          trade["stop_loss"], closed_at, sampled=True)
    paper, report = journal.examples(None)
    assert report["used"] == 1 and paper.iloc[0].win == 0
    state = engine.PaperEngine().state()
    assert state["stats"]["closed"] == 1 and state["account"]["balance"] < 10000


def test_backtest_uses_multiple_positions_and_shared_loss_pause(monkeypatch):
    candles = synthetic_candles(320)
    monkeypatch.setattr(settings, "strategy", "smc")
    monkeypatch.setattr(settings, "max_concurrent_trades", 3)
    monkeypatch.setattr(settings, "max_daily_loss_percent", 10)
    monkeypatch.setattr(settings, "max_weekly_loss_percent", 10)
    def candidate(data, index):
        signal = smc_signal()
        signal.update(price=float(data.close.iloc[index]), candle_time=data.datetime.iloc[index].isoformat())
        signal["levels_hint"] = {"stop": .5, "target": 2.0}
        return signal
    monkeypatch.setattr(backtest, "_signal_at", candidate)
    result = backtest.run_backtest(candles, backtest.BacktestSettings(10000, 1, 1, 1.5))
    assert result["metrics"]["open_trades"] == 3
    assert result["metrics"]["peak_open_trades"] == 3


def test_training_includes_actual_losses_with_historical_examples(isolated_journal, monkeypatch):
    signal = smc_signal()
    signal["training_context"] = prediction.entry_context(signal, "twelve_data")
    insert_trade(signal)
    times = pd.date_range("2024-01-01", periods=400, freq="D", tz="UTC")
    historical = pd.DataFrame([{"time": stamp.isoformat(), **prediction.features(signal),
                                "gate_score": 90, "win": i % 2, "r": 1.5 if i % 2 else -1}
                               for i, stamp in enumerate(times)])
    candles = synthetic_candles(300)
    monkeypatch.setattr(prediction, "training_history", lambda recent: candles)
    monkeypatch.setattr(prediction, "build_dataset", lambda *args, **kwargs: historical)
    report = prediction.train()
    assert report["samples"] == 401 and report["historical_samples"] == 400
    assert report["paper_learning"]["used"] == 1 and report["paper_learning"]["losses"] == 1
    artifact = json.loads(prediction.MODEL_PATH.read_text())
    assert artifact["learning_version"] == prediction.LEARNING_VERSION
    assert artifact["bias"] < 0  # the actual additional loss changes the fitted base rate
