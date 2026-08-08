from datetime import datetime, timezone
import numpy as np
import pandas as pd
from app.strategy import enrich, order_levels, signal_from
from app.learning import assess_signal
from app.trend import entry_allowed
import sqlite3
import json


def candles(rising=True):
    trend = np.linspace(1.10, 1.16, 120) if rising else np.linspace(1.16, 1.10, 120)
    close = trend + np.sin(np.arange(120) * 1.7) * 0.0008
    return pd.DataFrame({"datetime": pd.date_range(datetime.now(timezone.utc), periods=120, freq="15min"),
                         "open": close, "high": close + .0005, "low": close - .0005, "close": close})


def test_indicators_and_signal_are_finite():
    signal = signal_from(enrich(candles()))
    assert signal["action"] in {"BUY", "SELL", "HOLD"}
    assert signal["indicators"]["atr"] > 0
    assert signal["indicators"]["adx"] >= 0


def test_adx_floor_blocks_entries_in_range():
    flat = np.full(120, 1.15) + np.sin(np.arange(120) * 0.9) * 0.00005
    frame = pd.DataFrame({"datetime": pd.date_range(datetime.now(timezone.utc), periods=120, freq="15min"),
                          "open": flat, "high": flat + .00005, "low": flat - .00005, "close": flat})
    signal = signal_from(enrich(frame))
    assert signal["action"] == "HOLD"


def test_risk_is_one_percent():
    signal = signal_from(enrich(candles()))
    signal["action"] = "BUY"
    levels = order_levels(signal, 10_000, 1, 1, 1.5)
    assert levels["risk_amount"] == 100
    assert levels["stop_loss"] < levels["entry"] < levels["take_profit"]


def test_repeated_matching_losses_block_an_entry():
    signal = signal_from(enrich(candles()))
    signal["action"] = "BUY"
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE trade_patterns (id INTEGER PRIMARY KEY, side TEXT, outcome TEXT, features TEXT)")
    for _ in range(3):
        db.execute("INSERT INTO trade_patterns (side,outcome,features) VALUES (?,?,?)",
                   ("BUY", "LOSS", json.dumps(signal["indicators"])))
    result = assess_signal(db, signal, min_loss_matches=3, similarity_threshold=.85)
    assert result["blocked"] is True


def test_higher_timeframe_filter_requires_direction_agreement():
    bullish = {"weekly": {"trend": "BULLISH"}, "monthly": {"trend": "BULLISH"}}
    mixed = {"weekly": {"trend": "BULLISH"}, "monthly": {"trend": "BEARISH"}}
    unavailable = {"weekly": {"trend": "BULLISH"}, "monthly": {"trend": "UNAVAILABLE"}}
    assert entry_allowed("BUY", bullish) is True
    assert entry_allowed("SELL", bullish) is False
    assert entry_allowed("BUY", mixed) is False
    assert entry_allowed("BUY", unavailable) is False
