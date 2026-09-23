import pytest
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from app.strategy import enrich, market_structure, order_levels, signal_from, trading_session
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


def test_session_classification_uses_utc_liquidity_windows():
    assert trading_session("2026-08-12T08:00:00+00:00") == "LONDON"
    assert trading_session("2026-08-12T13:00:00+00:00") == "OVERLAP"
    assert trading_session("2026-08-12T18:00:00+00:00") == "NEW_YORK"


def test_signal_exposes_score_structure_and_session():
    frame = enrich(candles())
    signal = signal_from(frame)
    assert 0 <= signal["setup_score"] <= 80
    assert signal["indicators"]["structure"] in {"BULLISH", "BEARISH", "MIXED"}
    assert signal["indicators"]["session"] in {"LONDON", "OVERLAP", "NEW_YORK", "ASIAN_OFF_HOURS"}
    assert sum(signal["score_breakdown"].values()) == signal["setup_score"]


def test_order_levels_uses_the_structural_stop_when_a_signal_supplies_one():
    signal = {"action": "BUY", "price": 1.1000, "indicators": {"atr": 0.0020},
              "levels_hint": {"stop": 1.0980, "target": 1.1060}}
    levels = order_levels(signal, balance=10_000, risk_percent=1.0,
                          stop_multiplier=1.0, target_multiplier=1.5, execution_cost=0.00005)
    assert levels["stop_loss"] == 1.098
    assert levels["take_profit"] == 1.106
    # Sizing must use the real invalidation distance, not the 1.0 ATR default.
    assert levels["units"] == pytest.approx(100 / (0.0020 + 0.0001), rel=1e-6)


def test_order_levels_falls_back_to_atr_without_a_hint():
    signal = {"action": "SELL", "price": 1.1000, "indicators": {"atr": 0.0020}}
    levels = order_levels(signal, balance=10_000, risk_percent=1.0,
                          stop_multiplier=1.0, target_multiplier=1.5)
    assert levels["stop_loss"] == pytest.approx(1.1020, abs=1e-6)
    assert levels["take_profit"] == pytest.approx(1.0970, abs=1e-6)
