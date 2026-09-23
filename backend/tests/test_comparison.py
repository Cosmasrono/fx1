import pandas as pd
import numpy as np
import pytest

import compare_strategies as comparison
from app import backtest
from app.config import settings
from app.market import synthetic_candles


def report(trades, equity_return):
    return {"metrics": {"closed_trades": trades, "equity_return_percent": equity_return}}


def test_selection_abstains_on_losses_or_too_few_trades():
    assert comparison.choose_development({"classic": report(100, -1), "smc": report(5, 20)}) is None
    assert comparison.choose_development({"classic": report(50, 1), "smc": report(50, 2)}) == "smc"


def test_later_winner_cannot_replace_the_earlier_choice(monkeypatch):
    data = pd.DataFrame({"datetime": pd.date_range("2025-01-01", periods=1600, freq="15min", tz="UTC")})
    calls = []
    def replay(frame, name, start, config, higher):
        calls.append(name)
        if len(calls) <= 3:
            return report(50, 2 if name == "classic" else -1)
        return report(50, -2 if name == "classic" else 10)
    monkeypatch.setattr(comparison, "replay", replay)
    result = comparison.compare(data, backtest.BacktestSettings(10000, 1, 1, 1.5), progress=lambda _: None)
    assert result["selected_on_development"] == "classic"
    assert "lost equity" in result["conclusion"]
    assert calls == ["classic", "smc", "judas"] * 2


def test_warmup_cannot_trade_and_process_settings_restore_on_error(monkeypatch):
    start = pd.Timestamp("2026-01-05T10:00:00Z")
    frame = pd.DataFrame({"datetime": [start - pd.Timedelta(minutes=15), start]})
    def signal(data, index):
        return {"candle_time": data.datetime.iloc[index].isoformat(), "action": "BUY"}
    monkeypatch.setattr(backtest, "_signal_at", signal)
    previous = settings.strategy
    with pytest.raises(RuntimeError):
        with comparison.isolated_strategy("judas", start):
            assert settings.strategy == "judas"
            assert backtest._signal_at(frame, 0)["action"] == "HOLD"
            assert backtest._signal_at(frame, 1)["action"] == "BUY"
            raise RuntimeError("test restoration")
    assert settings.strategy == previous and backtest._signal_at is signal


def test_periods_only_share_pre_validation_warmup():
    frame = pd.DataFrame({"datetime": pd.date_range("2025-01-01", periods=1600, freq="15min", tz="UTC")})
    development, validation, start = comparison.periods(frame)
    assert development.datetime.max() < start
    assert (validation.datetime < start).sum() == 300
    assert (validation.datetime >= start).sum() == 400


def test_comparison_equity_includes_an_unclosed_losing_position(monkeypatch):
    candles = synthetic_candles(300)
    candles["close"] = np.linspace(1.2, 1.1, len(candles))
    candles["open"] = candles.close
    candles["high"], candles["low"] = candles.close + .0001, candles.close - .0001
    def signal(data, index):
        return {"action": "BUY", "price": float(data.close.iloc[index]),
                "candle_time": data.datetime.iloc[index].isoformat(), "setup_id": "one",
                "setup_score": 100, "score_breakdown": {},
                "indicators": {"session": "LONDON", "atr": .001},
                "levels_hint": {"stop": .5, "target": 2.0}}
    monkeypatch.setattr(backtest, "_signal_at", signal)
    result = backtest.run_backtest(candles, backtest.BacktestSettings(10000, 1, 1, 1.5))
    metrics = result["metrics"]
    assert metrics["closed_trades"] == 0 and metrics["open_trades"] == 1
    assert metrics["return_percent"] == 0
    assert metrics["ending_equity"] < metrics["ending_balance"]
    assert metrics["equity_return_percent"] < 0
