import numpy as np

from app.backtest import BacktestSettings, _higher_timeframe_context_at, _prepare_higher_timeframe_trends, run_backtest
from app.market import synthetic_candles


def test_backtest_returns_cost_aware_metrics():
    result = run_backtest(synthetic_candles(300), BacktestSettings(10_000, 1, 1, 1.5))

    assert result["bars"] > 200
    assert result["assumptions"] == {"spread_pips": 0.8, "slippage_pips": 0.1, "higher_timeframe_filter": False}
    assert result["metrics"]["starting_balance"] == 10_000
    assert result["metrics"]["max_drawdown_percent"] >= 0
    assert result["diagnostics"]["overall"]["trades"] == result["metrics"]["closed_trades"]
    assert set(result["diagnostics"]["by_side"]) == {"BUY", "SELL"}


def test_higher_timeframe_context_uses_only_completed_trends():
    weekly = synthetic_candles(100, "1week")
    monthly = synthetic_candles(100, "1month")
    for frame in (weekly, monthly):
        close = np.linspace(1.1, 1.3, len(frame)) + np.sin(np.arange(len(frame)) * 1.7) * 0.005
        frame["close"] = close
        frame["open"] = close
        frame["high"] = close + 0.001
        frame["low"] = close - 0.001

    trends = _prepare_higher_timeframe_trends({"weekly": weekly, "monthly": monthly})
    last_completed = max(trends["weekly"][-1][0], trends["monthly"][-1][0])
    context = _higher_timeframe_context_at(trends, last_completed)

    assert context == {"weekly": {"trend": "BULLISH"}, "monthly": {"trend": "BULLISH"}}
