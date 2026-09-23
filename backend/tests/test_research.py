import numpy as np
import pytest

import research
from app.config import settings


def candidate(kind="BULLISH", score=90, price=1.1000, stop=1.0980, target=1.1060):
    return {"kind": kind, "score": score, "price": price, "stop_hint": stop,
            "target_hint": target, "breakdown": {}, "killzone": "LONDON_OPEN",
            "swept_level": stop, "sweep_index": 0, "structure": "CHOCH",
            "poi": "FVG", "equilibrium": price}


def series(length, entry_bar, candidate_dict):
    signals = [None] * length
    signals[entry_bar] = candidate_dict
    return signals


def test_grid_is_unique_and_covers_the_documented_axes():
    grid = research.candidate_grid()
    assert len(grid) == len({tuple(sorted(p.items())) for p in grid})
    assert len(grid) == 3 * 2 * 2 * 3 * 3
    keys = {research.structural_key(p) for p in grid}
    # Only the detection axes force a fresh signal pass; filters are free.
    assert len(keys) == 3 * 2 * 2


def test_simulate_returns_none_without_trades():
    length = 400
    flat = np.full(length, 1.10)
    assert research.simulate([None] * length, flat, flat,
                             {"score_min": 50, "min_reward": 1.5}, 0, length) is None


def test_simulate_takes_profit_and_reports_a_positive_r_multiple():
    length = 400
    entry_bar = research.LOOKBACK + 10
    high = np.full(length, 1.1000)
    low = np.full(length, 1.1000)
    high[entry_bar + 1] = 1.2000  # blows through the 1.1060 target
    result = research.simulate(series(length, entry_bar, candidate()), high, low,
                               {"score_min": 50, "min_reward": 1.5}, 0, length)
    assert result["n"] == 1 and result["win_rate"] == 100.0
    assert result["net"] > 0 and result["returns"][0] > 0


def test_simulate_stops_out_and_caps_the_loss_near_one_r():
    length = 400
    entry_bar = research.LOOKBACK + 10
    high = np.full(length, 1.1000)
    low = np.full(length, 1.1000)
    low[entry_bar + 1] = 1.0000  # blows through the structural stop
    result = research.simulate(series(length, entry_bar, candidate()), high, low,
                               {"score_min": 50, "min_reward": 1.5}, 0, length)
    assert result["n"] == 1 and result["win_rate"] == 0.0
    # Sizing is derived from the stop distance, so a stop-out is about -1R.
    assert result["returns"][0] == pytest.approx(-1.0, abs=0.05)


def test_simulate_respects_the_score_filter():
    length = 400
    entry_bar = research.LOOKBACK + 10
    high = np.full(length, 1.1000); low = np.full(length, 1.1000)
    high[entry_bar + 1] = 1.2000
    signals = series(length, entry_bar, candidate(score=60))
    assert research.simulate(signals, high, low,
                             {"score_min": 90, "min_reward": 1.5}, 0, length) is None
    assert research.simulate(signals, high, low,
                             {"score_min": 50, "min_reward": 1.5}, 0, length)["n"] == 1


def test_simulate_rejects_an_implausibly_wide_structural_stop():
    length = 400
    entry_bar = research.LOOKBACK + 10
    high = np.full(length, 1.1000); low = np.full(length, 1.1000)
    high[entry_bar + 1] = 1.5000
    wide = candidate(stop=1.1000 - research.MAX_RISK_PRICE * 2, target=1.2000)
    assert research.simulate(series(length, entry_bar, wide), high, low,
                             {"score_min": 50, "min_reward": 1.5}, 0, length) is None


def test_simulate_only_trades_inside_the_requested_range():
    length = 800
    entry_bar = 400
    high = np.full(length, 1.1000); low = np.full(length, 1.1000)
    high[entry_bar + 1] = 1.2000
    signals = series(length, entry_bar, candidate())
    inside = research.simulate(signals, high, low,
                               {"score_min": 50, "min_reward": 1.5}, 300, 500)
    outside = research.simulate(signals, high, low,
                                {"score_min": 50, "min_reward": 1.5}, 500, 800)
    assert inside["n"] == 1
    assert outside is None


def test_simulate_holds_one_position_at_a_time():
    length = 400
    first = research.LOOKBACK + 10
    high = np.full(length, 1.1000); low = np.full(length, 1.1000)
    signals = [None] * length
    signals[first] = candidate()
    signals[first + 1] = candidate()  # ignored: the first trade is still open
    high[first + 5] = 1.2000
    result = research.simulate(signals, high, low,
                               {"score_min": 50, "min_reward": 1.5}, 0, length)
    assert result["n"] == 1


def test_execution_cost_tracks_the_configured_spread(monkeypatch):
    monkeypatch.setattr(settings, "spread_pips", 2.0)
    monkeypatch.setattr(settings, "slippage_pips", 0.5)
    monkeypatch.setattr(settings, "pip_size", 0.0001)
    assert research.execution_cost() == pytest.approx(0.00015)
