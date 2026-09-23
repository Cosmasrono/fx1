import numpy as np
import pandas as pd
import pytest

from app import active_strategy, judas, prediction
from app.config import settings
from app.strategy import enrich

DAY = pd.Timestamp("2026-01-06 00:00", tz="UTC")


def quiet(count: int, low: float, high: float) -> list[tuple]:
    """Alternating candles that stay well inside [low, high]."""
    mid, half = (low + high) / 2, (high - low) * 0.15
    bars = []
    for i in range(count):
        open_, close = (mid - half, mid + half) if i % 2 else (mid + half, mid - half)
        bars.append((open_, max(open_, close) + 0.0001, min(open_, close) - 0.0001, close))
    return bars


def session(london: list[tuple], low=1.1000, high=1.1020, filler: int = 0) -> pd.DataFrame:
    """A prior day, an Asian range touching exactly low/high, then London candles.

    ``filler`` quiet London candles come before ``london``; one forming candle
    is appended so the last ``london`` candle is the one evaluated.
    """
    mid = (low + high) / 2
    asian = quiet(24, low, high)
    asian[3] = (mid, high, mid - 0.0001, mid)
    asian[7] = (mid, mid + 0.0001, low, mid)
    bars = quiet(60, low, high) + asian + quiet(filler, low, high) + london + [(mid, mid, mid, mid)]
    index = pd.date_range(end=DAY + pd.Timedelta(minutes=15 * (len(bars) - 61)), periods=len(bars),
                          freq="15min")
    frame = pd.DataFrame({"datetime": index, "open": [b[0] for b in bars],
                          "high": [b[1] for b in bars], "low": [b[2] for b in bars],
                          "close": [b[3] for b in bars]})
    return enrich(frame)


SELL_SWEEP = [(1.1015, 1.1018, 1.1012, 1.1016),
              (1.1016, 1.1030, 1.1015, 1.1026),   # runs the Asian high, closes above it
              (1.1026, 1.1027, 1.1008, 1.1010)]   # closes back inside: the entry
BUY_SWEEP = [(1.1005, 1.1008, 1.1002, 1.1004),
             (1.1004, 1.1005, 1.0990, 1.0994),
             (1.0994, 1.1012, 1.0993, 1.1010)]


def test_session_fixture_places_the_asian_range_before_london():
    data = session(SELL_SWEEP)
    assert data.datetime.iloc[-4] == DAY + pd.Timedelta(hours=6)


def test_london_sweep_of_the_asian_high_that_closes_back_inside_is_a_sell():
    signal = judas.signal_from(session(SELL_SWEEP))
    assert signal["action"] == "SELL" and signal["model"] == "judas"
    assert signal["indicators"]["asian_high"] == pytest.approx(1.1020)
    assert signal["indicators"]["range_pips"] == pytest.approx(20.0)
    assert signal["setup_score"] == 100
    assert signal["levels_hint"]["stop"] > 1.1030  # beyond the sweep's extreme
    assert signal["levels_hint"]["target"] < signal["price"]


def test_london_sweep_of_the_asian_low_is_the_mirror_buy():
    signal = judas.signal_from(session(BUY_SWEEP))
    assert signal["action"] == "BUY"
    assert signal["levels_hint"]["stop"] < 1.0990
    assert signal["levels_hint"]["target"] > signal["price"]


def test_target_is_the_opposite_side_when_it_clears_the_minimum_reward():
    shallow = [(1.1022, 1.1027, 1.1021, 1.1026), (1.1026, 1.1026, 1.1021, 1.1022)]
    signal = judas.signal_from(session(shallow, low=1.1000, high=1.1025))
    assert signal["action"] == "SELL"
    assert signal["indicators"]["target_source"] == "LIQUIDITY"
    assert signal["levels_hint"]["target"] == pytest.approx(1.1000)


def test_target_falls_back_to_the_minimum_reward_when_the_range_is_too_close():
    signal = judas.signal_from(session(SELL_SWEEP))
    risk = signal["levels_hint"]["stop"] - signal["price"]
    assert signal["indicators"]["target_source"] == "R_MULTIPLE"
    assert signal["price"] - signal["levels_hint"]["target"] == pytest.approx(
        risk * settings.judas_min_reward, rel=1e-3)


def test_only_the_first_close_back_inside_is_an_entry():
    later = SELL_SWEEP + [(1.1010, 1.1012, 1.1006, 1.1008)]
    assert judas.signal_from(session(later))["action"] == "HOLD"


def test_no_entry_after_the_london_window_closes():
    signal = judas.signal_from(session(SELL_SWEEP, filler=16))  # entry candle at 10:30
    assert signal["action"] == "HOLD"
    assert "Outside" in signal["reason"]
    assert signal["indicators"]["asian_high"] == pytest.approx(1.1020)


def test_wide_range_without_a_rejection_candle_is_held_below_the_minimum():
    weak = [(1.1035, 1.1044, 1.1034, 1.1043), (1.1041, 1.1044, 1.1037, 1.1039)]  # 2-pip body
    signal = judas.signal_from(session(weak, low=1.1000, high=1.1040))
    assert signal["action"] == "HOLD"
    assert signal["setup_score"] == 60
    assert "below" in signal["reason"]


def test_signal_shape_matches_the_other_entry_models():
    signal = judas.signal_from(session(SELL_SWEEP))
    for key in ("candle_time", "action", "market_bias", "bias_score", "confidence", "setup_score",
                "score_breakdown", "price", "reason", "indicators", "levels_hint"):
        assert key in signal
    for key in ("atr", "rsi", "adx", "session", "structure", "ema20", "ema50"):
        assert key in signal["indicators"]


def test_dispatcher_runs_the_judas_model(monkeypatch):
    monkeypatch.setattr(settings, "strategy", "judas")
    assert active_strategy.signal_from(session(SELL_SWEEP))["model"] == "judas"


def test_prediction_features_cover_the_judas_inputs():
    row = prediction.features(judas.signal_from(session(SELL_SWEEP)))
    names = prediction.feature_names("judas")
    assert set(names) <= set(row)
    assert np.isfinite([row[name] for name in names]).all()
    assert row["setup_score"] == 100 and row["tight_range"] == 1.0
