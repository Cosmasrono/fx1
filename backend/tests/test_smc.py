import pandas as pd
import pytest

from app import smc
from app.config import settings
from app.market import synthetic_candles
from app.strategy import enrich


def frame_from(bars: list[tuple[float, float, float, float]], start="2026-01-05 08:00") -> pd.DataFrame:
    """Build an OHLC frame on a 15-minute grid inside the London killzone."""
    index = pd.date_range(start=start, periods=len(bars), freq="15min", tz="UTC")
    return pd.DataFrame({"datetime": index,
                         "open": [b[0] for b in bars], "high": [b[1] for b in bars],
                         "low": [b[2] for b in bars], "close": [b[3] for b in bars]})


def test_killzones_cover_the_documented_utc_windows():
    assert smc.killzone("2026-01-05 07:00Z") == "LONDON_OPEN"
    assert smc.killzone("2026-01-05 09:45Z") == "LONDON_OPEN"
    assert smc.killzone("2026-01-05 12:30Z") == "NEW_YORK_OPEN"
    assert smc.killzone("2026-01-05 15:30Z") == "LONDON_CLOSE"
    assert smc.killzone("2026-01-05 03:00Z") == "OUTSIDE"
    assert smc.killzone("2026-01-05 22:00Z") == "OUTSIDE"


def test_swing_points_find_the_fractal_extremes_only():
    bars = [(1.10, 1.101, 1.099, 1.100), (1.10, 1.102, 1.100, 1.101),
            (1.10, 1.110, 1.101, 1.109),  # index 2: swing high
            (1.10, 1.103, 1.099, 1.100), (1.10, 1.102, 1.098, 1.099),
            (1.10, 1.101, 1.090, 1.091),  # index 5: swing low
            (1.10, 1.104, 1.095, 1.103), (1.10, 1.105, 1.098, 1.104)]
    points = smc.swing_points(frame_from(bars), strength=2)
    assert smc.Swing(2, 1.110, "HIGH") in points
    assert smc.Swing(5, 1.090, "LOW") in points
    # The last `strength` bars are unconfirmed and must not be reported.
    assert all(point.index <= len(bars) - 3 for point in points)


def test_fair_value_gap_requires_non_overlapping_first_and_third_bars():
    bullish = frame_from([(1.100, 1.101, 1.099, 1.100),
                          (1.100, 1.108, 1.100, 1.107),
                          (1.107, 1.110, 1.103, 1.109)])  # low 1.103 > high 1.101
    gaps = smc.fair_value_gaps(bullish)
    assert [g.kind for g in gaps] == ["BULLISH"]
    assert (gaps[0].bottom, gaps[0].top) == (1.101, 1.103)

    overlapping = frame_from([(1.100, 1.105, 1.099, 1.100),
                              (1.100, 1.108, 1.100, 1.107),
                              (1.107, 1.110, 1.103, 1.109)])  # low 1.103 < high 1.105
    assert smc.fair_value_gaps(overlapping) == []


def test_bearish_fair_value_gap_is_detected_and_oriented():
    bearish = frame_from([(1.110, 1.111, 1.109, 1.110),
                          (1.110, 1.110, 1.102, 1.103),
                          (1.103, 1.107, 1.100, 1.101)])  # high 1.107 < low 1.109
    gaps = smc.fair_value_gaps(bearish)
    assert [g.kind for g in gaps] == ["BEARISH"]
    assert (gaps[0].bottom, gaps[0].top) == (1.107, 1.109)


def test_zone_contains_is_inclusive_of_its_edges():
    zone = smc.Zone(0, top=1.105, bottom=1.100, kind="BULLISH", origin="FVG")
    assert zone.contains(1.100) and zone.contains(1.105) and zone.contains(1.102)
    assert not zone.contains(1.0999) and not zone.contains(1.1051)


def test_order_block_picks_the_last_opposing_candle_before_the_impulse():
    bars = [(1.100, 1.101, 1.099, 1.1005),   # 0 bullish
            (1.1005, 1.1010, 1.0980, 1.0985),  # 1 bearish  <- expected block
            (1.0985, 1.1080, 1.0984, 1.1075)]  # 2 bullish impulse
    block = smc.order_block(frame_from(bars), impulse_index=2, kind="BULLISH")
    assert block is not None and block.index == 1
    assert (block.bottom, block.top) == (1.0980, 1.1010)


def test_displacement_requires_a_body_of_at_least_the_atr_factor():
    bars = [(1.100, 1.1005, 1.0995, 1.1002),   # small body
            (1.1002, 1.1080, 1.1000, 1.1075)]  # 0.0073 body
    frame = frame_from(bars)
    assert smc.displacement_index(frame, 0, 1, "BULLISH", atr=0.002, factor=0.6) == 1
    assert smc.displacement_index(frame, 0, 1, "BULLISH", atr=0.002, factor=6.0) is None
    # A bullish body must never satisfy a bearish displacement request.
    assert smc.displacement_index(frame, 0, 1, "BEARISH", atr=0.002, factor=0.6) is None


def test_liquidity_sweep_needs_the_close_back_inside_the_range():
    swings = [smc.Swing(1, 1.0990, "LOW")]
    rejected = frame_from([(1.100, 1.101, 1.099, 1.100),
                           (1.100, 1.1005, 1.0990, 1.0995),
                           (1.0995, 1.1002, 1.0980, 1.1001)])  # wicks under, closes above
    assert smc.liquidity_sweep(rejected, swings, "BULLISH", window=10) == (swings[0], 2)

    broke = frame_from([(1.100, 1.101, 1.099, 1.100),
                        (1.100, 1.1005, 1.0990, 1.0995),
                        (1.0995, 1.0998, 1.0980, 1.0982)])  # closes below: not a sweep
    assert smc.liquidity_sweep(broke, swings, "BULLISH", window=10) is None


def test_liquidity_sweep_ignores_events_older_than_the_window():
    swings = [smc.Swing(1, 1.0990, "LOW")]
    bars = [(1.100, 1.101, 1.099, 1.100), (1.100, 1.1005, 1.0990, 1.0995),
            (1.0995, 1.1002, 1.0980, 1.1001)] + [(1.100, 1.1010, 1.0995, 1.1005)] * 8
    frame = frame_from(bars)
    assert smc.liquidity_sweep(frame, swings, "BULLISH", window=10) is not None
    assert smc.liquidity_sweep(frame, swings, "BULLISH", window=3) is None


def test_dealing_range_uses_the_latest_confirmed_pair():
    swings = [smc.Swing(2, 1.110, "HIGH"), smc.Swing(5, 1.090, "LOW"),
              smc.Swing(9, 1.120, "HIGH")]
    assert smc.dealing_range(swings, upto=20) == (1.090, 1.120)
    # A cutoff must hide swings the engine could not have seen yet.
    assert smc.dealing_range(swings, upto=6) == (1.090, 1.110)
    assert smc.dealing_range([smc.Swing(2, 1.11, "HIGH")], upto=20) is None


def test_structure_break_only_counts_swings_that_predate_the_sweep():
    swings = [smc.Swing(1, 1.1050, "HIGH")]
    frame = frame_from([(1.100, 1.101, 1.099, 1.100)] * 3 +
                       [(1.100, 1.1080, 1.0995, 1.1070)])  # closes above 1.1050
    assert smc.structure_break(frame, swings, "BULLISH", after=2) == (3, "CHOCH")
    # A swing that formed after the sweep is not liquidity the sweep could target.
    assert smc.structure_break(frame, [smc.Swing(3, 1.105, "HIGH")], "BULLISH", after=2) is None


def test_signal_holds_when_no_sweep_exists(monkeypatch):
    monkeypatch.setattr(smc, "evaluate", lambda frame, kind, *rest: None)
    frame = enrich(_trending_frame())
    signal = smc.signal_from(frame)
    assert signal["action"] == "HOLD"
    assert signal["setup_score"] == 0
    assert signal["model"] == "smc"


def test_signal_shape_matches_the_classic_strategy_contract():
    signal = smc.signal_from(enrich(_trending_frame()))
    for key in ("candle_time", "action", "confidence", "setup_score",
                "score_breakdown", "price", "reason", "indicators"):
        assert key in signal, key
    for key in ("ema20", "ema50", "rsi", "macd", "macd_signal", "atr", "adx",
                "vol_20", "momentum_10", "session", "structure"):
        assert key in signal["indicators"], key
    assert signal["action"] in ("BUY", "SELL", "HOLD")


def test_low_scoring_setup_is_held_below_the_configured_minimum(monkeypatch):
    candidate = {"kind": "BULLISH", "score": 40, "breakdown": {}, "price": 1.1000,
                 "stop_hint": 1.0980, "target_hint": 1.1060, "killzone": "LONDON_OPEN",
                 "swept_level": 1.0985, "sweep_index": 3, "structure": "CHOCH",
                 "poi": "FVG", "equilibrium": 1.1010}
    monkeypatch.setattr(smc, "evaluate",
                        lambda frame, kind, *rest: candidate if kind == "BULLISH" else None)
    monkeypatch.setattr(settings, "smc_score_min", 70)
    signal = smc.signal_from(enrich(_trending_frame()))
    assert signal["action"] == "HOLD"
    assert "below 70" in signal["reason"]


def test_target_falls_back_to_r_multiple_when_liquidity_is_too_close(monkeypatch):
    # risk 0.0020, liquidity target only 0.0005 away -> 0.25R, below the 1.5R floor.
    candidate = {"complete": True, "kind": "BULLISH", "score": 90, "breakdown": {}, "price": 1.1000,
                 "stop_hint": 1.0980, "target_hint": 1.1005, "killzone": "LONDON_OPEN",
                 "swept_level": 1.0985, "sweep_index": 3, "structure": "CHOCH",
                 "poi": "FVG", "equilibrium": 1.0990}
    monkeypatch.setattr(smc, "evaluate",
                        lambda frame, kind, *rest: candidate if kind == "BULLISH" else None)
    monkeypatch.setattr(settings, "smc_score_min", 70)
    monkeypatch.setattr(settings, "smc_min_reward", 1.5)
    signal = smc.signal_from(enrich(_trending_frame()))
    assert signal["action"] == "BUY"
    assert signal["levels_hint"]["target"] == pytest.approx(1.1030, abs=1e-6)
    assert signal["indicators"]["target_source"] == "R_MULTIPLE"


def test_liquidity_target_is_kept_when_it_clears_the_reward_floor(monkeypatch):
    candidate = {"complete": True, "kind": "BULLISH", "score": 90, "breakdown": {}, "price": 1.1000,
                 "stop_hint": 1.0980, "target_hint": 1.1060, "killzone": "LONDON_OPEN",
                 "swept_level": 1.0985, "sweep_index": 3, "structure": "CHOCH",
                 "poi": "FVG", "equilibrium": 1.0990}
    monkeypatch.setattr(smc, "evaluate",
                        lambda frame, kind, *rest: candidate if kind == "BULLISH" else None)
    monkeypatch.setattr(settings, "smc_score_min", 70)
    signal = smc.signal_from(enrich(_trending_frame()))
    assert signal["levels_hint"]["target"] == pytest.approx(1.1060, abs=1e-6)
    assert signal["indicators"]["target_source"] == "LIQUIDITY"
    assert signal["levels_hint"]["stop"] == pytest.approx(1.0980, abs=1e-6)


def test_the_stronger_side_wins_when_both_directions_qualify(monkeypatch):
    def both(frame, kind, *rest):
        score = 90 if kind == "BEARISH" else 60
        return {"complete": True, "kind": kind, "score": score, "breakdown": {}, "price": 1.1000,
                "stop_hint": 1.1020 if kind == "BEARISH" else 1.0980,
                "target_hint": 1.0940 if kind == "BEARISH" else 1.1060,
                "killzone": "LONDON_OPEN", "swept_level": 1.1015, "sweep_index": 3,
                "structure": "CHOCH", "poi": "FVG", "equilibrium": 1.1010}
    monkeypatch.setattr(smc, "evaluate", both)
    monkeypatch.setattr(settings, "smc_score_min", 70)
    signal = smc.signal_from(enrich(_trending_frame()))
    assert signal["action"] == "SELL"
    assert signal["setup_score"] == 90


def _trending_frame() -> pd.DataFrame:
    """Enough well-formed bars for enrich() to produce finite indicators."""
    bars = []
    price = 1.1000
    for i in range(160):
        drift = 0.0004 if i % 7 < 4 else -0.0003
        open_ = price
        price = round(price + drift, 6)
        bars.append((open_, max(open_, price) + 0.0004, min(open_, price) - 0.0004, price))
    return frame_from(bars, start="2026-01-05 00:00")


def test_prepared_context_matches_the_per_window_computation():
    """The bulk-replay fast path must not change a single signal."""
    data = enrich(_trending_frame())
    context = smc.prepare(data)
    look = 60
    compared = 0
    for i in range(look, len(data) - 1):
        window = data.iloc[i - look:i + 2]
        slow = smc.signal_from(window)
        fast = smc.signal_from(window, context, offset=i - look)
        assert slow == fast, f"divergence at bar {i}"
        compared += 1
    assert compared > 40


def test_prepared_context_ignores_structure_from_before_the_window():
    """Swings and gaps confirmed by bars before the window are invisible live."""
    data = enrich(synthetic_candles(500))
    context = smc.prepare(data)
    look = 120
    for i in range(look, len(data) - 1, 3):
        window = data.iloc[i - look:i + 2]
        assert smc.signal_from(window) == smc.signal_from(window, context, offset=i - look), \
            f"divergence at bar {i}"


def test_prepare_respects_an_explicit_swing_strength():
    data = enrich(_trending_frame())
    loose, tight = smc.prepare(data, strength=2), smc.prepare(data, strength=4)
    assert loose.strength == 2 and tight.strength == 4
    # A wider fractal requirement can only ever confirm fewer swings.
    assert len(tight.swings) <= len(loose.swings)
