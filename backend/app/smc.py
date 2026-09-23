"""Smart Money Concepts / ICT entry model for EUR/USD.

The indicator-confluence model in ``strategy.py`` scores momentum agreement.
This module instead looks for a specific *sequence* of price behaviour that
ICT material calls a dealing-range reversal:

    1. price sweeps a known pool of liquidity (a prior swing high or low) and
       closes back inside the range -- the stop hunt,
    2. an energetic leg (displacement) breaks the opposing swing, which is a
       change of character when it reverses the prevailing structure,
    3. that leg leaves an inefficiency behind -- a fair value gap, or the last
       opposing candle before it, the order block,
    4. price retraces into that point of interest while sitting on the correct
       side of the dealing-range equilibrium (discount to buy, premium to
       sell), ideally inside a killzone.

Only step 4 is an entry. Steps 1-3 are context, so the model is directional by
construction and does not need a separate trend oracle. Every helper is pure
and takes an enriched frame, which keeps them testable and lets the backtest
evaluate a short window instead of the whole history.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import settings
from .strategy import market_structure, trading_session


@dataclass(frozen=True)
class Swing:
    index: int
    price: float
    kind: str  # "HIGH" or "LOW"


@dataclass(frozen=True)
class Zone:
    """A price band left behind by displacement (fair value gap or order block)."""
    index: int
    top: float
    bottom: float
    kind: str  # "BULLISH" or "BEARISH"
    origin: str  # "FVG" or "ORDER_BLOCK"

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


def killzone(value) -> str:
    """ICT killzones in UTC. Outside these windows liquidity is thin."""
    ts = pd.Timestamp(value)
    minutes = ts.hour * 60 + ts.minute
    if 7 * 60 <= minutes < 10 * 60:
        return "LONDON_OPEN"
    if 12 * 60 <= minutes < 15 * 60:
        return "NEW_YORK_OPEN"
    if 15 * 60 <= minutes < 16 * 60:
        return "LONDON_CLOSE"
    return "OUTSIDE"


def swing_points(frame: pd.DataFrame, strength: int = 2) -> list[Swing]:
    """Fractal swings: an extreme with ``strength`` weaker bars on either side.

    The final ``strength`` bars can still be invalidated by bars that have not
    printed yet, so they are deliberately not reported.
    """
    high, low = frame.high.to_numpy(), frame.low.to_numpy()
    points: list[Swing] = []
    for i in range(strength, len(frame) - strength):
        window = slice(i - strength, i + strength + 1)
        if (high[window] < high[i]).sum() == 2 * strength:
            points.append(Swing(i, float(high[i]), "HIGH"))
        if (low[window] > low[i]).sum() == 2 * strength:
            points.append(Swing(i, float(low[i]), "LOW"))
    return points


def fair_value_gaps(frame: pd.DataFrame, start: int = 0) -> list[Zone]:
    """Three-bar inefficiencies where bar 1 and bar 3 ranges do not overlap."""
    high, low = frame.high.to_numpy(), frame.low.to_numpy()
    zones: list[Zone] = []
    for i in range(max(start, 2), len(frame)):
        if low[i] > high[i - 2]:
            zones.append(Zone(i, float(low[i]), float(high[i - 2]), "BULLISH", "FVG"))
        elif high[i] < low[i - 2]:
            zones.append(Zone(i, float(low[i - 2]), float(high[i]), "BEARISH", "FVG"))
    return zones


def order_block(frame: pd.DataFrame, impulse_index: int, kind: str) -> Zone | None:
    """The last opposing candle before the displacement leg that broke structure."""
    open_, close = frame.open.to_numpy(), frame.close.to_numpy()
    high, low = frame.high.to_numpy(), frame.low.to_numpy()
    want_bearish_candle = kind == "BULLISH"
    for i in range(min(impulse_index, len(frame) - 1), max(impulse_index - 12, 0) - 1, -1):
        if (close[i] < open_[i]) == want_bearish_candle:
            return Zone(i, float(high[i]), float(low[i]), kind, "ORDER_BLOCK")
    return None


def displacement_index(frame: pd.DataFrame, start: int, end: int, kind: str,
                       atr: float, factor: float) -> int | None:
    """First bar in ``[start, end]`` whose body is an energetic move in ``kind``."""
    open_, close = frame.open.to_numpy(), frame.close.to_numpy()
    for i in range(max(start, 0), min(end, len(frame) - 1) + 1):
        body = close[i] - open_[i]
        if kind == "BULLISH" and body >= atr * factor:
            return i
        if kind == "BEARISH" and -body >= atr * factor:
            return i
    return None


def dealing_range(swings: list[Swing], upto: int) -> tuple[float, float] | None:
    """Most recent confirmed swing low/high pair, used for premium versus discount."""
    highs = [s for s in swings if s.kind == "HIGH" and s.index <= upto]
    lows = [s for s in swings if s.kind == "LOW" and s.index <= upto]
    if not highs or not lows:
        return None
    return lows[-1].price, highs[-1].price


def liquidity_sweep(frame: pd.DataFrame, swings: list[Swing], kind: str,
                    window: int) -> tuple[Swing, int] | None:
    """Latest run past a swing that closed back inside the range.

    ``kind`` is the resulting trade direction: a BULLISH setup needs sell-side
    liquidity (a swing low) to be taken and then rejected.

    Scanning newest-first returns the most recent sweep and lets the search
    stop as soon as it finds one, which keeps the per-bar cost bounded by
    ``window`` rather than by the whole swing history.
    """
    high, low, close = frame.high.to_numpy(), frame.low.to_numpy(), frame.close.to_numpy()
    last = len(frame) - 1
    pools = [s for s in swings if s.kind == ("LOW" if kind == "BULLISH" else "HIGH")]
    for i in range(last, max(last - window, 0) - 1, -1):
        for pool in reversed(pools):
            if pool.index >= i:
                continue
            if kind == "BULLISH" and low[i] < pool.price and close[i] > pool.price:
                return pool, i
            if kind == "BEARISH" and high[i] > pool.price and close[i] < pool.price:
                return pool, i
    return None


def structure_break(frame: pd.DataFrame, swings: list[Swing], kind: str,
                    after: int) -> tuple[int, str] | None:
    """First close beyond the opposing swing that already existed before ``after``."""
    close = frame.close.to_numpy()
    prior = [s for s in swings if s.kind == ("HIGH" if kind == "BULLISH" else "LOW")
             and s.index < after]
    if not prior:
        return None
    level = prior[-1]
    for i in range(after, len(frame)):
        if kind == "BULLISH" and close[i] > level.price:
            return i, "CHOCH"
        if kind == "BEARISH" and close[i] < level.price:
            return i, "CHOCH"
    return None


@dataclass(frozen=True)
class Context:
    """Swings and gaps precomputed once over a whole series.

    ``evaluate`` normally derives these from the window it is handed, which is
    O(window) per bar. A multi-year replay calls it hundreds of thousands of
    times, so ``prepare`` computes them once and ``evaluate`` slices instead.
    Both paths must produce identical signals; ``test_smc`` asserts that.
    """
    swings: list[Swing]
    gaps: list[Zone]
    strength: int


def prepare(frame: pd.DataFrame, strength: int | None = None) -> Context:
    """Precompute the whole-series swing and gap structure for fast replay."""
    strength = settings.smc_swing_strength if strength is None else strength
    return Context(swing_points(frame, strength), fair_value_gaps(frame), strength)


def _window_context(frame: pd.DataFrame, cutoff: int, offset: int,
                    context: Context | None) -> tuple[list[Swing], list[Zone]]:
    """Swings and gaps visible at ``cutoff``, expressed in window coordinates."""
    if context is None:
        history = frame.iloc[: cutoff + 1]
        return swing_points(history, settings.smc_swing_strength), fair_value_gaps(history)
    # A fractal needs `strength` bars on each side, so a swing at absolute index
    # j is only knowable once j + strength <= the cutoff bar -- and a window
    # can only find it if its `strength` earlier bars are inside the window
    # too. Likewise a gap needs its two earlier bars in the window. Without
    # these lower bounds the replay sees structure the live engine cannot.
    limit = offset + cutoff
    swings = [Swing(s.index - offset, s.price, s.kind) for s in context.swings
              if offset + context.strength <= s.index and s.index + context.strength <= limit]
    gaps = [Zone(z.index - offset, z.top, z.bottom, z.kind, z.origin)
            for z in context.gaps if offset + 2 <= z.index <= limit]
    return swings, gaps


def evaluate(frame: pd.DataFrame, kind: str, context: Context | None = None,
             offset: int = 0) -> dict | None:
    """Score one directional SMC setup against the last fully closed candle.

    ``context`` and ``offset`` are an optional fast path for bulk replay: pass
    the ``prepare`` result for the full series plus this window's start index.
    """
    row = frame.iloc[-2] if len(frame) > 2 else frame.iloc[-1]
    cutoff = len(frame) - 2
    atr = float(row.atr)
    if not np.isfinite(atr) or atr <= 0:
        return None
    history = frame.iloc[: cutoff + 1]
    swings, all_gaps = _window_context(frame, cutoff, offset, context)
    if len(swings) < 2:
        return None

    swept = liquidity_sweep(history, swings, kind, settings.smc_sweep_window)
    if swept is None:
        return None
    pool, sweep_index = swept

    shift = structure_break(history, swings, kind, sweep_index)
    impulse = displacement_index(history, sweep_index, shift[0] if shift else cutoff,
                                 kind, atr, settings.smc_displacement_atr)

    zones = [z for z in all_gaps if z.kind == kind and impulse is not None and shift is not None
             and impulse <= z.index <= shift[0] + 2]
    block = order_block(history, impulse, kind) if impulse is not None else None
    if block is not None:
        zones.append(block)
    price = float(row.close)
    # The entry must be a later retracement, not the displacement/formation bar.
    poi = next((z for z in reversed(zones) if z.index < cutoff and z.contains(price)), None)

    span = dealing_range(swings, cutoff)
    equilibrium = (span[0] + span[1]) / 2 if span else None
    if equilibrium is None:
        discount_ok = False
    else:
        discount_ok = price < equilibrium if kind == "BULLISH" else price > equilibrium
    zone = killzone(row.datetime)

    checks = {
        "liquidity_sweep": (True, 20),
        "structure_shift": (shift is not None, 20),
        "displacement": (impulse is not None, 20),
        "point_of_interest": (poi is not None, 20),
        "premium_discount": (discount_ok, 10),
        "killzone": (zone != "OUTSIDE", 10),
    }
    score = sum(weight for passed, weight in checks.values() if passed)
    breakdown = {name: (weight if passed else 0) for name, (passed, weight) in checks.items()}

    # Structural stop: beyond the wick that took the liquidity, plus a buffer.
    # That level is precisely what the setup claims will not trade through again,
    # which is a more meaningful invalidation point than a fixed ATR distance.
    buffer = atr * settings.smc_stop_buffer_atr
    swept_low = float(history.low.iloc[pool.index: sweep_index + 1].min())
    swept_high = float(history.high.iloc[pool.index: sweep_index + 1].max())
    stop = swept_low - buffer if kind == "BULLISH" else swept_high + buffer
    after_sweep = history.iloc[sweep_index + 1:]
    invalidated = bool((after_sweep.low <= stop).any() if kind == "BULLISH"
                       else (after_sweep.high >= stop).any())
    complete = (shift is not None and impulse is not None and poi is not None
                and sweep_index <= impulse <= shift[0] < cutoff and not invalidated)
    opposing = [s for s in swings if s.kind == ("HIGH" if kind == "BULLISH" else "LOW")]
    return {
        "kind": kind, "score": score, "breakdown": breakdown, "price": price,
        "stop_hint": stop, "target_hint": opposing[-1].price if opposing else None,
        "killzone": zone, "swept_level": pool.price, "sweep_index": sweep_index,
        "structure": shift[1] if shift else None,
        "poi": poi.origin if poi else None, "equilibrium": equilibrium,
        "complete": complete,
        "setup_id": f"smc:{kind}:{history.datetime.iloc[pool.index].isoformat()}:{pool.price:.6f}",
    }


def signal_from(frame: pd.DataFrame, context: Context | None = None,
                offset: int = 0) -> dict:
    """Emit the same signal shape as ``strategy.signal_from`` from SMC context.

    Keeping the contract identical means the engine, backtest, pattern memory
    and probability model all keep working without knowing which model ran.
    """
    row = frame.iloc[-2] if len(frame) > 2 else frame.iloc[-1]
    bullish = evaluate(frame, "BULLISH", context, offset)
    bearish = evaluate(frame, "BEARISH", context, offset)
    candidates = [c for c in (bullish, bearish) if c is not None]
    eligible = [c for c in candidates if c.get("complete", False)]
    best = max(eligible or candidates, key=lambda c: c["score"]) if candidates else None

    values = {key: round(float(row[key]), 6) for key in
              ("ema20", "ema50", "rsi", "macd", "macd_signal", "atr", "adx",
               "vol_20", "momentum_10", "hl_range", "price_ema20_ratio")}
    values["trend_long"] = int(row.trend_long)
    values["structure"] = market_structure(frame)
    values["session"] = trading_session(row.datetime)
    base = {"candle_time": row.datetime.isoformat(), "price": round(float(row.close), 6),
            "indicators": values, "model": "smc"}

    if best is None:
        trend_votes = [row.ema20 > row.ema50, row.macd > row.macd_signal, row.momentum_10 > 0]
        bullish_votes = sum(trend_votes)
        market_bias = "BULLISH" if bullish_votes >= 2 else "BEARISH"
        return {**base, "action": "HOLD", "market_bias": market_bias,
                "bias_score": round(max(bullish_votes, 3 - bullish_votes) / 3 * 100),
                "confidence": 0.0, "setup_score": 0,
                "score_breakdown": {}, "reason": "No liquidity sweep in the active window."}

    action = "BUY" if best["kind"] == "BULLISH" else "SELL"
    market_bias = best["kind"]
    values.update({"killzone": best["killzone"], "smc_structure": best["structure"],
                   "poi": best["poi"], "swept_level": round(best["swept_level"], 6),
                   "equilibrium": round(best["equilibrium"], 6) if best["equilibrium"] else None})
    score, price = best["score"], best["price"]
    direction = 1 if action == "BUY" else -1
    risk = (price - best["stop_hint"]) * direction
    if risk <= 0:
        return {**base, "action": "HOLD", "market_bias": market_bias, "bias_score": score,
                "confidence": score / 100, "setup_score": score,
                "score_breakdown": best["breakdown"],
                "reason": "Entry blocked: the structural stop is on the wrong side of entry."}

    # Prefer the opposing liquidity pool as the target. If that pool is already
    # too close to justify the structural risk, fall back to the configured
    # minimum reward rather than taking a sub-1R trade.
    target = best["target_hint"]
    reward = (target - price) * direction if target is not None else -1
    if reward / risk < settings.smc_min_reward:
        target = price + direction * risk * settings.smc_min_reward
        target_source = "R_MULTIPLE"
    else:
        target_source = "LIQUIDITY"
    values["target_source"] = target_source

    if score < settings.smc_score_min:
        reason = f"Entry blocked: SMC setup score {score}/100 is below {settings.smc_score_min}."
        action = "HOLD"
    elif not best.get("complete", False):
        reason = "Entry blocked: require a sweep, displacement and structure break, followed by a later retracement into a zone; the stop must remain intact."
        action = "HOLD"
    else:
        reason = (f"{action}: swept {best['swept_level']:.5f} liquidity, "
                  f"{best['structure'] or 'no'} structure shift, price in "
                  f"{best['poi'] or 'no'} zone during {best['killzone']}.")
    return {**base, "action": action, "market_bias": market_bias, "bias_score": score,
            "confidence": score / 100, "setup_score": score,
            "score_breakdown": best["breakdown"], "reason": reason,
            "setup_id": best.get("setup_id"),
            "levels_hint": {"stop": round(best["stop_hint"], 6), "target": round(target, 6)}}
