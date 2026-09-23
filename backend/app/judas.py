"""ICT Asian range / Judas swing entry model for EUR/USD.

ICT describes the London open as a "Judas swing": early London trade runs the
liquidity resting beyond the quiet Asian session's range -- the stops above its
high or below its low -- then reverses to seek the opposite side.

    1. the Asian session (00:00-06:00 UTC by default) builds a range,
    2. during the London window price trades beyond one side of it -- the sweep,
    3. a candle closes back inside the range -- the false move is confirmed,
       and that close is the entry,
    4. the stop sits beyond the sweep's extreme; the target is the opposite side
       of the Asian range, the liquidity the reversal is expected to seek.

Only the first close back inside after a side is swept is an entry, so each
side trades at most once a day. Hours are UTC and do not follow daylight
saving, like the SMC killzones. The signal shape matches the other entry
models, so the engine, backtest, risk gates and probability model work unchanged.
"""
import numpy as np
import pandas as pd

from .config import settings
from .market import interval_minutes
from .strategy import market_structure, trading_session

WEIGHTS = {"tight_range": 20, "liquidity_sweep": 30, "reclaim": 30, "rejection": 20}


def _hours(value: float) -> pd.Timedelta:
    return pd.Timedelta(hours=value)


def evaluate(history: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Range context and any setups completed by the last candle of ``history``."""
    row = history.iloc[-1]
    stamp = pd.Timestamp(row.datetime)
    day = stamp.normalize()
    range_start = day + _hours(settings.judas_range_start_hour)
    range_end = day + _hours(settings.judas_range_end_hour)
    window_end = day + _hours(settings.judas_window_end_hour)
    context = {"in_window": range_end <= stamp < window_end}

    today = history[history.datetime >= range_start]
    asian = today[today.datetime < range_end]
    expected = (range_end - range_start) / pd.Timedelta(minutes=interval_minutes())
    # A holiday or data gap leaves no meaningful range to trade against.
    if expected <= 0 or len(asian) < 0.75 * expected:
        return context, []
    low, high = float(asian.low.min()), float(asian.high.max())
    context.update(asian_low=low, asian_high=high, range_pips=(high - low) / settings.pip_size)
    atr = float(row.atr)
    if not context["in_window"] or not np.isfinite(atr) or atr <= 0:
        return context, []

    window = today[today.datetime >= range_end]
    opens, highs = window.open.to_numpy(), window.high.to_numpy()
    lows, closes = window.low.to_numpy(), window.close.to_numpy()
    last = len(window) - 1
    inside = (closes > low) & (closes < high)
    setups = []
    for kind in ("BEARISH", "BULLISH"):
        swept = np.flatnonzero(highs > high if kind == "BEARISH" else lows < low)
        if not len(swept):
            continue
        first = swept[0]
        reclaims = np.flatnonzero(inside[first:]) + first
        # Only the first close back inside is the entry; later candles in the
        # same session are the trade already in progress, not a new setup.
        if not len(reclaims) or reclaims[0] != last:
            continue
        if kind == "BEARISH":
            extreme = float(highs[first:last + 1].max())
            body = opens[last] - closes[last]
            stop, target, swept_level = extreme + atr * settings.judas_stop_buffer_atr, low, high
            depth = extreme - high
        else:
            extreme = float(lows[first:last + 1].min())
            body = closes[last] - opens[last]
            stop, target, swept_level = extreme - atr * settings.judas_stop_buffer_atr, high, low
            depth = low - extreme
        checks = {"tight_range": context["range_pips"] <= settings.judas_max_range_pips,
                  "liquidity_sweep": True, "reclaim": True,
                  "rejection": body >= atr * settings.judas_rejection_atr}
        breakdown = {name: WEIGHTS[name] if passed else 0 for name, passed in checks.items()}
        setups.append({"kind": kind, "score": sum(breakdown.values()), "breakdown": breakdown,
                       "price": float(closes[last]), "stop": stop, "target": target,
                       "swept_level": swept_level, "sweep_depth": depth})
    return context, setups


def signal_from(frame: pd.DataFrame) -> dict:
    """Emit the shared signal contract for the last fully closed candle."""
    row = frame.iloc[-2] if len(frame) > 2 else frame.iloc[-1]
    history = frame.iloc[:-1] if len(frame) > 2 else frame
    # One day of candles covers today's Asian range and London window.
    context, setups = evaluate(history.iloc[-(24 * 60 // interval_minutes() + 8):])

    values = {key: round(float(row[key]), 6) for key in
              ("ema20", "ema50", "rsi", "macd", "macd_signal", "atr", "adx",
               "vol_20", "momentum_10", "hl_range", "price_ema20_ratio")}
    values["trend_long"] = int(row.trend_long)
    values["structure"] = market_structure(frame)
    values["session"] = trading_session(row.datetime)
    if "asian_high" in context:
        values.update(asian_high=round(context["asian_high"], 6), asian_low=round(context["asian_low"], 6),
                      range_pips=round(context["range_pips"], 1))
    base = {"candle_time": row.datetime.isoformat(), "price": round(float(row.close), 6),
            "indicators": values, "model": "judas"}

    best = max(setups, key=lambda s: (s["score"], s["sweep_depth"])) if setups else None
    if best is None:
        votes = sum([row.ema20 > row.ema50, row.macd > row.macd_signal, row.momentum_10 > 0])
        window = (f"{settings.judas_range_end_hour:02d}:00-"
                  f"{settings.judas_window_end_hour:02d}:00 UTC")
        reason = (f"Outside the London Judas window ({window})." if not context["in_window"]
                  else "Asian range is incomplete for today." if "asian_high" not in context
                  else "Waiting for London to sweep the Asian range and close back inside.")
        return {**base, "action": "HOLD", "market_bias": "BULLISH" if votes >= 2 else "BEARISH",
                "bias_score": round(max(votes, 3 - votes) / 3 * 100), "confidence": 0.0,
                "setup_score": 0, "score_breakdown": {}, "reason": reason}

    action = "BUY" if best["kind"] == "BULLISH" else "SELL"
    direction = 1 if action == "BUY" else -1
    score, price = best["score"], best["price"]
    values.update(swept_level=round(best["swept_level"], 6),
                  sweep_depth_pips=round(best["sweep_depth"] / settings.pip_size, 1))
    risk = abs(price - best["stop"])
    common = {"market_bias": best["kind"], "bias_score": score, "confidence": score / 100,
              "setup_score": score, "score_breakdown": best["breakdown"]}
    if risk <= 0:
        return {**base, **common, "action": "HOLD",
                "reason": "Entry blocked: the stop sits at the entry price."}
    # The opposite side of the range is the natural target. If it is too close
    # to justify the risk, fall back to the minimum reward instead of a sub-1R trade.
    target = best["target"]
    if (target - price) * direction / risk < settings.judas_min_reward:
        target = price + direction * risk * settings.judas_min_reward
        values["target_source"] = "R_MULTIPLE"
    else:
        values["target_source"] = "LIQUIDITY"
    if score < settings.judas_score_min:
        action = "HOLD"
        reason = f"Entry blocked: Judas setup score {score}/100 is below {settings.judas_score_min}."
    else:
        side = "high" if action == "SELL" else "low"
        reason = (f"{action}: London swept the Asian {side} {best['swept_level']:.5f} "
                  f"and closed back inside the {context['range_pips']:.1f}-pip range.")
    return {**base, **common, "action": action, "reason": reason,
            "levels_hint": {"stop": round(best["stop"], 6), "target": round(target, 6)}}
