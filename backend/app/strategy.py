import math
import pandas as pd

from .config import settings


def trading_session(value) -> str:
    """Classify UTC time into practical EUR/USD liquidity windows."""
    hour = pd.Timestamp(value).hour
    if 12 <= hour < 16:
        return "OVERLAP"
    if 7 <= hour < 12:
        return "LONDON"
    if 16 <= hour < 21:
        return "NEW_YORK"
    return "ASIAN_OFF_HOURS"


def market_structure(frame: pd.DataFrame) -> str:
    """Detect simple higher-high/higher-low or lower-high/lower-low structure."""
    recent = frame.iloc[-8:-1] if len(frame) > 8 else frame.iloc[:-1]
    if len(recent) < 4:
        return "MIXED"
    midpoint = len(recent) // 2
    earlier, later = recent.iloc[:midpoint], recent.iloc[midpoint:]
    if later.high.max() > earlier.high.max() and later.low.min() > earlier.low.min():
        return "BULLISH"
    if later.low.min() < earlier.low.min() and later.high.max() < earlier.high.max():
        return "BEARISH"
    return "MIXED"


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["ema20"] = df.close.ewm(span=20, adjust=False).mean()
    df["ema50"] = df.close.ewm(span=50, adjust=False).mean()
    delta = df.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, float("nan"))))
    fast = df.close.ewm(span=12, adjust=False).mean()
    slow = df.close.ewm(span=26, adjust=False).mean()
    df["macd"] = fast - slow
    df["macd_signal"] = df.macd.ewm(span=9, adjust=False).mean()
    previous = df.close.shift(1)
    true_range = pd.concat([(df.high-df.low), (df.high-previous).abs(), (df.low-previous).abs()], axis=1).max(axis=1)
    df["atr"] = true_range.ewm(alpha=1/14, adjust=False).mean()
    up_move = df.high.diff()
    down_move = -df.low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move
    atr_wilder = true_range.ewm(alpha=1/14, adjust=False).mean().replace(0, float("nan"))
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_wilder
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_wilder
    di_sum = (plus_di + minus_di).replace(0, float("nan"))
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    df["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()
    df["vol_20"] = df.close.pct_change().rolling(20).std()
    df["momentum_10"] = df.close.pct_change(10)
    df["hl_range"] = df.high - df.low
    df["price_ema20_ratio"] = df.close / df.ema20
    df["trend_long"] = (df.ema20 > df.ema50).astype(int)
    return df.dropna().reset_index(drop=True)


def signal_from(frame: pd.DataFrame) -> dict:
    row = frame.iloc[-2] if len(frame) > 2 else frame.iloc[-1]  # last fully closed candle
    bullish = [row.ema20 > row.ema50, row.macd > row.macd_signal, 52 <= row.rsi <= 70, row.momentum_10 > 0]
    bearish = [row.ema20 < row.ema50, row.macd < row.macd_signal, 30 <= row.rsi <= 48, row.momentum_10 < 0]
    structure = market_structure(frame)
    session = trading_session(row.datetime)
    components = {
        "ema_trend": 15, "adx": 15, "macd": 10, "rsi": 10,
        "momentum": 10, "market_structure": 15, "session": 5,
    }
    bull_checks = [bullish[0], row.adx >= settings.adx_min, bullish[1], bullish[2], bullish[3],
                   structure == "BULLISH", session in ("LONDON", "OVERLAP")]
    bear_checks = [bearish[0], row.adx >= settings.adx_min, bearish[1], bearish[2], bearish[3],
                   structure == "BEARISH", session in ("LONDON", "OVERLAP")]
    bull_score = sum(weight for weight, passed in zip(components.values(), bull_checks) if passed)
    bear_score = sum(weight for weight, passed in zip(components.values(), bear_checks) if passed)
    market_bias = "BULLISH" if bull_score > bear_score else "BEARISH" if bear_score > bull_score else "NEUTRAL"
    bias_score = max(bull_score, bear_score)
    local_threshold = max(50, settings.setup_score_min - 20)
    raw_action = "BUY" if bull_score >= local_threshold and bull_score > bear_score else "SELL" if bear_score >= local_threshold and bear_score > bull_score else "HOLD"
    trending = bool(math.isfinite(row.adx) and row.adx >= settings.adx_min)
    action = raw_action if raw_action == "HOLD" or trending else "HOLD"
    setup_score = max(bull_score, bear_score)
    confidence = setup_score / 100
    if raw_action != "HOLD" and not trending:
        reason = f"Entry blocked: ADX {row.adx:.1f} is below the {settings.adx_min:.0f} trend threshold."
    else:
        reason = {"BUY": "Bullish EMA trend, MACD, RSI and momentum agree.",
                  "SELL": "Bearish EMA trend, MACD, RSI and momentum agree.",
                  "HOLD": f"Setup score {setup_score}/100 is below the local entry threshold."}[action]
    values = {key: round(float(row[key]), 6) for key in
              ("ema20", "ema50", "rsi", "macd", "macd_signal", "atr", "adx", "vol_20", "momentum_10", "hl_range", "price_ema20_ratio")}
    values["trend_long"] = int(row.trend_long)
    values["structure"] = structure
    values["session"] = session
    direction_checks = bull_checks if bull_score >= bear_score else bear_checks
    breakdown = {name: weight if passed else 0 for (name, weight), passed in zip(components.items(), direction_checks)}
    return {"candle_time": row.datetime.isoformat(), "action": action,
            "market_bias": market_bias, "bias_score": bias_score, "confidence": confidence,
            "setup_score": setup_score, "score_breakdown": breakdown,
            "price": round(float(row.close), 6), "reason": reason, "indicators": values}


def order_levels(signal: dict, balance: float, risk_percent: float, stop_multiplier: float,
                 target_multiplier: float, execution_cost: float = 0.0) -> dict:
    price = signal.get("execution_price", signal["price"])
    direction = 1 if signal["action"] == "BUY" else -1
    entry = price + direction * execution_cost
    hint = signal.get("levels_hint")
    if hint:
        # SMC supplies a structural stop beyond the swept wick. Size from that
        # real invalidation distance instead of a fixed ATR multiple, and keep
        # the cost inside the sizing distance so the modelled loss is not
        # understated.
        stop_loss, take_profit = hint["stop"], hint["target"]
        if ((entry - stop_loss) * direction <= 0
                or (take_profit - entry) * direction <= execution_cost):
            raise ValueError("Entry price is outside the structural stop/target range")
        stop_distance = abs(entry - stop_loss) + execution_cost
    else:
        atr = signal["indicators"]["atr"]
        if not math.isfinite(atr) or atr <= 0:
            raise ValueError("ATR is unavailable")
        stop_distance = atr * stop_multiplier + execution_cost
        stop_loss = entry - direction * stop_distance
        take_profit = entry + direction * atr * target_multiplier
    if not math.isfinite(stop_distance) or stop_distance <= 0:
        raise ValueError("Stop distance is unavailable")
    risk_amount = balance * (risk_percent / 100)
    units = risk_amount / stop_distance
    return {"entry": round(entry, 6), "stop_loss": round(stop_loss, 6),
            "take_profit": round(take_profit, 6),
            "units": round(units, 2), "risk_amount": round(risk_amount, 2)}
