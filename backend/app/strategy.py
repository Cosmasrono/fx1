import math
import pandas as pd

from .config import settings


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
    bull_score, bear_score = sum(bullish), sum(bearish)
    raw_action = "BUY" if bull_score == 4 else "SELL" if bear_score == 4 else "HOLD"
    trending = bool(math.isfinite(row.adx) and row.adx >= settings.adx_min)
    action = raw_action if raw_action == "HOLD" or trending else "HOLD"
    confidence = max(bull_score, bear_score) / 4
    if raw_action != "HOLD" and not trending:
        reason = f"Entry blocked: ADX {row.adx:.1f} is below the {settings.adx_min:.0f} trend threshold."
    else:
        reason = {"BUY": "Bullish EMA trend, MACD, RSI and momentum agree.",
                  "SELL": "Bearish EMA trend, MACD, RSI and momentum agree.",
                  "HOLD": "Indicators do not yet provide four-way confirmation."}[action]
    values = {key: round(float(row[key]), 6) for key in
              ("ema20", "ema50", "rsi", "macd", "macd_signal", "atr", "adx", "vol_20", "momentum_10", "hl_range", "price_ema20_ratio")}
    values["trend_long"] = int(row.trend_long)
    return {"candle_time": row.datetime.isoformat(), "action": action, "confidence": confidence,
            "price": round(float(row.close), 6), "reason": reason, "indicators": values}


def order_levels(signal: dict, balance: float, risk_percent: float, stop_multiplier: float,
                 target_multiplier: float, execution_cost: float = 0.0) -> dict:
    price, atr = signal["price"], signal["indicators"]["atr"]
    if not math.isfinite(atr) or atr <= 0:
        raise ValueError("ATR is unavailable")
    stop_distance = atr * stop_multiplier + execution_cost
    risk_amount = balance * (risk_percent / 100)
    units = risk_amount / stop_distance
    direction = 1 if signal["action"] == "BUY" else -1
    entry = price + direction * execution_cost
    return {"entry": round(entry, 6), "stop_loss": round(entry-direction*stop_distance, 6),
            "take_profit": round(entry+direction*atr*target_multiplier, 6),
            "units": round(units, 2), "risk_amount": round(risk_amount, 2)}


