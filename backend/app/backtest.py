from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from .strategy import enrich, signal_from
from .trend import entry_allowed


@dataclass(frozen=True)
class BacktestSettings:
    starting_balance: float
    risk_percent: float
    stop_multiplier: float
    target_multiplier: float
    spread_pips: float = 0.8
    slippage_pips: float = 0.1
    pip_size: float = 0.0001
    higher_timeframe_filter: bool = False


def run_backtest(
    candles: pd.DataFrame,
    settings: BacktestSettings,
    higher_timeframes: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """Replay closed candles using the same confirmation rules as the live paper engine.

    Entries are adjusted for half the configured spread plus slippage. Stops and targets
    are tested against subsequent candle ranges; if both are touched, the stop wins.
    """
    data = enrich(candles)
    higher_timeframes = higher_timeframes or {}
    higher_trends = _prepare_higher_timeframe_trends(higher_timeframes)
    balance = settings.starting_balance
    equity_peak = balance
    max_drawdown = 0.0
    open_trade = None
    closed = []
    monthly = defaultdict(lambda: {"trades": 0, "pnl": 0.0})
    execution_cost = (settings.spread_pips / 2 + settings.slippage_pips) * settings.pip_size

    for index in range(len(data)):
        row = data.iloc[index]
        candle_time = pd.Timestamp(row.datetime)

        if open_trade:
            exit_price = _exit_price(open_trade, float(row.high), float(row.low), execution_cost)
            if exit_price is not None:
                pnl = (exit_price - open_trade["entry"]) * open_trade["units"] * open_trade["direction"]
                pnl = round(pnl, 2)
                balance += pnl
                closed.append({
                    "side": open_trade["side"],
                    "entry_time": open_trade["entry_time"],
                    "exit_time": candle_time,
                    "pnl": pnl,
                    "reason": "STOP_LOSS" if exit_price == open_trade["stop"] else "TAKE_PROFIT",
                })
                month = candle_time.strftime("%Y-%m")
                monthly[month]["trades"] += 1
                monthly[month]["pnl"] += pnl
                open_trade = None

        if open_trade is None and index < len(data) - 1:
            signal = _signal_at(data, index)
            if signal["action"] in ("BUY", "SELL") and settings.higher_timeframe_filter:
                context = _higher_timeframe_context_at(higher_trends, candle_time)
                signal["indicators"]["weekly_trend"] = context["weekly"]["trend"]
                signal["indicators"]["monthly_trend"] = context["monthly"]["trend"]
                if not entry_allowed(signal["action"], context):
                    signal["action"] = "HOLD"
            if signal["action"] in ("BUY", "SELL"):
                atr = signal["indicators"]["atr"]
                direction = 1 if signal["action"] == "BUY" else -1
                entry = signal["price"] + direction * execution_cost
                stop_distance = atr * settings.stop_multiplier + execution_cost
                risk_amount = balance * settings.risk_percent / 100
                units = risk_amount / stop_distance
                open_trade = {
                    "entry": entry,
                    "stop": entry - direction * stop_distance,
                    "target": entry + direction * atr * settings.target_multiplier,
                    "units": units,
                    "direction": direction,
                    "side": signal["action"],
                    "entry_time": candle_time,
                }

        mark = balance if open_trade is None else balance + (float(row.close) - open_trade["entry"]) * open_trade["units"] * open_trade["direction"]
        equity_peak = max(equity_peak, mark)
        max_drawdown = max(max_drawdown, (equity_peak - mark) / equity_peak * 100)

    wins = [trade["pnl"] for trade in closed if trade["pnl"] > 0]
    losses = [trade["pnl"] for trade in closed if trade["pnl"] < 0]
    gross_profit, gross_loss = sum(wins), abs(sum(losses))
    diagnostics = _diagnostics(closed)
    return {
        "bars": len(data),
        "assumptions": {
            "spread_pips": settings.spread_pips,
            "slippage_pips": settings.slippage_pips,
            "higher_timeframe_filter": settings.higher_timeframe_filter,
        },
        "metrics": {
            "starting_balance": round(settings.starting_balance, 2),
            "ending_balance": round(balance, 2),
            "net_pnl": round(balance - settings.starting_balance, 2),
            "return_percent": round((balance / settings.starting_balance - 1) * 100, 2),
            "closed_trades": len(closed),
            "open_trades": 1 if open_trade else 0,
            "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else None,
            "average_win": round(gross_profit / len(wins), 2) if wins else 0,
            "average_loss": round(sum(losses) / len(losses), 2) if losses else 0,
            "max_drawdown_percent": round(max_drawdown, 2),
        },
        "monthly": [{"month": month, "trades": values["trades"], "pnl": round(values["pnl"], 2)} for month, values in sorted(monthly.items())],
        "diagnostics": diagnostics,
    }


def _signal_at(data: pd.DataFrame, index: int) -> dict:
    # signal_from treats the final row as still forming. Adding one later row makes
    # the indexed row the same fully closed candle that the live engine evaluates.
    return signal_from(data.iloc[: index + 2])


def _exit_price(trade: dict, high: float, low: float, execution_cost: float) -> float | None:
    if trade["direction"] == 1:
        stop_hit, target_hit = low <= trade["stop"], high >= trade["target"]
        if stop_hit:
            return trade["stop"]
        if target_hit:
            return trade["target"] - execution_cost
    else:
        stop_hit, target_hit = high >= trade["stop"], low <= trade["target"]
        if stop_hit:
            return trade["stop"]
        if target_hit:
            return trade["target"] + execution_cost
    return None


def _prepare_higher_timeframe_trends(frames: dict[str, pd.DataFrame]) -> dict[str, list[tuple[pd.Timestamp, str]]]:
    """Make each higher-timeframe value available only after its next candle starts.

    This deliberately delays use of a completed weekly/monthly candle by one
    source timestamp, avoiding the use of a candle's final close before that
    period was actually complete.
    """
    trends: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    for name in ("weekly", "monthly"):
        frame = frames.get(name)
        if frame is None or frame.empty:
            trends[name] = []
            continue
        enriched = enrich(frame)
        points = []
        for index in range(len(enriched) - 1):
            row, next_row = enriched.iloc[index], enriched.iloc[index + 1]
            direction = "BULLISH" if row.ema20 > row.ema50 else "BEARISH"
            points.append((pd.Timestamp(next_row.datetime), direction))
        trends[name] = points
    return trends


def _higher_timeframe_context_at(
    trends: dict[str, list[tuple[pd.Timestamp, str]]], candle_time: pd.Timestamp
) -> dict:
    context = {}
    for name in ("weekly", "monthly"):
        available = [point for point in trends.get(name, []) if point[0] <= candle_time]
        context[name] = {"trend": available[-1][1] if available else "UNAVAILABLE"}
    return context


def _diagnostics(closed: list[dict]) -> dict:
    def summary(trades: list[dict]) -> dict:
        wins = [trade["pnl"] for trade in trades if trade["pnl"] > 0]
        losses = [trade["pnl"] for trade in trades if trade["pnl"] < 0]
        gross_profit, gross_loss = sum(wins), abs(sum(losses))
        count = len(trades)
        average_win = gross_profit / len(wins) if wins else 0
        average_loss = sum(losses) / len(losses) if losses else 0
        breakeven = abs(average_loss) / (average_win + abs(average_loss)) * 100 if average_win and average_loss else None
        return {
            "trades": count,
            "net_pnl": round(sum(trade["pnl"] for trade in trades), 2),
            "win_rate": round(len(wins) / count * 100, 1) if count else 0,
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else None,
            "expectancy_per_trade": round(sum(trade["pnl"] for trade in trades) / count, 2) if count else 0,
            "breakeven_win_rate": round(breakeven, 1) if breakeven is not None else None,
        }

    side = {name: summary([trade for trade in closed if trade["side"] == name]) for name in ("BUY", "SELL")}
    hours = defaultdict(list)
    for trade in closed:
        hours[pd.Timestamp(trade["entry_time"]).hour].append(trade)
    by_hour = [{"hour_utc": hour, **summary(trades)} for hour, trades in sorted(hours.items())]
    streak = longest = 0
    for trade in closed:
        streak = streak + 1 if trade["pnl"] < 0 else 0
        longest = max(longest, streak)
    return {"overall": summary(closed), "by_side": side, "by_hour_utc": by_hour, "max_consecutive_losses": longest}
