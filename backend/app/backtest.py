from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from .active_strategy import active_name, signal_from
from .strategy import enrich, order_levels
from .config import settings as app_settings
from .execution import account_risk_gate, position_gate, filter_setup, exit_fill
from .learning import assess_examples, outcome_features
from .market import interval_minutes


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
    mark = balance
    equity_peak = balance
    max_drawdown = 0.0
    open_trades = []
    closed = []
    patterns = []
    next_id = 1
    peak_positions = 0
    used_setups = set()
    monthly = defaultdict(lambda: {"trades": 0, "pnl": 0.0})
    execution_cost = (settings.spread_pips / 2 + settings.slippage_pips) * settings.pip_size

    # The last provider row may still be forming and cannot settle a trade.
    for index in range(max(0, len(data) - 1)):
        row = data.iloc[index]
        candle_time = pd.Timestamp(row.datetime)
        decision_time = candle_time + pd.Timedelta(minutes=interval_minutes())

        for open_trade in list(open_trades):
            fill = exit_fill(open_trade, float(row.high), float(row.low), execution_cost)
            if fill is not None:
                exit_price, reason = fill
                pnl = (exit_price - open_trade["entry"]) * open_trade["units"] * open_trade["direction"]
                pnl = round(pnl, 2)
                balance += pnl
                closed.append({
                    "id": open_trade["id"],
                    "side": open_trade["side"],
                    "entry_time": open_trade["entry_time"],
                    "exit_time": candle_time,
                    "closed_at": decision_time.isoformat(),
                    "pnl": pnl,
                    "reason": reason,
                    "session": open_trade["session"],
                })
                month = candle_time.strftime("%Y-%m")
                monthly[month]["trades"] += 1
                monthly[month]["pnl"] += pnl
                patterns.append({"side": open_trade["side"], "outcome": "WIN" if pnl > 0 else "LOSS",
                                 "features": outcome_features(open_trade["signal"]), "pnl": pnl,
                                 "closed_at": decision_time.isoformat()})
                open_trades.remove(open_trade)

        if index >= 2:
            signal = _signal_at(data, index)
            signal["execution_price"] = float(data.open.iloc[index + 1])
            decision_time = pd.Timestamp(data.datetime.iloc[index + 1])
            signal["model"] = active_name()
            context = _higher_timeframe_context_at(higher_trends, decision_time)
            signal["indicators"]["weekly_trend"] = context["weekly"]["trend"]
            signal["indicators"]["monthly_trend"] = context["monthly"]["trend"]
            filter_setup(signal, context, settings.higher_timeframe_filter)
            if signal["action"] in ("BUY", "SELL"):
                memory = assess_examples(patterns, signal, app_settings.pattern_min_loss_matches,
                                         app_settings.pattern_similarity_threshold, decision_time)
                allowed = (not memory["blocked"] and account_risk_gate(
                    closed, balance, decision_time, open_trades, balance * settings.risk_percent / 100)["allowed"]
                           and (not signal.get("setup_id") or signal["setup_id"] not in used_setups)
                           and position_gate(signal["action"], open_trades)["allowed"])
                if not allowed:
                    signal["action"] = "HOLD"
            if signal["action"] in ("BUY", "SELL"):
                try:
                    levels = order_levels(signal, balance, settings.risk_percent,
                                          settings.stop_multiplier, settings.target_multiplier, execution_cost)
                except ValueError:
                    levels = None
                if levels is not None:
                    direction = 1 if signal["action"] == "BUY" else -1
                    open_trades.append({
                        **levels, "id": next_id, "signal": signal,
                        "direction": direction,
                        "side": signal["action"],
                        "entry_time": decision_time,
                        "session": signal["indicators"]["session"],
                    })
                    next_id += 1
                    if signal.get("setup_id"):
                        used_setups.add(signal["setup_id"])
                    peak_positions = max(peak_positions, len(open_trades))

        mark = balance + sum((float(data.open.iloc[index + 1]) - trade["entry"]) * trade["units"] * trade["direction"] for trade in open_trades)
        equity_peak = max(equity_peak, mark)
        max_drawdown = max(max_drawdown, (equity_peak - mark) / equity_peak * 100)

    wins = [trade["pnl"] for trade in closed if trade["pnl"] > 0]
    losses = [trade["pnl"] for trade in closed if trade["pnl"] < 0]
    gross_profit, gross_loss = sum(wins), abs(sum(losses))
    diagnostics = _diagnostics(closed)
    return {
        "bars": len(data),
        "assumptions": {
            "strategy": active_name(),
            "spread_pips": settings.spread_pips,
            "slippage_pips": settings.slippage_pips,
            "higher_timeframe_filter": settings.higher_timeframe_filter,
            "max_concurrent_trades": app_settings.max_concurrent_trades,
            "risk_percent": settings.risk_percent,
            "account_risk_limits": True,
            "entry_pricing": "Next candle open plus costs; live uses its first available sampled price",
            "setup_deduplication": True,
            "open_risk_reserved": True,
            "pattern_memory": "Learns only from earlier exits in this replay",
            "excluded_gates": ["Economic news calendar", "Prediction model (requires models trained before each replay period)",
                               "Live feed freshness checks"],
        },
        "metrics": {
            "starting_balance": round(settings.starting_balance, 2),
            "ending_balance": round(balance, 2),
            "ending_equity": round(mark, 2),
            "equity_return_percent": round((mark / settings.starting_balance - 1) * 100, 2),
            "net_pnl": round(balance - settings.starting_balance, 2),
            "return_percent": round((balance / settings.starting_balance - 1) * 100, 2),
            "closed_trades": len(closed),
            "open_trades": len(open_trades),
            "peak_open_trades": peak_positions,
            "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else None,
            "average_win": round(gross_profit / len(wins), 2) if wins else 0,
            "average_loss": round(sum(losses) / len(losses), 2) if losses else 0,
            "max_drawdown_percent": round(max_drawdown, 2),
        },
        "monthly": [{"month": month, "trades": values["trades"], "pnl": round(values["pnl"], 2)} for month, values in sorted(monthly.items())],
        "diagnostics": diagnostics,
    }


# Bars of context handed to the entry model at each step. The classic model
# reads only the last few rows, and the SMC model scans swings across the
# window, so a bounded slice is equivalent to the full prefix while keeping the
# replay linear instead of quadratic. It also matches what the live engine sees,
# which fetches a fixed number of candles rather than all history.
SIGNAL_LOOKBACK = 250


def _signal_at(data: pd.DataFrame, index: int) -> dict:
    # signal_from treats the final row as still forming. Adding one later row makes
    # the indexed row the same fully closed candle that the live engine evaluates.
    start = max(0, index + 2 - SIGNAL_LOOKBACK)
    return signal_from(data.iloc[start: index + 2])


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
    sessions = defaultdict(list)
    for trade in closed:
        sessions[trade.get("session", "UNKNOWN")].append(trade)
    by_session = {name: summary(trades) for name, trades in sorted(sessions.items())}
    streak = longest = 0
    for trade in closed:
        streak = streak + 1 if trade["pnl"] < 0 else 0
        longest = max(longest, streak)
    return {"overall": summary(closed), "by_side": side, "by_hour_utc": by_hour,
            "by_session": by_session, "max_consecutive_losses": longest}
