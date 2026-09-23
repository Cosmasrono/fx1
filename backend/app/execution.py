"""Execution and entry rules shared by paper trading and historical replay."""
from datetime import datetime, timedelta, timezone

from .config import settings
from .trend import entry_allowed


def timestamp(value) -> datetime:
    parsed = value.to_pydatetime() if hasattr(value, "to_pydatetime") else (
        value if isinstance(value, datetime) else
        datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def entry_threshold() -> int:
    model_min = {"smc": settings.smc_score_min, "judas": settings.judas_score_min}.get(settings.strategy, 0)
    return max(settings.setup_score_min, model_min)


def filter_setup(signal: dict, context: dict, higher_timeframe_filter: bool) -> None:
    if signal["action"] not in ("BUY", "SELL"):
        return
    if higher_timeframe_filter:
        if not entry_allowed(signal["action"], context):
            signal.update(action="HOLD", reason="Entry blocked: weekly/monthly trends do not agree with the setup.")
            return
        # Structural strategies already score out of 100. Only classic reserves
        # 20 points for higher-timeframe confirmation.
        if settings.strategy == "classic":
            signal["setup_score"] += 20
            signal["score_breakdown"]["higher_timeframes"] = 20
            signal["confidence"] = signal["setup_score"] / 100
    if signal["setup_score"] < entry_threshold():
        signal.update(action="HOLD", reason=f"Entry blocked: setup score is below {entry_threshold()}/100.")


def position_gate(action: str, trades: list[dict]) -> dict:
    if len(trades) >= settings.max_concurrent_trades:
        return {"allowed": False, "reason": f"Entry blocked: maximum concurrent positions reached ({len(trades)}/{settings.max_concurrent_trades})."}
    if any(trade["side"] != action for trade in trades):
        return {"allowed": False, "reason": "Entry blocked: a position in the opposite direction is open."}
    return {"allowed": True, "reason": "Position limits clear."}


def account_risk_gate(trades: list[dict], balance: float, now=None,
                      open_trades: list[dict] | None = None, new_risk: float = 0.0) -> dict:
    current = timestamp(now or datetime.now(timezone.utc))
    day = current.replace(hour=0, minute=0, second=0, microsecond=0)
    week = day - timedelta(days=day.weekday())
    closed = sorted((trade for trade in trades if trade.get("closed_at")
                     and timestamp(trade["closed_at"]) <= current),
                    key=lambda trade: (timestamp(trade["closed_at"]), trade.get("id", 0)))
    daily = sum(t["pnl"] for t in closed if timestamp(t["closed_at"]) >= day)
    weekly = sum(t["pnl"] for t in closed if timestamp(t["closed_at"]) >= week)
    base = {"daily_pnl": round(daily, 2), "weekly_pnl": round(weekly, 2), "resume_at": None}
    if balance <= 0:
        return {**base, "allowed": False, "reason": "Entry blocked: paper balance is exhausted."}
    # Limits use the balance at the beginning of the period, not the already
    # reduced balance. Parsed timestamps accept both SQLite and ISO formats.
    if daily <= -(balance - daily) * settings.max_daily_loss_percent / 100:
        return {**base, "allowed": False, "reason": "Entry blocked: daily loss limit reached.",
                "resume_at": (day + timedelta(days=1)).isoformat()}
    if weekly <= -(balance - weekly) * settings.max_weekly_loss_percent / 100:
        return {**base, "allowed": False, "reason": "Entry blocked: weekly loss limit reached.",
                "resume_at": (week + timedelta(days=7)).isoformat()}
    reserved = sum(max(float(t.get("risk_amount", 0)),
                       abs(t["entry"] - t["stop_loss"]) * t["units"])
                   for t in (open_trades or []))
    base.update(open_risk=round(reserved, 2), proposed_risk=round(new_risk, 2))
    for name, pnl, limit in (("daily", daily, settings.max_daily_loss_percent),
                             ("weekly", weekly, settings.max_weekly_loss_percent)):
        budget = (balance - pnl) * limit / 100 + pnl
        if reserved + new_risk > budget + 0.005:
            return {**base, "allowed": False,
                    "reason": f"Entry blocked: open and proposed risk exceed the remaining {name} loss budget."}
    recent = closed[-settings.max_consecutive_losses:]
    if len(recent) >= settings.max_consecutive_losses and all(t["pnl"] < 0 for t in recent):
        resume = timestamp(recent[-1]["closed_at"]) + timedelta(hours=settings.loss_cooldown_hours)
        if current < resume:
            return {**base, "allowed": False,
                    "reason": f"Entry paused after {settings.max_consecutive_losses} consecutive losses.",
                    "resume_at": resume.isoformat()}
    return {**base, "allowed": True, "reason": "Account risk limits clear."}


def exit_fill(trade: dict, high: float, low: float, cost: float) -> tuple[float, str] | None:
    buy = trade["side"] == "BUY"
    stop, target = trade["stop_loss"], trade["take_profit"]
    if (low <= stop if buy else high >= stop):
        return stop, "STOP_LOSS"
    if (high >= target if buy else low <= target):
        return target - (cost if buy else -cost), "TAKE_PROFIT"
    return None
