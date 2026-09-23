"""Read completed paper trades as labelled training examples; never modify the journal."""
import json
import sqlite3

import numpy as np
import pandas as pd

from . import database
from .config import settings


def closed_trades() -> list[dict]:
    if not database.DB_PATH.exists():
        return []
    with sqlite3.connect(database.DB_PATH.as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='trades'").fetchone():
            return []
        return [dict(row) for row in db.execute("SELECT * FROM trades WHERE status='CLOSED' ORDER BY id")]


def examples(candles: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    from .prediction import features, feature_names, fingerprint, _embargo
    from .market import interval_minutes
    names = feature_names(settings.strategy)
    trades = closed_trades()
    rows, skipped = [], {}
    prices = {} if candles is None else dict(zip(pd.to_datetime(candles.datetime, utc=True), candles.close))
    current = pd.Timestamp.now(tz="UTC")
    legacy = 0
    for trade in trades:
        why = None
        try:
            signal = json.loads(trade["signal_snapshot"])
            context = signal.get("training_context", {})
            stamp = pd.Timestamp(signal["candle_time"])
            stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
            closed = pd.to_datetime(trade["closed_at"], utc=True)
            if signal.get("execution_revision", 1) < 2:
                closed += pd.Timedelta(minutes=interval_minutes())
            if signal.get("model", "classic") != settings.strategy:
                why = "different_strategy"
            elif context and (context.get("fingerprint") != fingerprint()
                              or context.get("feed") != ("synthetic" if settings.use_synthetic_data else "twelve_data")):
                why = "different_settings_or_feed"
            elif not context and (settings.use_synthetic_data or stamp not in prices
                                  or abs(float(prices[stamp]) - float(signal["price"])) > 0.000002):
                # Old snapshots lack provenance. Admit them only if the entry
                # price and candle timestamp match this symbol's real history.
                why = "legacy_price_not_verified"
            elif closed > current or closed <= stamp or closed > stamp + _embargo():
                why = "outside_outcome_horizon"
            elif trade["reason"] not in ("STOP_LOSS", "TAKE_PROFIT") or trade["risk_amount"] <= 0:
                why = "unsupported_outcome"
            if why is None:
                vector = context.get("features") or features(signal)
                values = [float(vector[name]) for name in names]
                result = float(trade["pnl"]) / float(trade["risk_amount"])
                if not np.isfinite(values + [result]).all():
                    raise ValueError("Non-finite entry features")
                rows.append({"time": stamp.isoformat(), **dict(zip(names, values)),
                             "gate_score": sum(v for k, v in signal.get("score_breakdown", {}).items()
                                               if k != "higher_timeframes") or signal["setup_score"],
                             "win": int(trade["reason"] == "TAKE_PROFIT"), "r": result,
                             "source": "paper", "trade_id": trade["id"], "resolved_at": closed.isoformat()})
                legacy += not bool(context)
        except (KeyError, TypeError, ValueError, OverflowError):
            why = "incomplete_snapshot"
        if why:
            skipped[why] = skipped.get(why, 0) + 1
    return pd.DataFrame(rows), {"closed_trades": len(trades), "used": len(rows),
                               "wins": sum(row["win"] for row in rows),
                               "losses": sum(1 - row["win"] for row in rows),
                               "legacy_verified": int(legacy), "skipped": skipped,
                               "examined_trade_ids": [t["id"] for t in trades]}


def merge_examples(historical: pd.DataFrame, paper: pd.DataFrame) -> pd.DataFrame:
    historic = historical.copy()
    historic["source"] = "history"
    parts = [frame for frame in (historic, paper) if not frame.empty]
    if not parts:
        return historic
    combined = pd.concat(parts, ignore_index=True)
    combined["time"] = pd.to_datetime(combined.time, utc=True, format="mixed").map(lambda t: t.isoformat())
    # An actual outcome replaces the replay of that same entry; it must not
    # appear in training twice or on opposite sides of a validation boundary.
    return (combined.drop_duplicates(["time", "side"], keep="last")
            .sort_values("time", kind="stable").reset_index(drop=True))
