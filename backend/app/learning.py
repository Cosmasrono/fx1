"""Small, explainable memory of paper-trade outcomes.

This is intentionally a pattern guard, not a claim to predict markets.  It
only prevents a new paper entry after several *very similar* previous entries
ended in losses.
"""
import json
from datetime import datetime, timedelta, timezone

from .config import settings


def outcome_features(snapshot: dict) -> dict:
    values = dict(snapshot.get("indicators", {}))
    values["_strategy"] = snapshot.get("model", "classic")
    values["_feed"] = snapshot.get("training_context", {}).get("feed")
    values["_structure"] = {name: value > 0 for name, value in
                            snapshot.get("score_breakdown", {}).items()
                            if name != "higher_timeframes"}
    return values


def record_outcome(db, trade: dict, pnl: float, reason: str, closed_at: str) -> None:
    snapshot = json.loads(trade["signal_snapshot"])
    db.execute(
        """INSERT OR IGNORE INTO trade_patterns
           (trade_id, side, outcome, pnl, exit_reason, features, closed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (trade["id"], trade["side"], "WIN" if pnl > 0 else "LOSS", pnl, reason,
         json.dumps(outcome_features(snapshot)), closed_at),
    )


def assess_signal(db, signal: dict, min_loss_matches: int, similarity_threshold: float) -> dict:
    """Return the evidence from comparable historical paper trades."""
    rows = db.execute(
        "SELECT * FROM trade_patterns WHERE side=? ORDER BY id DESC",
        (signal["action"],),
    ).fetchall()
    return assess_examples([dict(row) for row in rows], signal, min_loss_matches, similarity_threshold)


def assess_examples(rows: list[dict], signal: dict, min_loss_matches: int,
                    similarity_threshold: float, now=None) -> dict:
    """Use only past, recent outcomes from the same strategy and feed."""
    from .execution import timestamp
    current = timestamp(now or datetime.now(timezone.utc))
    earliest = current - timedelta(days=settings.pattern_lookback_days)
    comparisons = []
    for row in rows:
        if row.get("side", signal["action"]) != signal["action"]:
            continue
        if row.get("closed_at") and not earliest <= timestamp(row["closed_at"]) <= current:
            continue
        historic = json.loads(row["features"]) if isinstance(row["features"], str) else row["features"]
        if historic.get("_strategy", signal.get("model", "classic")) != signal.get("model", "classic"):
            continue
        feed = signal.get("training_context", {}).get("feed")
        if feed and historic.get("_feed") and historic["_feed"] != feed:
            continue
        score = _similarity(outcome_features(signal), historic)
        if score >= similarity_threshold:
            comparisons.append((score, row["outcome"], row.get("pnl", 0)))

    losses = sum(outcome == "LOSS" for _, outcome, _ in comparisons)
    wins = sum(outcome == "WIN" for _, outcome, _ in comparisons)
    blocked = losses >= min_loss_matches and losses > wins
    return {
        "candidate_action": signal["action"],
        "similar_matches": len(comparisons),
        "loss_matches": losses,
        "win_matches": wins,
        "best_similarity": round(max((score for score, _, _ in comparisons), default=0), 3),
        "average_pnl": round(sum(pnl for _, _, pnl in comparisons) / len(comparisons), 2) if comparisons else None,
        "lookback_days": settings.pattern_lookback_days,
        "blocked": blocked,
    }


def _similarity(current: dict, historic: dict) -> float:
    """Compare scale-free entry behaviour; 1.0 means the same indicator setup."""
    def safe(value, default=0.0):
        return float(value) if value is not None else default

    def atr_features(values):
        atr = max(abs(safe(values.get("atr"))), 0.000001)
        vol = max(abs(safe(values.get("vol_20"))), 0.000001)
        return (
            safe(values.get("rsi")),
            (safe(values.get("ema20")) - safe(values.get("ema50"))) / atr,
            (safe(values.get("macd")) - safe(values.get("macd_signal"))) / atr,
            safe(values.get("momentum_10")) / vol,
            (safe(values.get("price_ema20_ratio"), 1.0) - 1.0) * safe(values.get("ema20")) / atr,
        )

    a, b = atr_features(current), atr_features(historic)
    scales = (20, 3, 1, 3, 2)
    distance = sum(min(abs(x - y) / scale, 1) for x, y, scale in zip(a, b, scales)) / len(scales)
    structural = [(current.get("_structure", {})[key], value)
                  for key, value in historic.get("_structure", {}).items()
                  if key in current.get("_structure", {})]
    if structural:
        distance = .7 * distance + .3 * sum(a != b for a, b in structural) / len(structural)
    return 1 - distance
