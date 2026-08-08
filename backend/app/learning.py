"""Small, explainable memory of paper-trade outcomes.

This is intentionally a pattern guard, not a claim to predict markets.  It
only prevents a new paper entry after several *very similar* previous entries
ended in losses.
"""
import json


def record_outcome(db, trade: dict, pnl: float, reason: str, closed_at: str) -> None:
    snapshot = json.loads(trade["signal_snapshot"])
    db.execute(
        """INSERT OR IGNORE INTO trade_patterns
           (trade_id, side, outcome, pnl, exit_reason, features, closed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (trade["id"], trade["side"], "WIN" if pnl > 0 else "LOSS", pnl, reason,
         json.dumps(snapshot["indicators"]), closed_at),
    )


def assess_signal(db, signal: dict, min_loss_matches: int, similarity_threshold: float) -> dict:
    """Return the evidence from comparable historical paper trades."""
    rows = db.execute(
        "SELECT outcome, features FROM trade_patterns WHERE side=? ORDER BY id DESC",
        (signal["action"],),
    ).fetchall()
    comparisons = []
    for row in rows:
        score = _similarity(signal["indicators"], json.loads(row["features"]))
        if score >= similarity_threshold:
            comparisons.append((score, row["outcome"]))

    losses = sum(outcome == "LOSS" for _, outcome in comparisons)
    wins = sum(outcome == "WIN" for _, outcome in comparisons)
    blocked = losses >= min_loss_matches and losses > wins
    return {
        "candidate_action": signal["action"],
        "similar_matches": len(comparisons),
        "loss_matches": losses,
        "win_matches": wins,
        "best_similarity": round(max((score for score, _ in comparisons), default=0), 3),
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
    return 1 - distance
