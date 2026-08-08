import json
import sqlite3

import pytest

from app.engine import PaperEngine


@pytest.mark.parametrize(
    ("side", "entry", "stop", "target", "high", "low"),
    [
        ("BUY", 1.1000, 1.0990, 1.1015, 1.1016, 1.1000),
        ("SELL", 1.1000, 1.1010, 1.0985, 1.1000, 1.0984),
    ],
)
def test_take_profit_closes_buy_and_sell(side, entry, stop, target, high, low):
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE account (id INTEGER PRIMARY KEY, balance REAL, equity REAL, updated_at TEXT)")
    db.execute("INSERT INTO account VALUES (1, 10000, 10000, '')")
    db.execute(
        "CREATE TABLE trades (id INTEGER PRIMARY KEY, status TEXT, closed_at TEXT, "
        "exit_price REAL, pnl REAL, reason TEXT)"
    )
    db.execute("INSERT INTO trades VALUES (1, 'OPEN', NULL, NULL, 0, NULL)")
    db.execute(
        "CREATE TABLE trade_patterns (trade_id INTEGER UNIQUE, side TEXT, outcome TEXT, "
        "pnl REAL, exit_reason TEXT, features TEXT, closed_at TEXT)"
    )
    trade = {
        "id": 1,
        "side": side,
        "entry": entry,
        "stop_loss": stop,
        "take_profit": target,
        "units": 10000,
        "signal_snapshot": json.dumps({"indicators": {"rsi": 60}}),
    }

    PaperEngine()._manage_open(db, trade, high, low, target, "2026-08-07T10:00:00+00:00")

    closed = db.execute("SELECT status, pnl, reason FROM trades WHERE id=1").fetchone()
    account = db.execute("SELECT balance, equity FROM account WHERE id=1").fetchone()
    assert closed["status"] == "CLOSED"
    assert closed["reason"] == "TAKE_PROFIT"
    assert closed["pnl"] > 0
    assert account["balance"] == account["equity"]
    assert account["balance"] > 10000


def test_timestamp_comparison_accepts_database_and_pandas_formats():
    database_value = "2026-08-07T09:05:00+00:00"
    candle_value = "2026-08-07 09:10:00+00:00"

    assert PaperEngine._timestamp(candle_value) > PaperEngine._timestamp(database_value)
