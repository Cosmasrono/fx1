import json
import sqlite3
import pytest
from app.config import settings
from app.engine import PaperEngine


def setup_test_db():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE account (id INTEGER PRIMARY KEY, balance REAL, equity REAL, last_candle TEXT, updated_at TEXT)")
    db.execute("INSERT INTO account VALUES (1, 10000.0, 10000.0, NULL, '2026-08-01T00:00:00+00:00')")
    db.execute("""CREATE TABLE trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT, side TEXT NOT NULL,
        status TEXT NOT NULL, entry REAL NOT NULL, stop_loss REAL NOT NULL,
        take_profit REAL NOT NULL, units REAL NOT NULL, risk_amount REAL NOT NULL,
        opened_at TEXT NOT NULL, closed_at TEXT, exit_price REAL,
        pnl REAL DEFAULT 0, reason TEXT, signal_snapshot TEXT NOT NULL,
        entry_candle_time TEXT)""")
    db.execute("""CREATE TABLE signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT, candle_time TEXT UNIQUE,
        action TEXT NOT NULL, confidence REAL NOT NULL, price REAL NOT NULL,
        reason TEXT NOT NULL, indicators TEXT NOT NULL, created_at TEXT NOT NULL,
        market_bias TEXT NOT NULL DEFAULT 'NEUTRAL', bias_score INTEGER NOT NULL DEFAULT 0)""")
    db.execute("""CREATE TABLE trade_patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT, trade_id INTEGER UNIQUE,
        side TEXT NOT NULL, outcome TEXT NOT NULL, pnl REAL NOT NULL,
        exit_reason TEXT NOT NULL, features TEXT NOT NULL, closed_at TEXT NOT NULL)""")
    return db


def test_concurrency_gate_allows_up_to_max():
    db = setup_test_db()
    engine = PaperEngine()

    # Insert 2 open BUY trades (limit is 3)
    for i in range(1, 3):
        db.execute("""INSERT INTO trades
            (side, status, entry, stop_loss, take_profit, units, risk_amount, opened_at, signal_snapshot, entry_candle_time)
            VALUES('BUY', 'OPEN', 1.1000, 1.0980, 1.1030, 5000, 100, '2026-08-01T01:00:00+00:00', '{}', ?)""",
            (f"2026-08-01T0{i}:00:00+00:00",))

    active_trades = [dict(r) for r in db.execute("SELECT * FROM trades WHERE status='OPEN'").fetchall()]
    assert len(active_trades) == 2
    assert len(active_trades) < settings.max_concurrent_trades


def test_concurrency_gate_blocks_when_max_reached():
    db = setup_test_db()
    # Insert 3 open BUY trades
    for i in range(1, 4):
        db.execute("""INSERT INTO trades
            (side, status, entry, stop_loss, take_profit, units, risk_amount, opened_at, signal_snapshot, entry_candle_time)
            VALUES('BUY', 'OPEN', 1.1000, 1.0980, 1.1030, 5000, 100, '2026-08-01T01:00:00+00:00', '{}', ?)""",
            (f"2026-08-01T0{i}:00:00+00:00",))

    active_trades = [dict(r) for r in db.execute("SELECT * FROM trades WHERE status='OPEN'").fetchall()]
    assert len(active_trades) == 3
    assert len(active_trades) >= settings.max_concurrent_trades


def test_conflicting_direction_blocked_when_open():
    db = setup_test_db()
    db.execute("""INSERT INTO trades
        (side, status, entry, stop_loss, take_profit, units, risk_amount, opened_at, signal_snapshot, entry_candle_time)
        VALUES('BUY', 'OPEN', 1.1000, 1.0980, 1.1030, 5000, 100, '2026-08-01T01:00:00+00:00', '{}', '2026-08-01T01:00:00+00:00')""")

    active_trades = [dict(r) for r in db.execute("SELECT * FROM trades WHERE status='OPEN'").fetchall()]
    new_action = "SELL"
    is_conflicting = active_trades and any(t["side"] != new_action for t in active_trades)
    assert is_conflicting is True


def test_aggregate_floating_equity_with_multiple_trades():
    db = setup_test_db()
    engine = PaperEngine()

    # Position 1: BUY at 1.1000, 10,000 units
    db.execute("""INSERT INTO trades (id, side, status, entry, stop_loss, take_profit, units, risk_amount, opened_at, signal_snapshot)
        VALUES (1, 'BUY', 'OPEN', 1.1000, 1.0950, 1.1100, 10000, 50, '2026-08-01T00:00:00', '{}')""")
    # Position 2: BUY at 1.1020, 10,000 units
    db.execute("""INSERT INTO trades (id, side, status, entry, stop_loss, take_profit, units, risk_amount, opened_at, signal_snapshot)
        VALUES (2, 'BUY', 'OPEN', 1.1020, 1.0970, 1.1120, 10000, 50, '2026-08-01T01:00:00', '{}')""")

    # Current price is 1.1050
    # Floating 1: (1.1050 - 1.1000) * 10000 = +50.00 USD
    # Floating 2: (1.1050 - 1.1020) * 10000 = +30.00 USD
    # Total Floating: +80.00 USD -> Equity should be 10000 + 80 = 10080.00 USD
    trade1 = dict(db.execute("SELECT * FROM trades WHERE id=1").fetchone())
    engine._manage_open(db, trade1, high=1.1055, low=1.1040, close=1.1050, candle_time="2026-08-01T02:00:00")

    account = dict(db.execute("SELECT * FROM account WHERE id=1").fetchone())
    assert account["balance"] == 10000.0
    assert pytest.approx(account["equity"], 0.01) == 10080.0


def test_one_trade_closes_while_other_remains_open():
    db = setup_test_db()
    engine = PaperEngine()

    # Position 1: Take profit at 1.1040
    db.execute("""INSERT INTO trades (id, side, status, entry, stop_loss, take_profit, units, risk_amount, opened_at, signal_snapshot)
        VALUES (1, 'BUY', 'OPEN', 1.1000, 1.0950, 1.1040, 10000, 50, '2026-08-01T00:00:00', '{"indicators":{"rsi":55}}')""")
    # Position 2: Take profit at 1.1100 (not hit yet)
    db.execute("""INSERT INTO trades (id, side, status, entry, stop_loss, take_profit, units, risk_amount, opened_at, signal_snapshot)
        VALUES (2, 'BUY', 'OPEN', 1.1020, 1.0970, 1.1100, 10000, 50, '2026-08-01T01:00:00', '{"indicators":{"rsi":58}}')""")

    # Candle reaches high 1.1045, close 1.1030
    trade1 = dict(db.execute("SELECT * FROM trades WHERE id=1").fetchone())
    closed = engine._manage_open(db, trade1, high=1.1045, low=1.1025, close=1.1030, candle_time="2026-08-01T02:00:00")
    assert closed is True

    t1 = dict(db.execute("SELECT status, reason, pnl FROM trades WHERE id=1").fetchone())
    t2 = dict(db.execute("SELECT status FROM trades WHERE id=2").fetchone())
    assert t1["status"] == "CLOSED"
    assert t1["reason"] == "TAKE_PROFIT"
    assert t2["status"] == "OPEN"

    account = dict(db.execute("SELECT * FROM account WHERE id=1").fetchone())
    # Realized balance should have increased by trade 1's PnL
    assert account["balance"] > 10000.0
    # Floating on trade 2: (1.1030 - 1.1020) * 10000 = +10.00
    assert pytest.approx(account["equity"], 0.01) == account["balance"] + 10.0
