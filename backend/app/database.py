import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "paper_trader.db"


@contextmanager
def connection():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    try:
        yield db
        db.commit()
    finally:
        db.close()



def initialize(starting_balance: float):
    with connection() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS account (
            id INTEGER PRIMARY KEY CHECK(id=1), balance REAL NOT NULL,
            equity REAL NOT NULL, last_candle TEXT, updated_at TEXT NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT, side TEXT NOT NULL,
            status TEXT NOT NULL, entry REAL NOT NULL, stop_loss REAL NOT NULL,
            take_profit REAL NOT NULL, units REAL NOT NULL, risk_amount REAL NOT NULL,
            opened_at TEXT NOT NULL, closed_at TEXT, exit_price REAL,
            pnl REAL DEFAULT 0, reason TEXT, signal_snapshot TEXT NOT NULL,
            entry_candle_time TEXT)""")
        columns = {row["name"] for row in db.execute("PRAGMA table_info(trades)")}
        if "entry_candle_time" not in columns:
            db.execute("ALTER TABLE trades ADD COLUMN entry_candle_time TEXT")
        db.execute("""CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT, candle_time TEXT UNIQUE,
            action TEXT NOT NULL, confidence REAL NOT NULL, price REAL NOT NULL,
            reason TEXT NOT NULL, indicators TEXT NOT NULL, created_at TEXT NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS trade_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT, trade_id INTEGER NOT NULL UNIQUE,
            side TEXT NOT NULL, outcome TEXT NOT NULL, pnl REAL NOT NULL,
            exit_reason TEXT NOT NULL, features TEXT NOT NULL, closed_at TEXT NOT NULL)""")
        db.execute("INSERT OR IGNORE INTO account VALUES (1, ?, ?, NULL, datetime('now'))", (starting_balance, starting_balance))
        # Preserve learning from trades closed before this table was introduced.
        for trade in db.execute("SELECT * FROM trades WHERE status='CLOSED'").fetchall():
            item = dict(trade)
            db.execute("""INSERT OR IGNORE INTO trade_patterns
                (trade_id,side,outcome,pnl,exit_reason,features,closed_at) VALUES(?,?,?,?,?,?,?)""",
                (item["id"], item["side"], "WIN" if item["pnl"] > 0 else "LOSS", item["pnl"],
                 item["reason"] or "UNKNOWN", json.dumps(json.loads(item["signal_snapshot"])["indicators"]),
                 item["closed_at"] or item["opened_at"]))


def row_dict(row):
    if row is None:
        return None
    item = dict(row)
    for key in ("signal_snapshot", "indicators"):
        if key in item and isinstance(item[key], str):
            item[key] = json.loads(item[key])
    return item
