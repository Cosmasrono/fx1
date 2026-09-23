import json
import sqlite3
from .learning import outcome_features
from contextlib import contextmanager
from pathlib import Path

# Execution revision 2 starts a separate journal; preserve the original evidence.
DB_PATH = Path(__file__).resolve().parents[1] / "paper_trader_v2.db"


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
        if "setup_id" not in columns:
            db.execute("ALTER TABLE trades ADD COLUMN setup_id TEXT")
        db.execute("CREATE UNIQUE INDEX IF NOT EXISTS trades_setup_id ON trades(setup_id) WHERE setup_id IS NOT NULL")
        db.execute("""CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT, candle_time TEXT UNIQUE,
            action TEXT NOT NULL, confidence REAL NOT NULL, price REAL NOT NULL,
            reason TEXT NOT NULL, indicators TEXT NOT NULL, created_at TEXT NOT NULL,
            market_bias TEXT NOT NULL DEFAULT 'NEUTRAL', bias_score INTEGER NOT NULL DEFAULT 0)""")
        signal_columns = {row["name"] for row in db.execute("PRAGMA table_info(signals)")}
        if "market_bias" not in signal_columns:
            db.execute("ALTER TABLE signals ADD COLUMN market_bias TEXT NOT NULL DEFAULT 'NEUTRAL'")
        if "bias_score" not in signal_columns:
            db.execute("ALTER TABLE signals ADD COLUMN bias_score INTEGER NOT NULL DEFAULT 0")
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
                 item["reason"] or "UNKNOWN", json.dumps(outcome_features(json.loads(item["signal_snapshot"]))),
                 item["closed_at"] or item["opened_at"]))
            db.execute("UPDATE trade_patterns SET features=? WHERE trade_id=?",
                       (json.dumps(outcome_features(json.loads(item["signal_snapshot"]))), item["id"]))


def row_dict(row):
    if row is None:
        return None
    item = dict(row)
    for key in ("signal_snapshot", "indicators"):
        if key in item and isinstance(item[key], str):
            item[key] = json.loads(item[key])
    return item
