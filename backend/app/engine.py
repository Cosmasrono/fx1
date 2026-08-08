import asyncio
import json
from datetime import datetime, timezone
from .config import settings
from .database import connection, row_dict
from .market import fetch_candles
from .strategy import enrich, order_levels, signal_from
from .learning import assess_signal, record_outcome
from .news import news_guard
from .trend import entry_allowed, higher_timeframe_context


class PaperEngine:
    def __init__(self):
        self.last_error = None
        self.feed = "waiting"
        self.running = False
        self._lock = asyncio.Lock()

    async def scan(self):
        async with self._lock:
            try:
                candles, self.feed = await fetch_candles()
                data = enrich(candles)
                signal = signal_from(data)
                context = await higher_timeframe_context() if settings.higher_timeframe_filter else {}
                news = await news_guard.evaluate()
                with connection() as db:
                    account = dict(db.execute("SELECT * FROM account WHERE id=1").fetchone())
                    open_trade = db.execute("SELECT * FROM trades WHERE status='OPEN' ORDER BY id DESC LIMIT 1").fetchone()
                    # The final API row may still be forming. Executing against
                    # it turns a temporary wick into a permanent paper loss.
                    completed = data.iloc[-2] if len(data) > 2 else data.iloc[-1]
                    if open_trade:
                        trade = dict(open_trade)
                        entry_candle = trade.get("entry_candle_time")
                        # Replay every completed candle since entry. Comparing
                        # parsed timestamps avoids the previous " " versus "T"
                        # ISO-format bug and catches targets hit before a retrace.
                        rows = [completed] if not entry_candle else [
                            row for _, row in data.iloc[:-1].iterrows()
                            if self._timestamp(row.datetime) > self._timestamp(entry_candle)
                        ]
                        for row in rows:
                            if self._manage_open(db, trade, float(row.high), float(row.low),
                                                 float(row.close), str(row.datetime)):
                                break
                    if signal["action"] in ("BUY", "SELL") and settings.higher_timeframe_filter:
                        signal["indicators"]["weekly_trend"] = context["weekly"]["trend"]
                        signal["indicators"]["monthly_trend"] = context["monthly"]["trend"]
                        if not entry_allowed(signal["action"], context):
                            signal["reason"] = (f"Entry blocked: {signal['action']} conflicts with the "
                                                "weekly/monthly trend filter.")
                            signal["action"] = "HOLD"
                    if signal["action"] in ("BUY", "SELL"):
                        learning = assess_signal(db, signal, settings.pattern_min_loss_matches,
                                                 settings.pattern_similarity_threshold)
                        signal["learning"] = learning
                        if learning["blocked"]:
                            signal["action"] = "HOLD"
                            signal["reason"] = ("Entry blocked: this setup matches "
                                                f"{learning['loss_matches']} previous losing paper trades.")
                    if signal["action"] in ("BUY", "SELL") and news["blocking"]:
                        event = news.get("event")
                        signal["action"] = "HOLD"
                        signal["reason"] = (f"Entry blocked around high-impact news: {event['name']}."
                                            if event else "Entry blocked because the news calendar is unavailable.")
                    if account["last_candle"] != signal["candle_time"]:
                        db.execute("INSERT OR IGNORE INTO signals(candle_time,action,confidence,price,reason,indicators,created_at) VALUES(?,?,?,?,?,?,?)",
                                   (signal["candle_time"], signal["action"], signal["confidence"], signal["price"], signal["reason"], json.dumps(signal["indicators"]), self.now()))
                        still_open = db.execute("SELECT id FROM trades WHERE status='OPEN'").fetchone()
                        if not still_open and signal["action"] in ("BUY", "SELL"):
                            account = dict(db.execute("SELECT * FROM account WHERE id=1").fetchone())
                            execution_cost = (settings.spread_pips / 2 + settings.slippage_pips) * settings.pip_size
                            levels = order_levels(signal, account["balance"], settings.risk_percent,
                                                  settings.atr_stop_multiplier, settings.atr_target_multiplier,
                                                  execution_cost)
                            db.execute("""INSERT INTO trades(side,status,entry,stop_loss,take_profit,units,risk_amount,opened_at,signal_snapshot,entry_candle_time)
                                VALUES(?,'OPEN',?,?,?,?,?,?,?,?)""", (signal["action"], levels["entry"], levels["stop_loss"], levels["take_profit"], levels["units"], levels["risk_amount"], self.now(), json.dumps(signal), signal["candle_time"]))
                        db.execute("UPDATE account SET last_candle=?,updated_at=? WHERE id=1", (signal["candle_time"], self.now()))
                self.last_error = None
                return signal
            except Exception as exc:
                self.last_error = str(exc)
                raise

    def _manage_open(self, db, trade: dict, high: float, low: float, close: float, candle_time: str):
        side = trade["side"]
        stop_hit = low <= trade["stop_loss"] if side == "BUY" else high >= trade["stop_loss"]
        target_hit = high >= trade["take_profit"] if side == "BUY" else low <= trade["take_profit"]
        # Conservative rule when both occur inside one candle: stop is assumed first.
        if not stop_hit and not target_hit:
            direction = 1 if side == "BUY" else -1
            floating = (close - trade["entry"]) * trade["units"] * direction
            account = db.execute("SELECT balance FROM account WHERE id=1").fetchone()
            db.execute("UPDATE account SET equity=?,updated_at=? WHERE id=1", (account["balance"] + floating, self.now()))
            return False
        execution_cost = (settings.spread_pips / 2 + settings.slippage_pips) * settings.pip_size
        direction = 1 if side == "BUY" else -1
        exit_price = trade["stop_loss"] if stop_hit else trade["take_profit"] - direction * execution_cost
        reason = "STOP_LOSS" if stop_hit else "TAKE_PROFIT"
        pnl = (exit_price - trade["entry"]) * trade["units"] * direction
        account = db.execute("SELECT balance FROM account WHERE id=1").fetchone()
        balance = account["balance"] + pnl
        db.execute("UPDATE trades SET status='CLOSED',closed_at=?,exit_price=?,pnl=?,reason=? WHERE id=?",
                   (candle_time, exit_price, round(pnl, 2), reason, trade["id"]))
        record_outcome(db, trade, pnl, reason, candle_time)
        db.execute("UPDATE account SET balance=?,equity=?,updated_at=? WHERE id=1", (balance, balance, self.now()))
        return True

    def state(self):
        with connection() as db:
            account = row_dict(db.execute("SELECT * FROM account WHERE id=1").fetchone())
            trade = row_dict(db.execute("SELECT * FROM trades WHERE status='OPEN' ORDER BY id DESC LIMIT 1").fetchone())
            signal = row_dict(db.execute("SELECT * FROM signals ORDER BY id DESC LIMIT 1").fetchone())
            trades = [row_dict(row) for row in db.execute("SELECT * FROM trades ORDER BY id DESC LIMIT 50").fetchall()]
            pattern_stats = dict(db.execute("""SELECT COUNT(*) AS recorded,
                COALESCE(SUM(outcome='LOSS'), 0) AS losses, COALESCE(SUM(outcome='WIN'), 0) AS wins
                FROM trade_patterns""").fetchone())
        closed = [t for t in trades if t["status"] == "CLOSED"]
        wins = sum(1 for t in closed if t["pnl"] > 0)
        return {"mode": "PAPER_ONLY", "symbol": settings.symbol, "interval": settings.interval,
                "feed": self.feed, "account": account, "open_trade": trade, "latest_signal": signal,
                "trades": trades, "stats": {"closed": len(closed), "wins": wins,
                "win_rate": round(wins / len(closed) * 100, 1) if closed else 0},
                "pattern_memory": pattern_stats, "news": news_guard.status(), "last_error": self.last_error}

    async def loop(self):
        self.running = True
        while self.running:
            try:
                await self.scan()
            except Exception:
                pass
            await asyncio.sleep(settings.scan_seconds)

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _timestamp(value) -> datetime:
        if hasattr(value, "to_pydatetime"):
            parsed = value.to_pydatetime()
        else:
            parsed = datetime.fromisoformat(str(value).replace(" ", "T").replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)


engine = PaperEngine()
