import asyncio
import json
from datetime import datetime, timedelta, timezone
from .config import settings
from .database import connection, row_dict
from .market import fetch_candles, validate_entry_data, interval_minutes
from .active_strategy import active_name, signal_from
from .strategy import enrich, order_levels
from .learning import assess_signal, record_outcome
from .news import news_guard
from .prediction import predict, status as prediction_status, entry_context, maybe_retrain
from .execution import account_risk_gate, position_gate, filter_setup, exit_fill, entry_threshold
from .trend import higher_timeframe_context


class PaperEngine:
    def __init__(self):
        self.last_error = None
        self.feed = "waiting"
        self.running = False
        self.latest_signal = None
        self._lock = asyncio.Lock()

    async def scan(self):
        async with self._lock:
            try:
                candles, self.feed = await fetch_candles()
                observed_at = self.now()
                data = enrich(candles)
                if len(data) < 3:
                    raise RuntimeError("Not enough usable candles to evaluate a closed setup.")
                signal = signal_from(data)
                signal["model"] = active_name()
                signal["training_context"] = {"feed": self.feed}
                data_quality = validate_entry_data(candles)
                # A delayed provider response must not fill a new order at an old close.
                latest = candles.iloc[-1]
                step = interval_minutes() * 60
                age = (self._timestamp(observed_at) - self._timestamp(latest.datetime)).total_seconds()
                if not 0 <= age < step:
                    data_quality = {"allowed": False, "reason": "Entry blocked: no current candle price is available."}
                elif (self._timestamp(latest.datetime) - self._timestamp(signal["candle_time"])).total_seconds() != step:
                    data_quality = {"allowed": False, "reason": "Entry blocked: the setup is not from the immediately preceding candle."}
                signal["execution_price"] = float(latest.close)
                # The probability model uses trend alignment as an input, so the
                # trends are recorded whenever it is on, even with the filter off.
                wants_context = settings.higher_timeframe_filter or settings.prediction_model_enabled
                context = await higher_timeframe_context() if wants_context else {}
                news = await news_guard.evaluate()
                if context:
                    signal["indicators"]["weekly_trend"] = context.get("weekly", {}).get("trend", "UNAVAILABLE")
                    signal["indicators"]["monthly_trend"] = context.get("monthly", {}).get("trend", "UNAVAILABLE")
                with connection() as db:
                    account = dict(db.execute("SELECT * FROM account WHERE id=1").fetchone())
                    open_trades = [dict(r) for r in db.execute("SELECT * FROM trades WHERE status='OPEN' ORDER BY id ASC").fetchall()]
                    for trade in open_trades:
                        # Only full candles starting after the actual fill may
                        # contribute high/low extremes. The entry candle contains
                        # unknown pre-entry movement and is managed by samples.
                        rows = [row for _, row in candles.iloc[:-1].iterrows()
                                if self._timestamp(row.datetime) >= self._timestamp(trade["opened_at"])]
                        closed = False
                        for row in rows:










                            if self._manage_open(db, trade, float(row.high), float(row.low),
                                                 float(row.close), str(row.datetime)):
                                closed = True
                                break
                        # Entry-quality gates must not prevent a valid current
                        # sample from managing an existing position.
                        if not closed and 0 <= age < step:
                            self._manage_open(db, trade, float(latest.close), float(latest.close),
                                              float(latest.close), observed_at, sampled=True)
                    balance = db.execute("SELECT balance FROM account WHERE id=1").fetchone()[0]
                    remaining = db.execute("SELECT side,entry,units FROM trades WHERE status='OPEN'").fetchall()
                    floating = sum((float(latest.close) - r["entry"]) * r["units"] * (1 if r["side"] == "BUY" else -1) for r in remaining)
                    db.execute("UPDATE account SET equity=? WHERE id=1", (round(balance + floating, 2),))
                    filter_setup(signal, context, settings.higher_timeframe_filter)
                    signal["data_quality"] = data_quality
                    if not data_quality["allowed"]:
                        signal.update(action="HOLD", reason=data_quality["reason"])
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
                    if signal["action"] in ("BUY", "SELL"):
                        if signal.get("setup_id") and db.execute("SELECT 1 FROM trades WHERE setup_id=?", (signal["setup_id"],)).fetchone():
                            signal.update(action="HOLD", reason="Entry blocked: this liquidity pool has already been traded.")
                    if signal["action"] in ("BUY", "SELL"):
                        try:
                            execution_cost = (settings.spread_pips / 2 + settings.slippage_pips) * settings.pip_size
                            levels = order_levels(signal, balance, settings.risk_percent,
                                                  settings.atr_stop_multiplier, settings.atr_target_multiplier, execution_cost)
                        except ValueError as exc:
                            signal.update(action="HOLD", reason=f"Entry blocked: {exc}.")
                    if signal["action"] in ("BUY", "SELL"):
                        prediction = predict(signal)
                        signal["prediction"] = prediction
                        if not prediction["allowed"]:
                            signal["action"] = "HOLD"
                            signal["reason"] = ("Entry blocked: model target-before-stop probability "
                                                f"{prediction['target_before_stop_probability']:.1%} is below "
                                                f"{prediction['threshold']:.1%}.")
                    account = dict(db.execute("SELECT * FROM account WHERE id=1").fetchone())
                    risk_gate = self._account_risk_gate(db, account["balance"],
                                                      new_risk=levels["risk_amount"] if signal["action"] in ("BUY", "SELL") else 0)
                    signal["risk_gate"] = risk_gate
                    if signal["action"] in ("BUY", "SELL") and not risk_gate["allowed"]:
                        signal["action"] = "HOLD"
                        signal["reason"] = risk_gate["reason"]
                    active_trades = [dict(r) for r in db.execute("SELECT * FROM trades WHERE status='OPEN'").fetchall()]
                    if signal["action"] in ("BUY", "SELL"):
                        capacity = position_gate(signal["action"], active_trades)
                        if not capacity["allowed"]:
                            signal.update(action="HOLD", reason=capacity["reason"])
                    if account["last_candle"] is None or self._timestamp(signal["candle_time"]) > self._timestamp(account["last_candle"]):
                        db.execute("""INSERT INTO signals
                            (candle_time,action,confidence,price,reason,indicators,created_at,market_bias,bias_score)
                            VALUES(?,?,?,?,?,?,?,?,?)
                            ON CONFLICT(candle_time) DO UPDATE SET
                                action=excluded.action, confidence=excluded.confidence,
                                price=excluded.price, reason=excluded.reason,
                                indicators=excluded.indicators, created_at=excluded.created_at,
                                market_bias=excluded.market_bias, bias_score=excluded.bias_score""",
                                   (signal["candle_time"], signal["action"], signal["confidence"],
                                    signal["price"], signal["reason"], json.dumps(signal["indicators"]),
                                    self.now(), signal.get("market_bias", "NEUTRAL"), signal.get("bias_score", 0)))
                        if signal["action"] in ("BUY", "SELL"):
                            signal["training_context"] = entry_context(signal, self.feed)
                            account = dict(db.execute("SELECT * FROM account WHERE id=1").fetchone())
                            signal["execution_revision"] = 2
                            db.execute("""INSERT INTO trades(side,status,entry,stop_loss,take_profit,units,risk_amount,opened_at,signal_snapshot,entry_candle_time,setup_id)
                                VALUES(?,'OPEN',?,?,?,?,?,?,?,?,?)""", (signal["action"], levels["entry"], levels["stop_loss"], levels["take_profit"], levels["units"], levels["risk_amount"], observed_at, json.dumps(signal), signal["candle_time"], signal.get("setup_id")))
                        db.execute("UPDATE account SET last_candle=?,updated_at=? WHERE id=1", (signal["candle_time"], self.now()))
                self.latest_signal = signal
                self.last_error = None
                # The trade transaction is committed before a background reader
                # takes its training snapshot. Training never blocks scans.
                try:
                    maybe_retrain(candles)
                except Exception as exc:
                    self.last_error = f"Automatic training could not start: {exc}"
                return signal
            except Exception as exc:
                self.last_error = str(exc)
                raise

    def _manage_open(self, db, trade: dict, high: float, low: float, close: float, candle_time: str, sampled: bool = False):
        stamp = self._timestamp(candle_time)
        if trade.get("opened_at"):
            opened = self._timestamp(trade["opened_at"])
            if stamp < opened or (sampled and stamp == opened):
                return False
        side = trade["side"]
        execution_cost = (settings.spread_pips / 2 + settings.slippage_pips) * settings.pip_size
        fill = exit_fill(trade, high, low, execution_cost)
        columns = {row[1] for row in db.execute("PRAGMA table_info(trades)").fetchall()}
        has_pricing_cols = {"side", "entry", "units"}.issubset(columns)

        # Conservative rule when both occur inside one candle: stop is assumed first.
        if fill is None:
            account = db.execute("SELECT balance FROM account WHERE id=1").fetchone()
            total_floating = 0.0
            if has_pricing_cols:
                remaining_open = db.execute("SELECT side, entry, units FROM trades WHERE status='OPEN'").fetchall()
                total_floating = sum(
                    (close - r["entry"]) * r["units"] * (1 if r["side"] == "BUY" else -1)
                    for r in remaining_open
                )
            db.execute("UPDATE account SET equity=?,updated_at=? WHERE id=1", (round(account["balance"] + total_floating, 2), self.now()))
            return False
        direction = 1 if side == "BUY" else -1
        exit_price, reason = fill
        # OHLC does not reveal the exact touch time. Record when the completed
        # bar becomes knowable; samples already carry their observation time.
        candle_time = (stamp if sampled else stamp + timedelta(minutes=interval_minutes())).isoformat()
        pnl = (exit_price - trade["entry"]) * trade["units"] * direction
        account = db.execute("SELECT balance FROM account WHERE id=1").fetchone()
        balance = account["balance"] + pnl
        db.execute("UPDATE trades SET status='CLOSED',closed_at=?,exit_price=?,pnl=?,reason=? WHERE id=?",
                   (candle_time, exit_price, round(pnl, 2), reason, trade["id"]))
        record_outcome(db, trade, pnl, reason, candle_time)
        total_floating = 0.0
        if has_pricing_cols:
            remaining_open = db.execute("SELECT side, entry, units FROM trades WHERE status='OPEN'").fetchall()
            total_floating = sum(
                (close - r["entry"]) * r["units"] * (1 if r["side"] == "BUY" else -1)
                for r in remaining_open
            )
        db.execute("UPDATE account SET balance=?,equity=?,updated_at=? WHERE id=1", (round(balance, 2), round(balance + total_floating, 2), self.now()))
        return True

    def state(self):
        with connection() as db:
            account = row_dict(db.execute("SELECT * FROM account WHERE id=1").fetchone())
            open_trades = [row_dict(r) for r in db.execute("SELECT * FROM trades WHERE status='OPEN' ORDER BY id ASC").fetchall()]
            trade = open_trades[-1] if open_trades else None
            signal = self.latest_signal or row_dict(db.execute("SELECT * FROM signals ORDER BY id DESC LIMIT 1").fetchone())
            trades = [row_dict(row) for row in db.execute("SELECT * FROM trades ORDER BY id DESC LIMIT 50").fetchall()]
            pattern_stats = dict(db.execute("""SELECT COUNT(*) AS recorded,
                COALESCE(SUM(outcome='LOSS'), 0) AS losses, COALESCE(SUM(outcome='WIN'), 0) AS wins
                FROM trade_patterns""").fetchone())
            totals = dict(db.execute("SELECT COUNT(*) AS closed, COALESCE(SUM(pnl>0),0) AS wins FROM trades WHERE status='CLOSED'").fetchone())
            current_risk = self._account_risk_gate(db, account["balance"])
        wins, count = totals["wins"], totals["closed"]
        return {"mode": "PAPER_ONLY", "symbol": settings.symbol, "interval": settings.interval,
                "strategy": active_name(),
                "feed": self.feed, "account": account, "open_trade": trade, "open_trades": open_trades,
                "max_concurrent_trades": settings.max_concurrent_trades,
                "configuration": {"starting_balance": settings.starting_balance, "risk_percent": settings.risk_percent,
                                  "signal_max_age_seconds": (settings.market_max_age_bars + 1) * interval_minutes() * 60,
                                  "entry_threshold": entry_threshold(), "higher_timeframe_filter": settings.higher_timeframe_filter,
                                  "max_consecutive_losses": settings.max_consecutive_losses,
                                  "loss_cooldown_hours": settings.loss_cooldown_hours,
                                  "target_description": (f"Structural stop; minimum {settings.smc_min_reward if settings.strategy == 'smc' else settings.judas_min_reward}R target"
                                                         if settings.strategy in ("smc", "judas") else
                                                         f"{settings.atr_stop_multiplier} ATR stop; {settings.atr_target_multiplier} ATR target")},
                "risk_gate": current_risk,
                "latest_signal": signal,
                "trades": trades, "stats": {"closed": count, "wins": wins,
                "win_rate": round(wins / count * 100, 1) if count else 0},
                "pattern_memory": pattern_stats, "prediction_model": prediction_status(),
                "news": news_guard.status(), "last_error": self.last_error}

    @staticmethod
    def _account_risk_gate(db, balance: float, now: datetime | None = None, new_risk: float = 0.0) -> dict:
        rows = db.execute("SELECT id, closed_at, pnl FROM trades WHERE status='CLOSED'").fetchall()
        columns = {row[1] for row in db.execute("PRAGMA table_info(trades)")}
        opened = ([dict(zip(("entry", "stop_loss", "units", "risk_amount"), r)) for r in
                   db.execute("SELECT entry,stop_loss,units,risk_amount FROM trades WHERE status='OPEN'")]
                  if {"entry", "stop_loss", "units", "risk_amount"}.issubset(columns) else [])
        return account_risk_gate([dict(zip(("id", "closed_at", "pnl"), row)) for row in rows], balance, now, opened, new_risk)

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
