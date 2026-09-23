import asyncio
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .backtest import BacktestSettings, run_backtest
from .database import initialize
from .engine import engine
from .market import fetch_candles
from .prediction import job as training_job, status as prediction_status


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize(settings.starting_balance)
    task = asyncio.create_task(engine.loop())
    yield
    engine.running = False
    task.cancel()


app = FastAPI(title="EUR/USD Paper Trader", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.allowed_origins.split(",") if origin.strip()],
    # Next.js may be opened through either localhost or 127.0.0.1, and its
    # development port is configurable. These are loopback-only origins.
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "mode": "PAPER_ONLY", "symbol": settings.symbol,
            "interval": settings.interval, "strategy": settings.strategy}


@app.get("/api/state")
def state():
    return engine.state()


@app.get("/api/ai-review")
def ai_review_status():
    from .ai_review import reviewer
    return reviewer.status()


@app.post("/api/ai-review")
async def ai_review(request: Request):
    from .ai_review import reviewer, ReviewError
    origin = request.headers.get("origin")
    allowed = {value.strip() for value in settings.allowed_origins.split(",") if value.strip()}
    if origin and origin not in allowed and not re.fullmatch(r"https?://(localhost|127\.0\.0\.1)(:\d+)?", origin):
        raise HTTPException(status_code=403, detail="Reviews must be requested from the configured dashboard origin.")
    try:
        return await reviewer.review(engine.state())
    except ReviewError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from None


@app.get("/api/pairs")
async def pairs():
    from .pairs import overview
    return await overview(engine.latest_signal)


@app.get("/api/candles")
async def candles(limit: int = 160):
    """Return recent OHLC candles for the dashboard chart."""
    if limit < 20 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 20 and 500")
    try:
        frame, feed = await fetch_candles(outputsize=limit)
        values = frame[["datetime", "open", "high", "low", "close"]].copy()
        values["datetime"] = values["datetime"].apply(lambda value: value.isoformat())
        return {"symbol": settings.symbol, "interval": settings.interval,
                "feed": feed, "candles": values.to_dict(orient="records")}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/scan")
async def scan():
    try:
        signal = await engine.scan()
        return {"ok": True, "signal": signal}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/backtest")
async def backtest():
    try:
        candles, feed = await fetch_candles(outputsize=5000)
        higher_timeframes = {}
        if settings.higher_timeframe_filter:
            weekly, monthly = await asyncio.gather(
                fetch_candles(outputsize=240, interval=settings.weekly_interval),
                fetch_candles(outputsize=240, interval=settings.monthly_interval),
            )
            higher_timeframes = {"weekly": weekly[0], "monthly": monthly[0]}
        result = await asyncio.to_thread(run_backtest, candles, BacktestSettings(
            starting_balance=settings.starting_balance,
            risk_percent=settings.risk_percent,
            stop_multiplier=settings.atr_stop_multiplier,
            target_multiplier=settings.atr_target_multiplier,
            higher_timeframe_filter=settings.higher_timeframe_filter,
            spread_pips=settings.spread_pips,
            slippage_pips=settings.slippage_pips,
            pip_size=settings.pip_size,
        ), higher_timeframes)
        result.update({"symbol": settings.symbol, "interval": settings.interval, "feed": feed})
        return result
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/model")
def model_status():
    return prediction_status()


@app.post("/api/model/train", status_code=202)
async def train_model():
    """Start local training in the background; poll GET /api/model for progress.

    No candle data leaves this service. A first run replays the cached
    multi-year history and takes minutes, so this returns immediately.
    """
    if training_job.running:
        raise HTTPException(status_code=409, detail="Model training is already running.")
    try:
        recent, _ = await fetch_candles(outputsize=5000)
    except Exception:
        recent = None  # the cached history alone is enough to train on
    if not training_job.start(recent):
        raise HTTPException(status_code=409, detail="Model training is already running.")
    return {"ok": True, "job": training_job.snapshot()}
