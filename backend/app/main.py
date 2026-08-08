import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .backtest import BacktestSettings, run_backtest
from .database import initialize
from .engine import engine
from .market import fetch_candles


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
    return {"ok": True, "mode": "PAPER_ONLY", "symbol": settings.symbol, "interval": settings.interval}


@app.get("/api/state")
def state():
    return engine.state()


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
        result = run_backtest(candles, BacktestSettings(
            starting_balance=settings.starting_balance,
            risk_percent=settings.risk_percent,
            stop_multiplier=settings.atr_stop_multiplier,
            target_multiplier=settings.atr_target_multiplier,
            higher_timeframe_filter=settings.higher_timeframe_filter,
        ), higher_timeframes)
        result.update({"symbol": settings.symbol, "interval": settings.interval, "feed": feed})
        return result
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
