# EUR/USD 15-Minute Paper Trader

A safety-locked demo trading system with:

- Next.js dashboard
- Python FastAPI analysis and paper-execution engine
- Twelve Data 15-minute EUR/USD candles
- EMA 20/50, RSI 14, MACD, ATR 14, volatility and momentum confirmations
- Automatic simulated entries, ATR stop-loss/take-profit, 1% risk sizing
- SQLite trade journal and equity tracking

It contains **no live broker integration** and cannot place real-money orders.

## 1. Start the Python service

```powershell
cd backend
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your Twelve Data key to `backend/.env`, then:

```powershell
uvicorn app.main:app --reload --port 8000
```

## 2. Start the dashboard

Open another PowerShell window:

```powershell
cd dashboard
Copy-Item .env.example .env.local
pnpm install
pnpm dev
```

Open http://localhost:3000.

## Demo feed

Set `USE_SYNTHETIC_DATA=true` in `backend/.env` to test without an API key. Synthetic candles are clearly labelled and must not be used for market decisions.

## High-impact news guard

The paper engine can block new entries around high-impact USD and EUR economic
releases. Create a free Financial Modeling Prep Basic key, then
set these values in `backend/.env`:

```text
NEWS_FILTER_ENABLED=true
FMP_API_KEY=your_key_here
```

The free plan currently permits 250 calls/day; the engine caches the calendar
for 60 minutes and normally uses about 24 calls/day. The default blackout is
30 minutes before through 15 minutes after an event.
News is a risk gate only; it does not predict BUY or SELL direction. The
historical backtest does not yet include economic-calendar events.

## Strategy

- BUY confirmation: EMA20 > EMA50, MACD > signal, RSI 52–70, positive 10-bar momentum.
- SELL confirmation: EMA20 < EMA50, MACD < signal, RSI 30–48, negative 10-bar momentum.
- Stop: 1.0 ATR; target: 1.5 ATR; account risk: 1%.
- Only one EUR/USD paper position can be open.
- The engine evaluates once per newly closed 15-minute candle.
- Weekly and monthly EMA trends must agree with a short-term BUY or SELL before
  the entry is permitted (configurable in `backend/.env`).
- Every closed paper trade records its entry indicators and outcome. A new entry is
  blocked only after it closely matches at least three earlier losing paper trades;
  this is a risk guard, not a price prediction.

No strategy guarantees a profit. Validate through extended paper trading before changing any rule.
