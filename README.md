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

From the repository root, run the backend test suite with:

```powershell
pytest
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

### OpenRouter explanations

The main dashboard's **Explain my trading** panel offers **Generate review**.
It sends a bounded summary of the current signal, indicators, risk gates,
historical model validation and at most ten recent paper trades to OpenRouter
and its chosen model provider. Credentials, full account objects and full trade
snapshots are excluded. Review text is displayed as text and cannot execute code,
place trades, change settings or override the local model and risk gates.

Configure `OPENROUTER_API_KEY` in `backend/.env`; the existing `api_key` name is
also supported for compatibility. Keep secrets out of frontend environment
files. Restart the backend after changing the key. `OPENROUTER_MODEL` defaults
to `openrouter/free`, which chooses an available free model; quota and capacity
limits still apply. A different configured model may incur charges.

Reviews run only on an explicit button press, with a one-minute per-process
cooldown and a 60-second provider timeout. GET `/api/ai-review` reports status
and the latest in-memory review without calling OpenRouter. POST creates a
review. Reviews are timestamped snapshots, may be inaccurate, and are cleared
on backend restart. Provider failures do not stop the trading engine.

### Pair comparison dashboard

Open http://localhost:3000/pairs or choose **Compare pairs** in the trading
dashboard. It monitors EUR/USD, GBP/USD and USD/JPY, showing latest candle
prices, movement since the first available candle of the latest UTC day,
recent price charts, EMA direction, RSI and ATR in each pair's pip units.
Times are displayed in the browser's local timezone. Quotes are candle
snapshots, not executable broker bid/ask prices.

The page refreshes every five minutes while visible and supports manual
refresh. Requests share the existing symbol-specific candle cache. Opening
the monitor uses additional market-data API requests for the other pairs.
Unavailable or stale prices are labelled per pair; an old page or failed
refresh suppresses its displayed trading decision.

`GET /api/pairs` never switches the paper engine's configured symbol or
places an order. Only that configured pair can display the engine's current
decision, and only when its decision candle matches the latest closed candle.
Other pairs are monitoring-only. Paper results are aggregated from matching
symbol metadata in real-feed trade snapshots; synthetic and unidentified
records are excluded. No results are copied between pairs. Pairs without
recorded trades are labelled **Not tested**. Multi-pair automated execution
and independent strategy validation are separate work.

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

### Corrected paper execution (revision 2)

New runs use `backend/paper_trader_v2.db`. The original `paper_trader.db` is
preserved for comparison and is not used to train the corrected strategy.
Restart the backend to select the new journal; its initial balance comes from
`STARTING_BALANCE`. The new model and training state use
`prediction_model_v3.json` and `training_state_v3.json`, preserving the old files.
An old model cannot approve corrected setups; an unavailable model is reported
as unavailable and the other paper-entry rules still apply.

Paper entries use the latest current-candle price sample plus configured costs,
not the signal candle's old closing price. Stale prices and setups older than
the immediately preceding candle cannot open positions. Stops and targets use
only full candles beginning at or after the actual entry time, or later sampled
prices. The entry candle's earlier high/low is never used retroactively.
With candle snapshots, a touch between polls during the entry candle can be
missed; tick data would be needed to resolve that period exactly. Completed-bar
exits are timestamped at the bar's end, since OHLC does not reveal the touch time.

Each SMC liquidity pool can open only one trade, even after the trade closes or
the backend restarts. The daily and weekly checks reserve open-position risk
and the proposed order's risk before allowing an entry. These are limits under
the paper fill assumptions; gaps or different broker execution can exceed them.

Historical replay enters at the next candle's open plus costs. That is a
boundary-entry approximation to the paper engine's first available sample;
it does not reproduce polling delays. Model training uses the same next-open
assumption and a new cache fingerprint. Retrain offline with
`python train_model.py --offline` from `backend` after changing the strategy.
Run `python validate_execution.py --bars 5000` from `backend` to replay recent
cached prices with the account risk and setup reuse controls. It writes
`backend/data/execution_revision_2_recent_validation.json`. This diagnostic
excludes prediction and news gates and is not an untouched validation set.

- BUY confirmation: EMA20 > EMA50, MACD > signal, RSI 52–70, positive 10-bar momentum.
- SELL confirmation: EMA20 < EMA50, MACD < signal, RSI 30–48, negative 10-bar momentum.
- Stop: 1.0 ATR; target: 1.5 ATR; account risk: 1%.
- Up to `MAX_CONCURRENT_TRADES` EUR/USD paper positions can be open (default 3), all in the same direction.
- The engine evaluates once per newly closed 15-minute candle.
- It polls for market data every 60 seconds by default, but never evaluates the
  same closed candle twice.
- Weekly and monthly EMA trends must agree with a short-term BUY or SELL before
  the entry is permitted (configurable in `backend/.env`).
- Every closed paper trade records its entry indicators and outcome. A new entry is
  blocked only after it closely matches at least three earlier losing paper trades;
  this is a risk guard, not a price prediction.

No strategy guarantees a profit. Validate through extended paper trading before changing any rule.

## Entry models

`STRATEGY` in `backend/.env` selects which model produces BUY/SELL/HOLD. All
entry models pass through the engine's risk gates. The backtest uses the active
model and shares sizing, fills, position limits, account loss controls and
pattern learning. Historical news and time-appropriate saved prediction models
are not available to this replay, so those exclusions appear in its report.
The current saved model is never applied backwards to periods it trained on.

### `classic` (default)

EMA/MACD/RSI/momentum confluence, scored out of 100 and gated on ADX. See the
Strategy section above.

### `smc` — Smart Money Concepts / ICT

Trades a sequence rather than an indicator reading. A setup needs, in order:

1. **Liquidity sweep** - price runs past a confirmed swing high or low and
   closes back inside the range (20 points).
2. **Structure shift** - a close beyond the opposing swing that existed before
   the sweep, a change of character (20 points).
3. **Displacement** - a candle body of at least `SMC_DISPLACEMENT_ATR` x ATR in
   the setup direction (20 points).
4. **Point of interest** - price has retraced into the fair value gap left by
   that leg, or into the order block that preceded it (20 points).
5. **Premium/discount** - buys sit below and sells above the equilibrium of the
   current dealing range (10 points).
6. **Killzone** - London open 07:00-10:00, New York open 12:00-15:00 or London
   close 15:00-16:00 UTC (10 points).

Entries require all four steps (sweep, displacement, structure break and a later
retracement into a zone), an intact structural stop, and `SMC_SCORE_MIN` of 100.
A high score cannot substitute for missing steps. Unlike the classic model, an SMC signal
carries its own **structural stop** placed `SMC_STOP_BUFFER_ATR` x ATR beyond
the wick that took the liquidity, and targets the next opposing pool. If that
pool is closer than `SMC_MIN_REWARD` times the structural risk, the target
falls back to that R multiple instead. `ATR_STOP_MULTIPLIER` and
`ATR_TARGET_MULTIPLIER` are ignored while `STRATEGY=smc`.

Direction comes from the sweep itself, so the model does not need a separate
trend oracle - but the weekly/monthly filter and every other risk gate still
apply on top.

### `judas` — ICT Asian range / Judas swing

ICT's "Judas swing" is the London open's false move: early London trade runs
the stops resting beyond the quiet Asian session's range, then reverses toward
the other side. All times are UTC and do not follow daylight saving.

1. **Asian range** - high and low of 00:00-06:00; a range of at most
   `JUDAS_MAX_RANGE_PIPS` scores 20 points.
2. **London sweep** - between 06:00 and 10:00, price trades beyond one side
   (30 points).
3. **Close back inside** - the first candle that closes back inside the range
   after that sweep is the entry (30 points). Later candles never re-signal,
   so each side trades at most once a day.
4. **Rejection** - that candle's body is at least `JUDAS_REJECTION_ATR` x ATR
   in the reversal direction (20 points).

Entries need `JUDAS_SCORE_MIN` of 100. The stop sits `JUDAS_STOP_BUFFER_ATR`
x ATR beyond the sweep's extreme; the target is the opposite side of the Asian
range, or `JUDAS_MIN_REWARD` x risk when that side is closer.

## Validating a strategy honestly

For an account-level comparison using the corrected execution rules, run from
`backend`:

```powershell
.\.venv\Scripts\python.exe compare_strategies.py --bars 20000
```

This compares one existing preset each for Classic, SMC and Judas, with shared
costs and risk settings. The earlier 75% selects a candidate before the later
25% is evaluated. A strategy must have positive earlier equity return and at
least 30 closed trades to qualify; otherwise the comparison abstains. That
trade-count floor is a screening rule, not proof of statistical significance.
The later period receives earlier candles for indicator warm-up but cannot
open trades during warm-up. It starts with a fresh simulated account and memory.
The report includes open-position equity and writes `backend/data/strategy_comparison.json`.
It never changes the running strategy, journal or saved prediction model.

This is exploratory: the cached periods have been available to previous
research and are not a pristine holdout. Prediction/news gates are excluded.
The older `research.py` parameter sweep below is a separate legacy experiment;
its simulator does not reproduce all corrected paper execution and entry rules.
Use the shared-engine comparison above for the current implementation.

A parameter sweep is a multiple-comparisons experiment. Try enough variants on
one price series and the best one looks profitable whether or not any edge
exists, so a raw "best backtest" number means very little on its own.

`backend/research.py` runs the two checks that do mean something:

```powershell
cd backend
python research.py --download --years 5   # fetch history (cached after the first run)
python research.py --folds 6
```

**A. Full-sample sweep, selection-bias corrected.** Reports the best
configuration and its *deflated Sharpe ratio* - the probability its result
beats what the best of N random trials would show by luck alone. It also
reports how many trades a sweep of that size would actually need.

**B. Walk-forward.** Splits history into ordered blocks, picks parameters using
only each training block, then scores those parameters on the block that
follows. No fold can see its own future, so the pooled trades are genuinely out
of sample.

A strategy can pass A and fail B (parameters do not persist) or pass B and fail
A (it works, but the sweep winner was luck). Only passing both is evidence.

`app/history.py` handles the data: the provider caps a request at 5000 rows,
about seven weeks of 15-minute candles, so it pages backwards through
`end_date` windows and caches to `backend/data/`. Every window is written as it
arrives, so an interrupted download resumes instead of re-spending API quota.
The free plan allows 8 requests/minute; the default of 6 leaves headroom and
throttling is retried rather than aborting the run.

`app/validation.py` holds the statistics (deflated Sharpe, probabilistic
Sharpe, expected maximum Sharpe, minimum backtest length, walk-forward fold
construction) and is unit-tested independently of any strategy.

## Local prediction model

The optional local model estimates whether an existing BUY/SELL setup will hit
its take-profit before its stop-loss. It does not invent directions and sends
no market data to an AI service. The dashboard shows the probability for the
current setup, the model's out-of-sample record, and each journal trade's score.

Train it from the dashboard's **Train model** button, or with progress output:

```powershell
cd backend
python research.py --download --years 5   # once, if backend/data is empty
python train_model.py
```

Training replays every setup in the cached multi-year history (topped up with
the latest candles), labels each by whether target or stop came first, and fits
a logistic model on trend, volatility, timing, higher-timeframe alignment and,
for SMC, the setup's own structure, stop width and reward. The first run takes
about 15 minutes; replayed setups are cached in `backend/data/`, so later runs
only replay new candles.

It is then judged honestly by walk-forward: five later periods, each scored by
a model trained only on earlier data, with a gap as long as the outcome horizon
so no label leaks across. The threshold is chosen on earlier data too. The
report compares win rate and average R of all setups against the setups the
model allows, counted only over setups the engine would actually trade. If the
filter did not improve the out-of-sample average, its threshold is 0 and it
never blocks an entry.

A model is ignored (and reported as needing a retrain) if the strategy or any
setting that changes signals has changed since it was trained.
`PREDICTION_MIN_PROBABILITY` overrides the validated threshold; leave it unset.

A filter can only choose among the strategy's setups. If every group of setups
loses on average, the filter reduces losses but cannot make the strategy
profitable, and the dashboard says so.

## Learning from completed paper trades

Training combines cached market-history setups with eligible completed trades
from `paper_trader.db`. Both targets and stops are examples. The actual paper
outcome replaces a replay example for the same entry time and direction,
avoiding duplicate training evidence. Open trades and outcomes beyond the
model's outcome horizon are excluded from the classifier.

New entries save their exact model input vector, strategy-settings fingerprint,
symbol, interval and feed. Mismatched settings/feeds and incomplete snapshots
are excluded. Older trades without this metadata are admitted only when their
strategy and entry candle/price match the cached real price history. Their
original settings cannot be fully verified; the training report counts these
legacy examples separately. No API credentials are written into snapshots.

`PREDICTION_AUTO_RETRAIN=true` schedules background training after
`PREDICTION_RETRAIN_NEW_TRADES` new closed outcomes (default 5), with at least
`PREDICTION_RETRAIN_HOURS` between attempts (default 6). Attempt timestamps
persist across restarts. The dashboard reports included trades, pending
outcomes, exclusions and training progress. Failed training leaves the previous
model intact. The threshold stays off if walk-forward validation finds no
benefit; a few paper losses are not enough to claim a reliable probability.

Training without network access is also supported:

```powershell
cd backend
python train_model.py --offline
```

The existing Twelve Data key continues to fetch recent prices for normal
training and scans. The prediction model itself runs locally. An unnamed
`api_key` variable is not used: an additional provider must be identified and
integrated explicitly before any data is sent to it.

The separate pattern guard remembers recent similar wins and losses from the
same strategy, including structural setup checks. Its memory is limited by
`PATTERN_LOOKBACK_DAYS` (default 90). In a backtest it learns only from exits
that have already occurred in that replay, never from the live journal.

## Risk pauses and data checks

Daily and weekly loss checks accept both SQLite and ISO timestamps and use the
balance at the start of the period. After `MAX_CONSECUTIVE_LOSSES`, entries
pause for `LOSS_COOLDOWN_HOURS` (default 24) from the latest loss. Other loss
limits still apply when that cooldown ends. Open positions remain managed.
The dashboard shows the pause and the time at which it can be rechecked.

New entries are blocked for stale/future data, insufficient history or recent
weekday candle gaps. Invalid prices, duplicate timestamps and inconsistent
OHLC ranges are rejected. `MARKET_MAX_AGE_BARS` controls freshness (default 3).
Dashboard thresholds, trend-filter status, risk sizing and target descriptions
come from the backend configuration. Setup scores are rule scores, separate
from the model's estimated probability.

Model validation reports compare strategy candidates, not complete account
simulations: they do not include portfolio capacity, news or pattern-memory
gates. The dashboard displays that scope alongside its validation results.

The main dashboard card shows the last evaluated entry decision separately
from market bias, with its reason and signal candle time. Connection failures,
engine errors and expired signals suppress approval. The model summary compares
average R with and without filtering, highlights the latest unseen period and
provides a table of every validation period. A model allowing a setup does not
mean all other entry gates passed.

Recent candle requests share an in-memory cache inside each backend process.
Concurrent chart and engine requests reuse a snapshot for up to 60 seconds,
expiring at candle boundaries. Larger cached windows can serve smaller requests;
each caller receives its own copy. Failed refreshes raise an error instead of
serving expired cached data. Separate backend workers have separate caches.
