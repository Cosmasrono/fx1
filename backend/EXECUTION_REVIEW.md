# Execution correction and validation

## Changes

- Entries use the current candle's sampled closing price plus configured costs.
  A stale current candle or an older setup cannot create an entry.
- Exit ranges cannot precede the actual entry. Entry-candle exits use subsequent
  price samples; full later candles are replayed in chronological order per trade.
- Completed-bar exits are recorded at the end of the bar rather than its start.
- SMC requires a sweep, displacement, structure break and a subsequent zone
  retracement. The stop must not already have been breached after the sweep.
- A stable liquidity-pool ID and a unique database index prevent repeat trades,
  including after closure and restart.
- Open-position risk and proposed risk must fit within daily and weekly budgets.
- Invalid structural levels after repricing block the entry.
- Historical replay and model labels use the next candle's open plus costs.

## Separate evaluation files

The corrected engine uses `paper_trader_v2.db`; the original `paper_trader.db`
remains in place. The new database initializes on backend startup, using the
configured starting balance. Previous trades do not enter the new journal's
pattern memory or training feedback.

Model version 3 uses `prediction_model_v3.json` and `training_state_v3.json`.
Previous model files and candidate caches remain available for comparison.
Restart the backend to load the corrected code and journal.

## Verification

All 126 backend tests pass, including regression cases for the pre-entry exit,
stale-price rejection, actual sampled entry pricing, consumed setups after a
restart, each missing SMC step, invalidated stops, reserved risk and next-open
historical execution. Whitespace checks pass for the modified tracked files.

The recent replay used 5,000 cached bars from July 24 to September 14, 2026,
with SMC, the higher-timeframe filter off, 1% risk per order and the existing
spread/slippage assumptions:

| Metric | Result |
| --- | ---: |
| Closed trades | 15 |
| Open trades at end | 1 |
| Peak concurrent positions | 2 |
| Win rate | 26.7% |
| Realized P&L | -$517.47 |
| Ending balance | $9,482.53 |
| Realized return | -5.17% |
| Profit factor | 0.50 |

This result remains negative. It is a retrospective diagnostic on previously
available history, not an untouched holdout or evidence of future profitability.
The ending balance excludes the remaining position's floating P&L.

Reproduce from `backend`:

```powershell
.\.venv\Scripts\python.exe validate_execution.py --bars 5000
.\.venv\Scripts\python.exe train_model.py --offline
```

The account replay writes `data/execution_revision_2_recent_validation.json`.
The trained model's report contains its separate walk-forward setup evaluation.

## Full-cache model validation

Offline retraining completed on 5,760 resolved setups spanning August 2021 to
September 2026, with no old-journal feedback included. Across five later test
periods (4,662 baseline setups), the results were:

| Metric | Result |
| --- | ---: |
| Mean ranking AUC | 0.514 |
| Baseline win rate | 37.0% |
| Baseline average return | -0.099R |
| Filtered win rate | 39.6% |
| Filtered average return | -0.059R |
| Final fitted model's entry threshold | 43.7% |

R is the configured amount risked per setup. The filter improves this historical
candidate average, but it remains negative. Its final threshold was selected
using the training procedure; the walk-forward result evaluates thresholds
selected separately within each fold. A model accepting a setup does not show
that the setup, strategy or full account simulation is profitable.

The latest test period was worse after filtering: -0.194R without the filter
versus -0.235R with it. The pooled improvement is therefore not consistent
across periods and does not justify calling the corrected strategy profitable.

## Limits

The source is candle snapshots, not tick or broker quote history. Entry-candle
touches between samples cannot be reconstructed exactly. Historical next-open
fills approximate boundary execution and do not reproduce polling delays.
Account replay excludes news and time-appropriate prediction models; classifier
validation evaluates candidates without account capacity, setup deduplication,
news or pattern-memory gates. Neither is a live execution result.

The engine's historical exit recovery remains bounded by the fetched candle
window. Extended downtime can leave missing price history. Stop-first treatment
of a candle touching both levels is conservative, and assumed fixed costs and
stop fills do not model broker gaps. The reserved-risk limits therefore describe
the paper model rather than a guaranteed maximum real-world loss.
