# Forex strategy comparison — September 15, 2026

## Outcome

None of the three current presets demonstrated consistently positive results
across both periods. SMC was selected using the earlier period, but lost equity
in the later period and had fewer than 30 later closed trades. Judas had the
strongest later result, but failed the earlier screen. Switching to it based on
the later result would be a new research hypothesis, not a validated selection.

The running SMC configuration, prediction model and trade journal were not
changed by this comparison.

## Results

The earlier period covers February 18–July 24, 2026; the later period covers
July 24–September 14, 2026. Exact boundaries are stored in the JSON report.

| Preset | Earlier equity return | Earlier closed trades | Later equity return | Later closed trades | Later profit factor |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classic | -49.72% | 272 | -22.46% | 108 | 0.65 |
| SMC | +2.21% | 111 | -4.53% | 16 | 0.61 |
| Judas | -14.60% | 79 | +8.66% | 33 | 1.51 |

SMC's earlier profit factor was only 1.03. Its later account ended with
$9,592.10 realized balance and $9,546.82 equity because one position remained
open. The other later accounts had no open positions. Equity returns include
open-position P&L at the final available opening-price mark.

The later SMC result differs from the earlier 5,000-bar diagnostic because
this comparison supplies 300 preceding bars for indicator warm-up. No trades
are allowed in that warm-up period.

## Method

- 20,000 cached EUR/USD 15-minute candles, in chronological order.
- One existing preset per strategy, with no parameter search.
- Shared $10,000 initial balance, 1% risk per order, 0.8-pip spread and
  0.1-pip slippage settings; higher-timeframe filter disabled as configured.
- Shared position, daily/weekly loss, consecutive-loss and pattern-memory rules.
- Corrected next-open execution and SMC setup-deduplication rules.
- Earlier 75% used for selection. The candidate needed positive equity return
  and at least 30 closed trades; highest earlier equity return won.
- The choice was frozen before later-period results were evaluated. Each
  period starts with a fresh simulated account and pattern memory.
- The later 25% receives earlier candles only for indicator warm-up.

The 30-trade minimum is an exploratory screening floor, not statistical proof.
The cached history has been used in previous research and is not a pristine
holdout. These results do not establish future profitability or prove that all
possible variants of a strategy fail.

Prediction and news gates are excluded. The saved current prediction model
cannot be applied to historical periods it trained on. Price data is cached
OHLC, not broker bid/ask or tick history. Fixed costs, next-open fills,
stop-first ambiguous candles and sampled drawdown are simulation assumptions.

## Reproduce

From `backend`:

```powershell
.\.venv\Scripts\python.exe compare_strategies.py --bars 20000
```

Detailed results and settings: `data/strategy_comparison.json`.

The existing 130-test suite passed; the added open-position equity regression
also passed along with the other comparison tests (131 distinct tests total).
Tests cover selection without later-period results, warm-up entry blocking,
temporary-setting restoration, abstention and floating-loss accounting.

## Interpretation

The evidence does not support an automatic strategy replacement. A separate
forward paper trial of Judas could test whether its recent improvement
persists, with rules fixed before collecting new outcomes. Its recent gain
alone is insufficient justification to promote it to the active strategy.
