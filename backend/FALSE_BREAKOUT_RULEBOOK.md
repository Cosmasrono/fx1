# False-breakout study: rulebook v1

Status: proposed paper-research specification, not implemented or validated.
Code/configuration reviewed September 16, 2026. Loaded configuration selects
`smc`, EUR/USD, 15min; the separate `judas` module is the closest existing model.
The numerical thresholds below are starting hypotheses, not proven advantages.

## The setup to learn

Test whether a failed break of a fixed overnight range is followed by a move
toward the opposite boundary. Price alone does not establish who placed orders
or their motives. Study EUR/USD on 15-minute candles only.

| Rule | Frozen study definition |
| --- | --- |
| Range | High/low of all 24 candles starting 00:00 through 05:45 UTC on the same day. Skip missing, duplicate or invalid bars. |
| Size | Range must be positive and no wider than 30 pips; EUR/USD pip = 0.0001. |
| Window | Sweep and reclaim candles must start at or after 06:00 UTC and close before 10:00 UTC. These are fixed UTC hours, without daylight-saving adjustment. In Nairobi: range 03:00–09:00, entries 09:00–13:00. |
| Sweep | High strictly exceeds the range high (sell candidate), or low strictly falls below the range low (buy candidate). No minimum penetration in v1. |
| Confirmation | First candle closing strictly inside the range after the first sweep. Sweep and reclaim may occur in the same candle. Sell requires a bearish body; buy requires a bullish body; body size must be at least 0.3 ATR. |
| ATR | Use the existing 14-period exponentially smoothed true range (`alpha=1/14`) through the closed confirmation candle, with preceding history for warm-up. |
| Entry trigger | Completion of that reclaim candle. This version does **not** wait for a later break of its high/low. Backtest at the next candle open plus configured costs; forward paper at the first fresh observable price after confirmation. No entry at/after 10:00 UTC. |
| Stop | Beyond the most extreme price from first sweep through reclaim, plus 0.25 ATR outward. Never widen after entry. |
| Target | Opposite range boundary, fixed. Skip if reward divided by stop distance is below 1.5 at the cost-adjusted entry; never extend the target to manufacture that ratio. |
| Ambiguity/repeats | If both boundaries have been swept before entry, skip the day. At most the first candidate per side; a failed confirmation consumes that side. At most one filled trade per UTC day and one open study trade. Persist identifiers across restarts. |
| Exit | First stop/target hit; use stop-first when a historical bar touches both and order is unknown. No trailing, partial exits or time exit in v1; an open trade blocks further study entries. |
| Paper risk | Size at 1% of current paper balance to the stop. Apply existing daily 2%, weekly 5%, and consecutive-loss gates, including reserved open risk. This is a comparison assumption, not a live-account recommendation. |
| Data/costs | Real-feed data only; reject stale/non-contiguous entry data and invalid stop/target geometry. Use the existing 0.8-pip spread/0.1-pip slippage simulation, then stress costs separately. These fixed values are not observed broker quotes. |

No AI, prediction, news or higher-timeframe filter in the isolated baseline.
Record scheduled-news context separately; adding a news exclusion changes the
version and requires a defined calendar and time window. Record every skip.

## Does the current bot follow this?

No. Selecting `judas` alone would not implement this specification.

| Area | Existing implementation | Study difference |
| --- | --- | --- |
| Active model | `app/active_strategy.py` dispatches to loaded `smc`. | Requires a separately identified study implementation/run. |
| Range data | `judas.evaluate` accepts at least 75% of expected candles. | Requires all 24 unique valid bars. |
| Mandatory checks | Sweep + reclaim score 60; tight range and rejection each add 20. Threshold is 80. | Both tight range and rejection are mandatory; current code can accept either one alone. |
| Trigger | First close inside, including same-candle sweep/reclaim. | Matches the chosen reclaim trigger; not the subsequent-break example discussed earlier. |
| Target | Falls back to a 1.5R target when the opposite boundary is too close. | Skip instead; recheck ratio at actual entry. |
| Session boundary | Checks candle start is before 10:00, so 09:45 candle can signal. | Confirmation must finish before 10:00 and entry must be before 10:00. |
| Repeats/ambiguity | First reclaim per side; chooses highest score/deepest sweep if both qualify. | Skip dual-sided sweeps; enforce persistent daily fill limit. |
| Execution | Engine samples the current candle; backtest uses next open. | Preserve realistic fills and report polling differences; no assumed fill at the signal close. |

References: `app/judas.py`, `app/config.py`, `app/strategy.py`,
`app/engine.py`, `app/backtest.py`, `tests/test_judas.py`.

## Practice and evidence log

Replay charts with future candles hidden. For each candidate save UTC date,
range boundaries/width, sweep and confirmation timestamps, ATR, checklist
results, planned/actual entry, stop, target, initial risk, costs, skip reason,
exit and net result in R (net profit/loss divided by initial monetary risk).
Save a chart at the decision and after the exit. Classify an execution mistake
separately from a valid losing trade. Include skipped and ambiguous examples.

First manually label 20 examples, including failures, to check that the rules
are reproducible. That count is a learning exercise, not evidence of an edge.
Then implement this version in isolation, verify the boundary cases above,
and compare the code's decisions with those labels before evaluating returns.

Freeze rules before testing later data. Previously examined history is
development data: the existing Judas result (-14.60% earlier, +8.66% later)
does not validate this new version. See [comparison](STRATEGY_COMPARISON.md).
Report trade count, net average R, profit factor, drawdown, open-position
equity, uncertainty and results by time period; stress execution costs.
Use newly collected forward paper data for additional evidence. Do not
promote on win rate alone or retune after individual losses. Any change starts
a new version and must be evaluated again on subsequent untouched data.
