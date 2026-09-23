"""Offline comparison of three fixed forex strategies on chronological periods.

Run as a separate CLI process, never inside the API: temporary settings and
signal wrapping are local to this process. No broker calls, journal writes,
model training or configuration-file changes are made.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from app import backtest
from app.config import settings
from app.history import load_cache
from app.market import validate_candles

STRATEGIES = ("classic", "smc", "judas")
MIN_TRADES = 30  # Screening floor, not a statistical guarantee.


@contextmanager
def isolated_strategy(name, entry_start):
    previous, original_signal = settings.strategy, backtest._signal_at
    def bounded_signal(data, index):
        signal = original_signal(data, index)
        if pd.Timestamp(signal["candle_time"]) < entry_start:
            signal.update(action="HOLD", reason="Comparison warm-up: no entries.")
        return signal
    try:
        settings.strategy = name
        backtest._signal_at = bounded_signal
        yield
    finally:
        settings.strategy = previous
        backtest._signal_at = original_signal


def choose_development(results, minimum=MIN_TRADES):
    eligible = {name: result for name, result in results.items()
                if result["metrics"]["closed_trades"] >= minimum
                and result["metrics"]["equity_return_percent"] > 0}
    return max(eligible, key=lambda name: eligible[name]["metrics"]["equity_return_percent"]) if eligible else None


def periods(candles, fraction=.75):
    if len(candles) < 1200 or not .5 <= fraction <= .85:
        raise ValueError("Need at least 1200 bars and a development share between 0.5 and 0.85")
    split = int(len(candles) * fraction)
    start = pd.Timestamp(candles.datetime.iloc[split])
    return candles.iloc[:split].copy(), candles.iloc[max(0, split - 300):].copy(), start


def replay(frame, name, start, config, higher):
    with isolated_strategy(name, start):
        return backtest.run_backtest(frame, config, higher)


def compare(candles, config, higher=None, progress=print):
    development, validation, validation_start = periods(candles)
    result = {
        "scope": "Exploratory comparison on existing cached history; later period was not used for this run's selection, but is not a pristine holdout",
        "selection_rule": "Highest development equity return among strategies with positive equity return and at least 30 closed trades; otherwise abstain",
        "minimum_trades": MIN_TRADES,
        "shared_settings": asdict(config),
        "risk_settings": {k: getattr(settings, k) for k in (
            "max_concurrent_trades", "max_daily_loss_percent", "max_weekly_loss_percent",
            "max_consecutive_losses", "loss_cooldown_hours", "setup_score_min", "adx_min",
            "pattern_min_loss_matches", "pattern_similarity_threshold", "pattern_lookback_days")},
        "strategy_settings": {k: v for k, v in settings.model_dump().items()
                              if k.startswith(("smc_", "judas_"))},
        "periods": {"development_start": str(development.datetime.iloc[0]),
                    "development_end": str(development.datetime.iloc[-1]),
                    "validation_start": str(validation_start),
                    "validation_end": str(validation.datetime.iloc[-1])},
        "notes": ["One existing preset per strategy; no parameter search",
                  "Each period starts with a fresh simulated account and pattern memory",
                  "Validation receives 300 earlier bars for indicator warm-up only; no entries before validation_start",
                  "Next-open fills; stop-first on ambiguous bars; fixed execution costs",
                  "Equity includes open positions marked at the final row's open; drawdown uses sampled marks",
                  "Prediction, news and live-feed freshness gates excluded",
                  "The final row supplies an opening price but cannot settle a trade",
                  "Results do not change the active strategy"],
        "development": {}, "validation": {},
    }
    for name in STRATEGIES:
        progress(f"Development: {name} ({len(development):,} bars)")
        report = replay(development, name, pd.Timestamp(development.datetime.iloc[0]), config, higher or {})
        result["development"][name] = report
        progress(f"  {report['metrics']}")
    # This choice is frozen BEFORE examining any later-period results.
    result["selected_on_development"] = choose_development(result["development"])
    progress(f"Development choice: {result['selected_on_development'] or 'none qualified'}")
    for name in STRATEGIES:
        progress(f"Validation: {name} ({len(validation):,} bars including warm-up)")
        report = replay(validation, name, validation_start, config, higher or {})
        result["validation"][name] = report
        progress(f"  {report['metrics']}")
    chosen = result["selected_on_development"]
    if chosen is None:
        result["conclusion"] = "No strategy qualified on the earlier period. Do not select a replacement from the later-period winner."
    else:
        metrics = result["validation"][chosen]["metrics"]
        if metrics["closed_trades"] < MIN_TRADES:
            result["conclusion"] = f"{chosen} qualified earlier, but has too few later closed trades for the screening rule."
        elif metrics["equity_return_percent"] <= 0:
            result["conclusion"] = f"{chosen} qualified earlier but lost equity in the later period. No demonstrated improvement."
        else:
            result["conclusion"] = f"{chosen} passed this exploratory screen; it needs new forward paper evidence before any deployment decision."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=int, default=20000)
    args = parser.parse_args()
    candles = load_cache(settings.symbol, settings.interval)
    if candles is None or args.bars < 1200:
        parser.error("Need cached history and --bars >= 1200")
    candles = candles.tail(args.bars).reset_index(drop=True)
    validate_candles(candles)
    higher = {}
    if settings.higher_timeframe_filter:
        for name, interval in (("weekly", settings.weekly_interval), ("monthly", settings.monthly_interval)):
            frame = load_cache(settings.symbol, interval)
            if frame is None:
                parser.error(f"Enabled trend filter requires cached {interval} history")
            higher[name] = frame
    config = backtest.BacktestSettings(
        settings.starting_balance, settings.risk_percent, settings.atr_stop_multiplier,
        settings.atr_target_multiplier, settings.spread_pips, settings.slippage_pips,
        settings.pip_size, settings.higher_timeframe_filter)
    result = compare(candles, config, higher, lambda message: print(message, flush=True))
    output = Path(__file__).resolve().parent / "data" / "strategy_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result["conclusion"])
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
