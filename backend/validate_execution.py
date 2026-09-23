"""Replay cached history with the active execution rules, without network calls."""
import argparse
import json
from pathlib import Path

from app.backtest import BacktestSettings, run_backtest
from app.config import settings
from app.history import load_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=int, default=5000, help="Number of recent cached bars (default 5000)")
    args = parser.parse_args()
    if args.bars < 300:
        parser.error("--bars must be at least 300")
    candles = load_cache(settings.symbol, settings.interval)
    if candles is None:
        parser.error("No cached market history is available")
    candles = candles.tail(args.bars).reset_index(drop=True)
    higher = {}
    if settings.higher_timeframe_filter:
        for name, interval in (("weekly", settings.weekly_interval), ("monthly", settings.monthly_interval)):
            frame = load_cache(settings.symbol, interval)
            if frame is None:
                parser.error(f"The enabled higher-timeframe filter requires cached {interval} history")
            higher[name] = frame
    print(f"Replaying {len(candles):,} cached bars: {candles.datetime.iloc[0]} to {candles.datetime.iloc[-1]}", flush=True)
    result = run_backtest(candles, BacktestSettings(
        settings.starting_balance, settings.risk_percent,
        settings.atr_stop_multiplier, settings.atr_target_multiplier,
        settings.spread_pips, settings.slippage_pips, settings.pip_size,
        settings.higher_timeframe_filter), higher)
    result.update(data_start=str(candles.datetime.iloc[0]), data_end=str(candles.datetime.iloc[-1]),
                  scope="Retrospective diagnostic on cached history, not an untouched holdout")
    output = Path(__file__).resolve().parent / "data" / "execution_revision_2_recent_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
