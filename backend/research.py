"""Walk-forward validation runner for the SMC entry model.

Answers two separate questions, because they are not the same question:

  A. Full-sample sweep -- is the best configuration credible once you account
     for having tried many? Reported as a deflated Sharpe ratio.
  B. Walk-forward -- if you had picked parameters using only past data at each
     point, would the resulting trades have made money? Reported as pooled
     out-of-sample trades that no fold's selection could see.

A strategy can pass A and fail B (the parameters do not persist) or pass B and
fail A (it works but the sweep winner was luck). Only passing both is evidence.

Usage:
    python research.py --years 5 --folds 6
    python research.py --download        # fetch/refresh history first
"""
import argparse
import asyncio
import itertools
import sys
import time

import numpy as np

from app import history, smc, validation
from app.config import settings
from app.strategy import enrich

LOOKBACK = 150
RISK_PERCENT = 1.0
START_BALANCE = 10_000.0
MAX_RISK_PRICE = 0.006  # reject setups whose structural stop is absurdly wide


def execution_cost() -> float:
    return (settings.spread_pips / 2 + settings.slippage_pips) * settings.pip_size


def candidate_grid() -> list[dict]:
    """Parameter combinations offered to each walk-forward training block."""
    grid = []
    for disp, window, strength, score_min, min_reward in itertools.product(
            (0.4, 0.6, 1.0), (10, 20), (2, 3), (50, 70, 90), (1.5, 2.0, 3.0)):
        grid.append({"disp": disp, "window": window, "strength": strength,
                     "score_min": score_min, "min_reward": min_reward})
    return grid


def structural_key(params: dict) -> tuple:
    """The part of a config that changes signal detection rather than filtering."""
    return params["disp"], params["window"], params["strength"]


def precompute(data, key: tuple, progress=None) -> list:
    """Best directional candidate per bar for one structural configuration."""
    disp, window, strength = key
    settings.smc_displacement_atr = disp
    settings.smc_sweep_window = window
    settings.smc_swing_strength = strength
    context = smc.prepare(data, strength)
    out: list = [None] * len(data)
    started = time.time()
    for i in range(LOOKBACK + 2, len(data) - 1):
        frame = data.iloc[i - LOOKBACK:i + 2]
        offset = i - LOOKBACK
        found = [c for c in (smc.evaluate(frame, "BULLISH", context, offset),
                             smc.evaluate(frame, "BEARISH", context, offset)) if c]
        if found:
            out[i] = max(found, key=lambda c: c["score"])
    if progress:
        progress(f"  precomputed disp={disp} window={window} strength={strength}: "
                 f"{sum(1 for x in out if x)} candidate bars in {time.time()-started:.0f}s")
    return out


def simulate(signals, high, low, params: dict, lo: int, hi: int) -> dict | None:
    """Replay one configuration over [lo, hi), mirroring backtest.py semantics."""
    cost = execution_cost()
    balance, trades, open_trade = START_BALANCE, [], None
    for i in range(max(lo, LOOKBACK + 2), hi):
        if open_trade:
            exit_price = None
            if open_trade["dir"] == 1:
                if low[i] <= open_trade["stop"]:
                    exit_price = open_trade["stop"]
                elif high[i] >= open_trade["target"]:
                    exit_price = open_trade["target"] - cost
            else:
                if high[i] >= open_trade["stop"]:
                    exit_price = open_trade["stop"]
                elif low[i] <= open_trade["target"]:
                    exit_price = open_trade["target"] + cost
            if exit_price is not None:
                pnl = (exit_price - open_trade["entry"]) * open_trade["units"] * open_trade["dir"]
                balance += pnl
                trades.append({"pnl": round(pnl, 2), "r": pnl / open_trade["risk"]})
                open_trade = None
        if open_trade is None:
            candidate = signals[i]
            if candidate is None or candidate["score"] < params["score_min"]:
                continue
            direction = 1 if candidate["kind"] == "BULLISH" else -1
            price = candidate["price"]
            risk_price = abs(price - candidate["stop_hint"])
            if risk_price <= 0 or risk_price > MAX_RISK_PRICE:
                continue
            target = candidate["target_hint"]
            reward = (target - price) * direction if target is not None else -1.0
            if reward / risk_price < params["min_reward"]:
                target = price + direction * risk_price * params["min_reward"]
            entry = price + direction * cost
            stop_distance = risk_price + cost
            risk_amount = balance * RISK_PERCENT / 100
            open_trade = {"entry": entry, "stop": entry - direction * stop_distance,
                          "target": target, "units": risk_amount / stop_distance,
                          "dir": direction, "risk": risk_amount}
    if not trades:
        return None
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit, gross_loss = sum(wins), abs(sum(losses))
    return {"n": len(trades), "net": round(sum(pnls), 2),
            "win_rate": round(len(wins) / len(trades) * 100, 1),
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else None,
            "returns": [t["r"] for t in trades]}


def report_full_sample(data, signal_cache, grid, log):
    """Sweep everything, then ask whether the winner beats best-of-N luck."""
    high, low = data.high.to_numpy(), data.low.to_numpy()
    results = []
    for params in grid:
        outcome = simulate(signal_cache[structural_key(params)], high, low,
                           params, 0, len(data) - 1)
        if outcome and outcome["n"] >= 10:
            outcome["sharpe"] = validation.sharpe_ratio(outcome["returns"])
            results.append((params, outcome))
    if not results:
        log("no configuration produced enough trades")
        return None
    sharpes = [o["sharpe"] for _, o in results]
    best_params, best = max(results, key=lambda item: item[1]["net"])
    stats = validation.deflated_sharpe_ratio(best["returns"], sharpes, n_trials=len(grid))
    log("")
    log("A. FULL-SAMPLE SWEEP (in-sample; selection bias corrected)")
    log(f"   configurations with >=10 trades : {len(results)} of {len(grid)}")
    log(f"   profitable configurations       : {sum(1 for _, o in results if o['net'] > 0)}")
    log(f"   best config                     : {best_params}")
    log(f"   best: trades={best['n']} win={best['win_rate']}% "
        f"PF={best['profit_factor']} net=${best['net']}")
    log(f"   observed Sharpe (per trade)     : {stats['observed_sharpe']}")
    log(f"   expected max Sharpe from luck   : {stats['expected_max_sharpe']}  "
        f"({stats['n_trials']} trials)")
    log(f"   DEFLATED SHARPE RATIO           : {stats['deflated_sharpe_ratio']}")
    log(f"   (undeflated PSR vs zero         : {stats['psr_vs_zero']})")
    needed = validation.minimum_backtest_length(len(grid), max(stats["observed_sharpe"], 1e-6))
    log(f"   trades needed for {len(grid)} trials      : {needed:,.0f} "
        f"(this sweep had {best['n']})")
    return stats


def report_walk_forward(data, signal_cache, grid, folds, log):
    """Choose parameters on each training block, score them on the next block."""
    high, low = data.high.to_numpy(), data.low.to_numpy()

    def evaluate(params, lo, hi):
        return simulate(signal_cache[structural_key(params)], high, low, params, lo, hi)

    result = validation.walk_forward(len(data) - 1, folds, grid, evaluate)
    summary = result.summary()
    log("")
    log("B. WALK-FORWARD (out-of-sample; parameters chosen on past data only)")
    log(f"   folds                           : {summary['folds']} "
        f"({summary['folds_scored']} produced trades)")
    log(f"   folds profitable out of sample  : {summary['folds_profitable_out_of_sample']}")
    log(f"   pooled out-of-sample trades     : {summary['pooled_trades']}")
    log(f"   pooled net                      : ${summary['pooled_net']}")
    log(f"   pooled Sharpe (per trade)       : {summary['pooled_sharpe']}")
    if result.pooled_returns:
        psr = validation.probabilistic_sharpe_ratio(result.pooled_returns, 0.0)
        log(f"   P(true Sharpe > 0)              : {psr:.4f}")
    log("")
    log("   fold detail:")
    for fold in result.folds:
        if not fold.test_result:
            log(f"     fold {fold.index}: no out-of-sample trades")
            continue
        chosen = fold.chosen
        log(f"     fold {fold.index}: train net=${fold.train_result['net']:>9.2f} "
            f"(n={fold.train_result['n']:>3})  ->  test net=${fold.test_result['net']:>9.2f} "
            f"(n={fold.test_result['n']:>3}, win={fold.test_result['win_rate']}%)  "
            f"disp={chosen['disp']} win={chosen['window']} st={chosen['strength']} "
            f"scr={chosen['score_min']} R={chosen['min_reward']}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=float, default=5.0)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--download", action="store_true",
                        help="fetch history from the provider before running")
    args = parser.parse_args()

    def log(message=""):
        print(message, flush=True)

    if args.download:
        asyncio.run(history.download(years=args.years, progress=log))
    candles = history.load_cache(settings.symbol, settings.interval)
    if candles is None or candles.empty:
        log("No cached history. Run with --download first.")
        return 1
    data = enrich(candles)
    log(f"history: {len(data)} enriched bars, "
        f"{data.datetime.min().date()} -> {data.datetime.max().date()}")

    grid = candidate_grid()
    keys = sorted({structural_key(p) for p in grid})
    log(f"grid: {len(grid)} configurations across {len(keys)} signal passes")
    signal_cache = {key: precompute(data, key, log) for key in keys}

    report_full_sample(data, signal_cache, grid, log)
    report_walk_forward(data, signal_cache, grid, args.folds, log)
    log("")
    log("Reminder: A measures whether the sweep winner is distinguishable from "
        "luck.\nB measures whether the process of choosing parameters generalises. "
        "Neither\nis a forecast, and both assume the modelled spread and slippage hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
