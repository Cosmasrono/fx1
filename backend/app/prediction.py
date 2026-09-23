"""Local win-probability model for rule-based trade candidates.

It estimates whether a BUY/SELL setup from the active entry model reaches its
take-profit before its stop-loss. It never creates a direction, sends no market
data anywhere, and only blocks entries when out-of-sample evidence showed that
blocking improved the average result.

Training replays the cached multi-year history from ``history.py`` rather than
the ~7 weeks a single API request returns: a few weeks of highly correlated
setups cannot tell skill from luck. Replayed setups are cached per strategy
configuration, so only new candles are replayed on a retrain.
"""
import bisect
import hashlib
import json
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import history, smc
from . import journal
from .active_strategy import active_name, signal_from
# Same window the backtest hands the entry model, which matches what the live
# engine sees after enrich() trims its warm-up rows.
from .backtest import SIGNAL_LOOKBACK, _prepare_higher_timeframe_trends
from .config import settings
from .market import interval_minutes
from .strategy import enrich

MODEL_PATH = Path(__file__).resolve().parents[1] / "prediction_model_v3.json"
CANDIDATE_DIR = history.CACHE_DIR
VERSION = 3
LEARNING_VERSION = 1
TRAINING_STATE_PATH = Path(__file__).resolve().parents[1] / "training_state_v3.json"
BASE_FEATURES = ("rsi", "adx", "momentum", "vol_20", "ema_gap_atr", "macd_gap_atr",
                 "price_gap_atr", "side", "hour_sin", "hour_cos", "htf_aligned")
SMC_FEATURES = ("setup_score", "structure_shift", "displacement", "point_of_interest",
                "premium_discount", "killzone", "stop_atr", "reward_r", "liquidity_target",
                "equilibrium_gap_atr", "swept_gap_atr", "poi_fvg", "poi_order_block")
JUDAS_FEATURES = ("setup_score", "tight_range", "rejection", "stop_atr", "reward_r",
                  "liquidity_target", "range_atr", "sweep_depth_atr")
# Settings that change which setups a strategy emits, beyond the shared ones.
JUDAS_SETTINGS = ("judas_range_start_hour", "judas_range_end_hour", "judas_window_end_hour",
                  "judas_max_range_pips", "judas_rejection_atr", "judas_stop_buffer_atr",
                  "judas_min_reward", "judas_score_min")
# The engine adds a higher-timeframe bonus to setup_score after the entry model
# runs. Summing only the model's own components keeps training and live equal.
SMC_COMPONENTS = ("liquidity_sweep", "structure_shift", "displacement",
                  "point_of_interest", "premium_discount", "killzone")
FOLDS = 5
# The threshold search must keep at least this many validation setups, so a
# lucky handful of trades can never be the evidence for a filter.
MIN_KEPT_SHARE = 0.2
MIN_KEPT_TRADES = 30


def _quiet(stage: str, done: int = 0, total: int = 0) -> None:
    pass


def feature_names(strategy: str) -> tuple[str, ...]:
    return BASE_FEATURES + {"smc": SMC_FEATURES, "judas": JUDAS_FEATURES}.get(strategy, ())


def _population() -> str:
    """Which setups reach the model: the trend filter decides, so it scopes the evidence."""
    return "trend-aligned setups" if settings.higher_timeframe_filter else "setups"


def execution_cost() -> float:
    return (settings.spread_pips / 2 + settings.slippage_pips) * settings.pip_size


def fingerprint() -> str:
    """Identifies the settings a model and its cached setups were built under."""
    keys = {name: getattr(settings, name) for name in (
        "symbol", "interval", "strategy", "use_synthetic_data", "spread_pips", "slippage_pips",
        "pip_size", "atr_stop_multiplier", "atr_target_multiplier", "adx_min", "setup_score_min",
        "smc_swing_strength", "smc_sweep_window", "smc_displacement_atr", "smc_stop_buffer_atr",
        "smc_score_min", "smc_min_reward", "prediction_max_horizon_bars")}
    if settings.strategy == "judas":
        keys.update({name: getattr(settings, name) for name in JUDAS_SETTINGS})
    keys.update(version=VERSION, lookback=SIGNAL_LOOKBACK)
    return hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest()[:12]


def _levels(signal: dict, cost: float) -> tuple[int, float, float, float, float]:
    """Direction, entry, stop, target and per-unit risk, exactly as order_levels sizes them."""
    direction = 1 if signal["action"] == "BUY" else -1
    entry = signal.get("execution_price", signal["price"]) + direction * cost
    hint = signal.get("levels_hint")
    if hint:
        stop, target = hint["stop"], hint["target"]
        risk = abs(entry - stop) + cost
    else:
        atr = signal["indicators"]["atr"]
        risk = atr * settings.atr_stop_multiplier + cost
        stop = entry - direction * risk
        target = entry + direction * atr * settings.atr_target_multiplier
    return direction, entry, stop, target, risk


def features(signal: dict) -> dict[str, float]:
    """Scale-free description of a setup. Trend terms are signed by trade direction."""
    values = signal["indicators"]
    direction = 1.0 if signal["action"] == "BUY" else -1.0
    atr = max(abs(float(values["atr"])), 1e-9)
    price = float(signal["price"])
    ema20 = float(values["ema20"])
    stamp = pd.Timestamp(signal["candle_time"])
    angle = (stamp.hour + stamp.minute / 60) / 24 * 2 * math.pi
    expected = "BULLISH" if direction > 0 else "BEARISH"
    row = {
        "rsi": float(values["rsi"]),
        "adx": float(values["adx"]),
        "momentum": float(values["momentum_10"]) * direction,
        "vol_20": float(values["vol_20"]),
        "ema_gap_atr": (ema20 - float(values["ema50"])) / atr * direction,
        "macd_gap_atr": (float(values["macd"]) - float(values["macd_signal"])) / atr * direction,
        "price_gap_atr": (price - ema20) / atr * direction,
        "side": direction,
        "hour_sin": math.sin(angle),
        "hour_cos": math.cos(angle),
        "htf_aligned": float(values.get("weekly_trend") == expected
                             and values.get("monthly_trend") == expected),
    }
    model = signal.get("model")
    if model in ("smc", "judas"):
        breakdown = signal.get("score_breakdown") or {}
        cost = execution_cost()
        _, entry, _, target, risk = _levels(signal, cost)
        risk = max(risk, 1e-9)
        row.update({
            "setup_score": float(sum(value for name, value in breakdown.items()
                                     if name != "higher_timeframes")),
            "stop_atr": risk / atr,
            "reward_r": (abs(target - entry) - cost) / risk,
            "liquidity_target": float(values.get("target_source") == "LIQUIDITY"),
        })
    if model == "judas":
        breakdown = signal.get("score_breakdown") or {}
        row.update({
            "tight_range": float(breakdown.get("tight_range", 0) > 0),
            "rejection": float(breakdown.get("rejection", 0) > 0),
            "range_atr": float(values.get("range_pips", 0)) * settings.pip_size / atr,
            "sweep_depth_atr": float(values.get("sweep_depth_pips", 0)) * settings.pip_size / atr,
        })
    if model == "smc":
        equilibrium, swept = values.get("equilibrium"), values.get("swept_level")
        row.update({
            **{name: float(breakdown.get(name, 0) > 0) for name in SMC_COMPONENTS[1:]},
            "equilibrium_gap_atr": (price - equilibrium) / atr * direction if equilibrium else 0.0,
            "swept_gap_atr": (price - swept) / atr * direction if swept is not None else 0.0,
            "poi_fvg": float(values.get("poi") == "FVG"),
            "poi_order_block": float(values.get("poi") == "ORDER_BLOCK"),
        })
    return row


# --------------------------------------------------------------------------
# Historical replay
# --------------------------------------------------------------------------

def _higher_timeframe_lookup(candles: pd.DataFrame) -> dict:
    """Weekly and monthly trend over time, rebuilt from the intraday history.

    Periods are labelled by their start (Monday / the 1st), as the provider
    labels them, and backtest's helper only exposes a period's trend once the
    next period has begun -- the same "last completed candle" the engine reads.
    """
    frame = candles[["datetime", "open", "high", "low", "close"]]
    stamp = frame.datetime
    keys = {"weekly": (stamp - pd.to_timedelta(stamp.dt.dayofweek, unit="D")).dt.normalize(),
            "monthly": stamp.dt.normalize() - pd.to_timedelta(stamp.dt.day - 1, unit="D")}
    ohlc = {"open": "first", "high": "max", "low": "min", "close": "last"}
    frames = {name: frame.drop(columns="datetime").groupby(key.to_numpy()).agg(ohlc)
              .rename_axis("datetime").reset_index() for name, key in keys.items()}
    for grouped in frames.values():
        grouped["datetime"] = pd.to_datetime(grouped["datetime"], utc=True)
    trends = _prepare_higher_timeframe_trends(frames)
    return {name: ([time for time, _ in points], [trend for _, trend in points])
            for name, points in trends.items()}


def _trends_at(lookup: dict, when: pd.Timestamp) -> dict[str, str]:
    result = {}
    for name, (times, trends) in lookup.items():
        position = bisect.bisect_right(times, when) - 1
        result[name] = trends[position] if position >= 0 else "UNAVAILABLE"
    return result


def _prepare(data: pd.DataFrame):
    """Whole-series SMC structure, precomputed once for a fast replay."""
    if settings.strategy != "smc":
        return None
    context = smc.prepare(data)
    return context, [s.index for s in context.swings], [z.index for z in context.gaps]


def _signal_at(data: pd.DataFrame, index: int, prepared) -> dict:
    """The live signal for closed candle ``index``; the next row plays the forming candle."""
    start = max(0, index + 2 - SIGNAL_LOOKBACK)
    window = data.iloc[start:index + 2]
    if prepared is None:
        return signal_from(window)
    context, swing_index, gap_index = prepared
    # smc filters the context by position anyway; handing it only this
    # window's slice keeps each step O(window) instead of O(history).
    local = smc.Context(
        context.swings[bisect.bisect_left(swing_index, start):bisect.bisect_right(swing_index, index + 1)],
        context.gaps[bisect.bisect_left(gap_index, start):bisect.bisect_right(gap_index, index + 1)],
        context.strength)
    return smc.signal_from(window, local, start)


def _outcome(signal: dict, high, low, index: int, cost: float) -> tuple[int, float] | None:
    """(win, R) for a setup entered at the close of ``index``, or None if unresolved."""
    direction, entry, stop, target, risk = _levels(signal, cost)
    if not math.isfinite(risk) or risk <= 0:
        return None
    last = min(index + 1 + settings.prediction_max_horizon_bars, len(high))
    for j in range(index + 1, last):
        stop_hit = low[j] <= stop if direction == 1 else high[j] >= stop
        target_hit = high[j] >= target if direction == 1 else low[j] <= target
        if stop_hit:  # conservative when both print in one candle, as the engine assumes
            return 0, (stop - entry) * direction / risk
        if target_hit:
            return 1, (target - direction * cost - entry) * direction / risk
    return None


def _columns() -> list[str]:
    return ["time", *feature_names(settings.strategy), "gate_score", "win", "r"]


def _replay(data, lookup, prepared, start: int, stop: int, progress) -> pd.DataFrame:
    names = feature_names(settings.strategy)
    cost = execution_cost()
    high, low = data.high.to_numpy(), data.low.to_numpy()
    first = max(start, SIGNAL_LOOKBACK)
    rows = []
    for index in range(first, stop):
        if (index - first) % 1000 == 0:
            progress("Replaying candles", index - first, stop - first)
        signal = _signal_at(data, index, prepared)
        if signal["action"] not in ("BUY", "SELL"):
            continue
        signal["execution_price"] = float(data.open.iloc[index + 1])
        direction, entry, stop_price, target, risk = _levels(signal, cost)
        if (entry - stop_price) * direction <= 0 or (target - entry) * direction <= cost:
            continue
        trends = _trends_at(lookup, data.datetime.iloc[index + 1])
        signal["indicators"]["weekly_trend"] = trends["weekly"]
        signal["indicators"]["monthly_trend"] = trends["monthly"]
        outcome = _outcome(signal, high, low, index, cost)
        if outcome is None:
            continue
        row = features(signal)
        rows.append({"time": signal["candle_time"], **{name: row[name] for name in names},
                     "gate_score": signal["setup_score"], "win": outcome[0], "r": outcome[1]})
    return pd.DataFrame(rows, columns=_columns())


def _candidate_paths() -> tuple[Path, Path]:
    stem = CANDIDATE_DIR / f"prediction_candidates_{fingerprint()}"
    return stem.with_suffix(".csv"), stem.with_suffix(".json")


def _load_candidates() -> tuple[pd.DataFrame | None, str | None]:
    table, meta = _candidate_paths()
    if not table.exists() or not meta.exists():
        return None, None
    frame = pd.read_csv(table, float_precision="round_trip")
    if list(frame.columns) != _columns():
        return None, None
    return frame, json.loads(meta.read_text(encoding="utf-8"))["replayed_through"]


def _save_candidates(frame: pd.DataFrame, through) -> None:
    table, meta = _candidate_paths()
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(table, index=False)
    meta.write_text(json.dumps({"replayed_through": pd.Timestamp(through).isoformat()}), encoding="utf-8")


def build_dataset(candles: pd.DataFrame, progress=_quiet, use_cache: bool = True) -> pd.DataFrame:
    """Every resolved setup in ``candles``, labelled with its outcome, oldest first."""
    progress("Preparing history", 0, 0)
    data = enrich(candles)
    lookup = _higher_timeframe_lookup(candles)
    prepared = _prepare(data)
    # A setup older than the outcome horizon can never change, so it is safe to
    # cache. Newer setups are replayed on every run until they settle.
    settled = max(0, len(data) - 1 - settings.prediction_max_horizon_bars)
    cached, through = _load_candidates() if use_cache else (None, None)
    start = 0 if through is None else int(data.datetime.searchsorted(pd.Timestamp(through), side="right"))
    fresh = _replay(data, lookup, prepared, start, settled, progress)
    parts = [part for part in (cached, fresh) if part is not None and not part.empty]
    old = pd.concat(parts, ignore_index=True) if parts else fresh
    if use_cache and settled > 0 and (not fresh.empty or cached is None):
        _save_candidates(old, data.datetime.iloc[settled - 1])
    recent = _replay(data, lookup, prepared, max(start, settled), len(data) - 1, progress)
    parts = [part for part in (old, recent) if not part.empty]
    frame = pd.concat(parts, ignore_index=True) if parts else old
    return (frame.drop_duplicates("time", keep="last").sort_values("time", kind="stable")
            .reset_index(drop=True))


def training_history(recent: pd.DataFrame | None) -> pd.DataFrame | None:
    """The cached multi-year history, topped up with the latest download."""
    if settings.use_synthetic_data:
        return recent  # never mix synthetic candles into the real history cache
    cached = history.load_cache(settings.symbol, settings.interval)
    if recent is None or len(recent) < 2:
        return cached
    # The newest API row can still be forming; keep closed candles only.
    combined = history.merge(cached, recent.iloc[:-1])
    if cached is None or len(combined) > len(cached):
        history.save_cache(combined, settings.symbol, settings.interval)
    return combined


# --------------------------------------------------------------------------
# Model and out-of-sample validation
# --------------------------------------------------------------------------

def _fit(x: np.ndarray, y: np.ndarray) -> dict:
    """L2-regularised logistic regression on standardised features."""
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    normalized = (x - mean) / scale
    weights = np.zeros(normalized.shape[1])
    bias = math.log(y.mean() / (1 - y.mean()))
    for _ in range(1500):
        probabilities = 1 / (1 + np.exp(-np.clip(normalized @ weights + bias, -30, 30)))
        error = probabilities - y
        weights -= .05 * ((normalized.T @ error) / len(y) + .01 * weights)
        bias -= .05 * error.mean()
    return {"mean": mean, "scale": scale, "weights": weights, "bias": float(bias)}


def _probabilities(model: dict, x: np.ndarray) -> np.ndarray:
    score = ((x - np.asarray(model["mean"])) / np.asarray(model["scale"])) @ np.asarray(model["weights"])
    return 1 / (1 + np.exp(-np.clip(score + model["bias"], -30, 30)))


def _trainable(y: np.ndarray) -> bool:
    return len(y) >= settings.prediction_min_samples and 0 < y.mean() < 1


def _auc(p: np.ndarray, y: np.ndarray) -> float | None:
    """Probability a random winner is ranked above a random loser (0.5 = no skill)."""
    positive = y == 1
    wins, losses = int(positive.sum()), int((~positive).sum())
    if not wins or not losses:
        return None
    ranks = pd.Series(p).rank().to_numpy()
    return round(float((ranks[positive].sum() - wins * (wins + 1) / 2) / (wins * losses)), 3)


def _summary(y: np.ndarray, r: np.ndarray) -> dict:
    count = len(r)
    return {"trades": int(count),
            "win_rate": round(float(y.mean()), 3) if count else None,
            "avg_r": round(float(r.mean()), 3) if count else None,
            "total_r": round(float(r.sum()), 1)}


def _eligible(frame: pd.DataFrame) -> pd.Series:
    """Setups the engine would actually reach the model with under current gates."""
    if settings.higher_timeframe_filter:
        bonus = 20 if settings.strategy == "classic" else 0
        return (frame.htf_aligned == 1) & (frame.gate_score + bonus >= settings.setup_score_min)
    return frame.gate_score >= settings.setup_score_min


def _embargo() -> pd.Timedelta:
    # Training labels look this far ahead, so training must stop this far
    # before any data it is scored on, or tomorrow's prices leak into it.
    # Provider timestamps label candle opens; the last outcome candle must
    # finish as well before its label becomes available to training.
    return pd.Timedelta(minutes=(settings.prediction_max_horizon_bars + 1) * interval_minutes())


def _choose_threshold(p: np.ndarray, r: np.ndarray) -> float:
    """Cut with the best average R on validation; 0.0 means "do not filter"."""
    best_threshold, best = 0.0, float(r.mean())
    floor = max(MIN_KEPT_TRADES, MIN_KEPT_SHARE * len(r))
    for threshold in np.quantile(p, np.linspace(0, .8, 17)):
        kept = p >= threshold
        if kept.sum() >= floor and r[kept].mean() > best:
            best_threshold, best = float(threshold), float(r[kept].mean())
    return best_threshold


def _validated_threshold(x, y, r, eligible, times: pd.Series, embargo) -> float:
    """Fit on the older 75%, pick the cut on the newest 25% of what the engine trades."""
    cut = times.iloc[int(len(times) * .75)]
    fit_mask = (times < cut - embargo).to_numpy()
    check = (times >= cut).to_numpy() & eligible
    if not _trainable(y[fit_mask]) or check.sum() < MIN_KEPT_TRADES:
        return 0.0
    return _choose_threshold(_probabilities(_fit(x[fit_mask], y[fit_mask]), x[check]), r[check])


def _fold_masks(times: pd.Series, embargo, folds: int = FOLDS):
    """Expanding-window folds; each training set ends one embargo before its test block."""
    edges = pd.date_range(times.min(), times.max(), periods=folds + 2)
    for k in range(1, folds + 1):
        upper = times < edges[k + 1] if k < folds else times <= edges[k + 1]
        yield (times < edges[k] - embargo).to_numpy(), ((times >= edges[k]) & upper).to_numpy()


def walk_forward(frame: pd.DataFrame, names: list[str]) -> dict:
    """Score the whole train-then-choose-threshold procedure on data it never saw."""
    times = pd.to_datetime(frame["time"], utc=True).reset_index(drop=True)
    x = frame[names].to_numpy(float)
    y, r = frame.win.to_numpy(float), frame.r.to_numpy(float)
    eligible = _eligible(frame).to_numpy()
    embargo = _embargo()
    folds = []
    tested, kept = np.zeros(len(frame), bool), np.zeros(len(frame), bool)
    for number, (train_mask, test_mask) in enumerate(_fold_masks(times, embargo), start=1):
        test_mask &= eligible
        if not _trainable(y[train_mask]) or not test_mask.any():
            continue
        threshold = _validated_threshold(x[train_mask], y[train_mask], r[train_mask],
                                         eligible[train_mask],
                                         times[train_mask].reset_index(drop=True), embargo)
        p = _probabilities(_fit(x[train_mask], y[train_mask]), x[test_mask])
        index = np.flatnonzero(test_mask)
        allowed = index[p >= threshold]
        tested[index], kept[allowed] = True, True
        folds.append({"fold": number, "start": times[index[0]].isoformat(),
                      "end": times[index[-1]].isoformat(), "auc": _auc(p, y[index]),
                      "threshold": round(threshold, 3), "baseline": _summary(y[index], r[index]),
                      "filtered": _summary(y[allowed], r[allowed])})
    baseline, filtered = _summary(y[tested], r[tested]), _summary(y[kept], r[kept])
    helps = bool(folds) and filtered["trades"] > 0 and filtered["avg_r"] > baseline["avg_r"]
    aucs = [fold["auc"] for fold in folds if fold["auc"] is not None]
    return {"filter_helps": helps, "auc": round(float(np.mean(aucs)), 3) if aucs else None,
            "baseline": baseline, "filtered": filtered,
            "losses_avoided": int(((y == 0) & tested & ~kept).sum()),
            "wins_given_up": int(((y == 1) & tested & ~kept).sum()), "folds": folds}


def train(recent: pd.DataFrame | None = None, progress=_quiet) -> dict:
    """Build, validate and save the model. ``recent`` tops up the cached history."""
    candles = training_history(recent)
    historical = (build_dataset(candles, progress, use_cache=not settings.use_synthetic_data)
                  if candles is not None and not candles.empty else pd.DataFrame(columns=_columns()))
    progress("Learning from completed paper trades", 0, 0)
    paper, paper_report = journal.examples(candles)
    frame = journal.merge_examples(historical, paper)
    names = list(feature_names(settings.strategy))
    x, y, r = frame[names].to_numpy(float), frame.win.to_numpy(float), frame.r.to_numpy(float)
    if not _trainable(y):
        raise ValueError(f"Need at least {settings.prediction_min_samples} resolved setups "
                         f"containing wins and losses; found {len(y)}.")
    progress("Testing on unseen periods", 0, 0)
    evaluation = walk_forward(frame, names)
    progress("Fitting final model", 0, 0)
    times = pd.to_datetime(frame["time"], utc=True)
    helps = evaluation.pop("filter_helps")
    # When the procedure did not help out of sample, the model still reports a
    # probability but is never allowed to block an entry.
    threshold = (_validated_threshold(x, y, r, _eligible(frame).to_numpy(), times, _embargo())
                 if helps else 0.0)
    model = _fit(x, y)
    report = {"trained_at": datetime.now(timezone.utc).isoformat(), "strategy": active_name(),
              "data_start": times.iloc[0].isoformat(), "data_end": times.iloc[-1].isoformat(),
              "samples": int(len(y)), "win_rate": round(float(y.mean()), 3),
              "population": _population(),
              "threshold": round(threshold, 3), "filter_helps": helps,
              "used_latest_candles": recent is not None, "out_of_sample": evaluation,
              "historical_samples": int((frame.source == "history").sum()),
              "paper_learning": paper_report,
              "validation_scope": "Strategy setups; excludes portfolio limits, news and pattern-memory gates."}
    artifact = {"version": VERSION, "strategy": active_name(), "fingerprint": fingerprint(),
                "features": names, "mean": model["mean"].tolist(), "scale": model["scale"].tolist(),
                "weights": model["weights"].tolist(), "bias": model["bias"],
                "threshold": threshold, "report": report, "learning_version": LEARNING_VERSION}
    temporary = MODEL_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    os.replace(temporary, MODEL_PATH)  # the engine never reads a half-written file
    return report


# --------------------------------------------------------------------------
# Live use
# --------------------------------------------------------------------------

_loaded: tuple[tuple, dict] | None = None


def _load() -> dict | None:
    """The saved artifact, re-read only when the file changes."""
    global _loaded
    try:
        stat = MODEL_PATH.stat()
    except FileNotFoundError:
        return None
    key = (str(MODEL_PATH), stat.st_mtime_ns, stat.st_size)
    if _loaded is None or _loaded[0] != key:
        _loaded = key, json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    return _loaded[1]


def compatibility(artifact: dict) -> str | None:
    """Why this artifact must not be used with the current settings, if it must not."""
    if artifact.get("version") != VERSION:
        return "The saved model predates the current feature set. Retrain it."
    if artifact.get("strategy") != active_name():
        return (f"The saved model was trained for the {artifact.get('strategy')} strategy, "
                f"but {active_name()} is active. Retrain it.")
    if artifact.get("fingerprint") != fingerprint():
        return "Strategy settings changed since training, so its validation no longer applies. Retrain it."
    if artifact.get("report", {}).get("population") != _population():
        # The threshold was chosen on the setups the old filter setting let through.
        return "The weekly/monthly trend filter setting changed since training. Retrain it."
    return None


def threshold_for(artifact: dict) -> float:
    override = settings.prediction_min_probability
    return float(override) if override is not None else float(artifact["threshold"])


def predict(signal: dict) -> dict:
    if not settings.prediction_model_enabled:
        return {"available": False, "enabled": False, "allowed": True, "reason": "Model filter disabled."}
    artifact = _load()
    if artifact is None:
        return {"available": False, "enabled": True, "allowed": True,
                "reason": "Model is not trained yet; strategy rules remain active."}
    problem = compatibility(artifact)
    if problem:
        return {"available": False, "enabled": True, "allowed": True, "reason": problem}
    row = features(signal)
    x = np.asarray([[row[name] for name in artifact["features"]]], dtype=float)
    probability = float(_probabilities(artifact, x)[0])
    threshold = threshold_for(artifact)
    allowed = probability >= threshold
    return {"available": True, "enabled": True, "allowed": allowed,
            "target_before_stop_probability": round(probability, 3), "threshold": round(threshold, 3),
            "reason": f"{probability:.0%} chance of target before stop; "
                      f"{'at or above' if allowed else 'below'} the {threshold:.0%} threshold."}


def entry_context(signal: dict, feed: str) -> dict:
    """Freeze the exact input vector and non-secret provenance at entry."""
    return {"fingerprint": fingerprint(), "strategy": active_name(), "feed": feed,
            "symbol": settings.symbol, "interval": settings.interval, "features": features(signal)}


class TrainingJob:
    """One background training run at a time, with progress for the dashboard.

    A first run replays years of candles and takes minutes, far longer than an
    HTTP request should block. It runs in a thread so the API keeps answering.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state: dict = {"status": "idle"}
        try:
            self._state = json.loads(TRAINING_STATE_PATH.read_text(encoding="utf-8"))
            if self._state.get("status") == "running":
                self._state.update(status="error", error="Training was interrupted; it can be retried.")
        except (OSError, ValueError):
            pass

    @property
    def running(self) -> bool:
        return self._state.get("status") == "running"

    def start(self, recent: pd.DataFrame | None) -> bool:
        with self._lock:
            if self.running:
                return False
            self._state = {"status": "running", "stage": "Starting", "done": 0, "total": 0,
                           "started_at": datetime.now(timezone.utc).isoformat()}
            try:
                self._persist()
            except OSError:
                self._state.update(status="error", error="Could not persist the training job.")
                raise
        threading.Thread(target=self._run, args=(recent,), daemon=True).start()
        return True

    def _run(self, recent) -> None:
        try:
            train(recent, self._progress)
            self._state = {**self._state, "status": "done", "stage": "Finished",
                           "finished_at": datetime.now(timezone.utc).isoformat()}
        except Exception as exc:
            self._state = {**self._state, "status": "error", "error": str(exc),
                           "finished_at": datetime.now(timezone.utc).isoformat()}
        finally:
            self._persist()

    def _persist(self):
        temporary = TRAINING_STATE_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._state), encoding="utf-8")
        os.replace(temporary, TRAINING_STATE_PATH)

    def _progress(self, stage: str, done: int = 0, total: int = 0) -> None:
        self._state = {**self._state, "stage": stage, "done": done, "total": total}

    def snapshot(self) -> dict:
        return dict(self._state)


job = TrainingJob()


def maybe_retrain(recent: pd.DataFrame | None = None, now=None) -> bool:
    if not settings.prediction_model_enabled or not settings.prediction_auto_retrain or job.running:
        return False
    current = pd.to_datetime(now, utc=True) if now is not None else pd.Timestamp.now(tz="UTC")
    artifact = _load()
    report = artifact.get("report", {}) if artifact else {}
    attempts = [value for value in (job.snapshot().get("started_at"), report.get("trained_at")) if value]
    if attempts and current - max(pd.to_datetime(value, utc=True) for value in attempts) < pd.Timedelta(hours=settings.prediction_retrain_hours):
        return False
    examined = set(report.get("paper_learning", {}).get("examined_trade_ids", []))
    new_count = sum(trade["id"] not in examined for trade in journal.closed_trades())
    if new_count < settings.prediction_retrain_new_trades:
        return False
    return job.start(recent)


def status() -> dict:
    artifact = _load()
    paper_report = artifact.get("report", {}).get("paper_learning", {}) if artifact else {}
    examined = set(paper_report.get("examined_trade_ids", []))
    closed = journal.closed_trades()
    base = {"enabled": settings.prediction_model_enabled, "job": job.snapshot(),
            "learning": {"enabled": settings.prediction_auto_retrain,
                         "closed_trades": len(closed), "trained_trades": paper_report.get("used", 0),
                         "new_trades": sum(t["id"] not in examined for t in closed),
                         "retrain_after": settings.prediction_retrain_new_trades,
                         "cooldown_hours": settings.prediction_retrain_hours}}
    if artifact is None:
        return {**base, "available": False}
    problem = compatibility(artifact)
    if problem:
        # An old or mismatched artifact may lack fields this version reads.
        return {**base, "available": False, "problem": problem}
    return {**base, "available": True, "problem": None,
            "threshold": round(threshold_for(artifact), 3),
            "threshold_source": "env" if settings.prediction_min_probability is not None else "validated",
            "report": artifact.get("report", {})}
