import json
import math

import numpy as np
import pandas as pd
import pytest

from app import history, prediction, smc
from app.config import settings
from app.market import synthetic_candles
from app.strategy import enrich


def smc_signal(action="BUY"):
    return {"action": action, "model": "smc", "price": 1.1050,
            "candle_time": "2026-01-05T08:15:00+00:00", "setup_score": 110,
            "score_breakdown": {"liquidity_sweep": 20, "structure_shift": 20, "displacement": 20,
                                "point_of_interest": 20, "premium_discount": 10, "killzone": 0,
                                "higher_timeframes": 20},
            "levels_hint": {"stop": 1.1030, "target": 1.1090},
            "indicators": {"rsi": 58, "adx": 24, "momentum_10": .002, "vol_20": .001,
                           "ema20": 1.1040, "ema50": 1.1020, "macd": .001, "macd_signal": .0005,
                           "atr": .002, "session": "LONDON", "weekly_trend": "BULLISH",
                           "monthly_trend": "BULLISH", "target_source": "LIQUIDITY", "poi": "FVG",
                           "equilibrium": 1.1060, "swept_level": 1.1035}}


def classic_signal():
    signal = smc_signal()
    del signal["model"], signal["levels_hint"]
    return signal


def test_smc_features_cover_every_model_input_and_are_finite():
    row = prediction.features(smc_signal())
    names = prediction.feature_names("smc")
    assert set(names) <= set(row)
    assert np.isfinite([row[name] for name in names]).all()
    # The engine's higher-timeframe bonus is excluded so training and live agree.
    assert row["setup_score"] == 90
    assert row["poi_fvg"] == 1.0 and row["killzone"] == 0.0


def test_trend_features_are_signed_by_trade_direction():
    buy, sell = prediction.features(smc_signal("BUY")), prediction.features(smc_signal("SELL"))
    assert buy["ema_gap_atr"] == pytest.approx(-sell["ema_gap_atr"]) and buy["ema_gap_atr"] > 0
    assert buy["htf_aligned"] == 1.0 and sell["htf_aligned"] == 0.0


def test_classic_features_omit_smc_inputs():
    row = prediction.features(classic_signal())
    assert set(prediction.feature_names("classic")) <= set(row)
    assert "stop_atr" not in row


def test_threshold_stays_off_when_higher_probability_means_worse_results():
    p = np.linspace(.1, .6, 400)
    r = np.where(p < .35, 1.5, -1.0)
    assert prediction._choose_threshold(p, r) == 0.0


def test_threshold_is_chosen_when_higher_probability_means_better_results():
    p = np.linspace(.1, .6, 400)
    r = np.where(p > .4, 1.5, -1.0)
    threshold = prediction._choose_threshold(p, r)
    assert threshold > 0
    assert r[p >= threshold].mean() > r.mean()


def test_threshold_never_rests_on_a_handful_of_trades():
    p = np.linspace(0, 1, 400)
    r = np.full(400, -1.0)
    r[-5:] = 3.0
    threshold = prediction._choose_threshold(p, r)
    assert (p >= threshold).sum() >= prediction.MIN_KEPT_SHARE * len(p)


def test_walk_forward_training_always_ends_an_embargo_before_its_test_block():
    times = pd.Series(pd.date_range("2024-01-01", periods=2000, freq="h", tz="UTC"))
    embargo = pd.Timedelta(hours=24)
    folds = list(prediction._fold_masks(times, embargo))
    assert len(folds) == prediction.FOLDS
    for train, test in folds:
        assert train.any() and test.any()
        assert times[train].max() + embargo <= times[test].min()
    blocks = [set(np.flatnonzero(test)) for _, test in folds]
    assert all(not a & b for a, b in zip(blocks, blocks[1:]))


def test_replay_signal_matches_the_live_window_computation(monkeypatch):
    """The sliced SMC context used for fast replay must not change a single signal."""
    monkeypatch.setattr(settings, "strategy", "smc")
    data = enrich(synthetic_candles(600))
    prepared = prediction._prepare(data)
    compared = 0
    for index in range(prediction.SIGNAL_LOOKBACK, len(data) - 1, 5):
        window = data.iloc[max(0, index + 2 - prediction.SIGNAL_LOOKBACK):index + 2]
        assert prediction._signal_at(data, index, prepared) == smc.signal_from(window)
        compared += 1
    assert compared > 50


def test_higher_timeframe_trend_is_only_known_after_its_period_closes():
    candles = synthetic_candles(24 * 4 * 7 * 30, "15min")
    lookup = prediction._higher_timeframe_lookup(candles)
    times, _ = lookup["weekly"]
    assert times, "enough weeks to warm up the indicators"
    # A weekly trend becomes visible on a Monday, when the prior week has closed.
    assert all(time.dayofweek == 0 and time.hour == 0 for time in times)
    assert prediction._trends_at(lookup, times[0] - pd.Timedelta(minutes=15))["weekly"] == "UNAVAILABLE"


@pytest.fixture
def model_path(tmp_path, monkeypatch):
    path = tmp_path / "model.json"
    monkeypatch.setattr(prediction, "MODEL_PATH", path)
    monkeypatch.setattr(settings, "strategy", "smc")
    monkeypatch.setattr(settings, "prediction_model_enabled", True)
    monkeypatch.setattr(settings, "prediction_min_probability", None)
    return path


def write_constant_model(path, probability, threshold, **overrides):
    names = list(prediction.feature_names("smc"))
    artifact = {"version": prediction.VERSION, "strategy": "smc",
                "fingerprint": prediction.fingerprint(), "features": names,
                "mean": [0.0] * len(names), "scale": [1.0] * len(names),
                "weights": [0.0] * len(names), "bias": math.log(probability / (1 - probability)),
                "threshold": threshold, "report": {"population": prediction._population()}}
    artifact.update(overrides)
    path.write_text(json.dumps(artifact), encoding="utf-8")


def test_model_blocks_setups_below_its_validated_threshold(model_path):
    write_constant_model(model_path, .30, .35)
    result = prediction.predict(smc_signal())
    assert result["available"] and not result["allowed"]
    assert result["target_before_stop_probability"] == pytest.approx(.30)
    assert result["threshold"] == pytest.approx(.35)


def test_env_threshold_overrides_the_validated_one(model_path, monkeypatch):
    write_constant_model(model_path, .30, .35)
    monkeypatch.setattr(settings, "prediction_min_probability", .25)
    assert prediction.predict(smc_signal())["allowed"]


def test_a_model_without_proven_benefit_never_blocks(model_path):
    write_constant_model(model_path, .05, 0.0)
    assert prediction.predict(smc_signal())["allowed"]


@pytest.mark.parametrize("overrides", [{"strategy": "classic"}, {"fingerprint": "stale"}, {"version": 1},
                                       {"report": {"population": "a different trend-filter setting"}}])
def test_an_incompatible_model_is_reported_and_never_blocks(model_path, overrides):
    write_constant_model(model_path, .05, .35, **overrides)
    result = prediction.predict(smc_signal())
    assert result["allowed"] and not result["available"]
    assert "Retrain" in result["reason"]
    assert prediction.status()["available"] is False


def test_status_reports_a_version_one_artifact_instead_of_failing(model_path):
    """The previous model file has no threshold; status feeds /api/state and must not raise."""
    model_path.write_text(json.dumps({"version": 1, "features": ["rsi"], "mean": [0], "scale": [1],
                                      "weights": [0], "bias": 0, "report": {"samples": 10}}),
                          encoding="utf-8")
    status = prediction.status()
    assert status["available"] is False and "Retrain" in status["problem"]
    assert prediction.predict(smc_signal())["allowed"]


def test_training_saves_a_validated_model_and_resumes_from_its_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction, "MODEL_PATH", tmp_path / "model.json")
    monkeypatch.setattr(prediction, "CANDIDATE_DIR", tmp_path)
    monkeypatch.setattr(history, "CACHE_DIR", tmp_path)
    for name, value in {"strategy": "classic", "use_synthetic_data": False,
                        "higher_timeframe_filter": False, "prediction_min_samples": 40,
                        "prediction_model_enabled": True, "prediction_min_probability": None}.items():
        monkeypatch.setattr(settings, name, value)
    candles = synthetic_candles(2400)

    full = prediction.build_dataset(candles, use_cache=False)
    prediction.build_dataset(candles.iloc[:1800])  # seeds the setup cache
    resumed = prediction.build_dataset(candles)
    assert len(full) >= 40
    pd.testing.assert_frame_equal(resumed, full, check_dtype=False)

    report = prediction.train(candles)
    artifact = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    assert artifact["strategy"] == "classic" and artifact["fingerprint"] == prediction.fingerprint()
    evidence = report["out_of_sample"]
    assert evidence["filtered"]["trades"] <= evidence["baseline"]["trades"]
    if not report["filter_helps"]:
        assert artifact["threshold"] == 0.0
    assert prediction.predict(classic_signal())["available"]
    # Training tops up the real history cache with closed candles only.
    assert len(history.load_cache(settings.symbol, settings.interval)) == len(candles) - 1
