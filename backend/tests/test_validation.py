import numpy as np
import pytest

from app import validation as v


def test_norm_ppf_inverts_the_normal_cdf():
    for p in (0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.999999):
        assert v._norm_cdf(v._norm_ppf(p)) == pytest.approx(p, abs=1e-9)
    assert v._norm_ppf(0.5) == pytest.approx(0.0, abs=1e-12)
    assert v._norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)


def test_norm_ppf_rejects_impossible_probabilities():
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError):
            v._norm_ppf(bad)


def test_sharpe_ratio_handles_degenerate_samples():
    assert v.sharpe_ratio([]) == 0.0
    assert v.sharpe_ratio([1.0]) == 0.0
    assert v.sharpe_ratio([5.0, 5.0, 5.0]) == 0.0  # zero dispersion
    assert v.sharpe_ratio([1.0, -1.0]) == pytest.approx(0.0, abs=1e-12)


def test_sharpe_ratio_matches_mean_over_standard_deviation():
    values = [3.0, -1.0, 2.0, 0.5, -2.5]
    expected = np.mean(values) / np.std(values, ddof=1)
    assert v.sharpe_ratio(values) == pytest.approx(expected)


def test_probabilistic_sharpe_rises_with_sample_length():
    rng = np.random.default_rng(7)
    short = rng.normal(0.1, 1.0, 30)
    long = rng.normal(0.1, 1.0, 3000)
    # Same underlying edge; more evidence must mean more confidence.
    assert v.probabilistic_sharpe_ratio(long) > v.probabilistic_sharpe_ratio(short)
    assert 0.0 <= v.probabilistic_sharpe_ratio(short) <= 1.0


def test_probabilistic_sharpe_is_near_half_for_a_zero_edge_sample():
    rng = np.random.default_rng(11)
    noise = rng.normal(0.0, 1.0, 5000)
    assert v.probabilistic_sharpe_ratio(noise, 0.0) == pytest.approx(0.5, abs=0.25)


def test_expected_maximum_sharpe_grows_with_the_number_of_trials():
    spread = list(np.random.default_rng(3).normal(0, 0.2, 50))
    few = v.expected_maximum_sharpe(spread, n_trials=5)
    many = v.expected_maximum_sharpe(spread, n_trials=500)
    assert 0 < few < many
    # With no dispersion across trials there is nothing to correct for.
    assert v.expected_maximum_sharpe([0.3] * 20) == 0.0
    assert v.expected_maximum_sharpe([0.3]) == 0.0


def test_deflated_sharpe_punishes_a_winner_picked_from_many_trials():
    rng = np.random.default_rng(5)
    returns = rng.normal(0.15, 1.0, 200)
    trials = list(rng.normal(0.0, 0.3, 400))
    one = v.deflated_sharpe_ratio(returns, trials, n_trials=2)
    many = v.deflated_sharpe_ratio(returns, trials, n_trials=400)
    assert many["deflated_sharpe_ratio"] < one["deflated_sharpe_ratio"]
    assert many["expected_max_sharpe"] > one["expected_max_sharpe"]
    assert many["observations"] == 200 and many["n_trials"] == 400


def test_minimum_backtest_length_grows_with_trials_and_shrinks_with_edge():
    assert v.minimum_backtest_length(10) < v.minimum_backtest_length(1000)
    assert v.minimum_backtest_length(100, 2.0) < v.minimum_backtest_length(100, 0.5)
    assert v.minimum_backtest_length(1) == float("inf")
    assert v.minimum_backtest_length(100, 0.0) == float("inf")


def test_make_folds_are_ordered_non_overlapping_and_never_peek_ahead():
    folds = v.make_folds(1000, 5)
    assert len(folds) == 5
    for fold in folds:
        assert fold.train[0] < fold.train[1] <= fold.test[0] < fold.test[1]
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.test[1] <= later.train[0]
    assert folds[-1].test[1] == 1000


def test_make_folds_declines_impossible_requests():
    assert v.make_folds(10, 0) == []
    assert v.make_folds(3, 5) == []


def test_walk_forward_selects_on_train_and_reports_the_test_block():
    # "good" wins in training everywhere, but loses out of sample. A correct
    # harness must still report the out-of-sample loss rather than the fit.
    folds = v.make_folds(1000, 2)
    train_blocks = {f.train for f in folds}

    def evaluate(params, lo, hi):
        if params["name"] == "good":
            training = (lo, hi) in train_blocks
            return {"net": 100.0 if training else -50.0,
                    "returns": [1.0] * 5 if training else [-2.0] * 5}
        return {"net": 10.0, "returns": [0.5] * 5}

    result = v.walk_forward(1000, 2, [{"name": "good"}, {"name": "meh"}], evaluate)
    assert [f.chosen["name"] for f in result.folds] == ["good", "good"]
    summary = result.summary()
    assert summary["folds"] == 2 and summary["folds_scored"] == 2
    assert summary["pooled_trades"] == 10
    # "good" wins every training block but must be scored on its test block.
    assert result.folds[0].test_result["net"] == -50.0


def test_walk_forward_survives_candidates_that_never_trade():
    result = v.walk_forward(600, 3, [{"name": "silent"}], lambda p, lo, hi: None)
    assert result.summary()["folds_scored"] == 0
    assert result.pooled_returns == []
