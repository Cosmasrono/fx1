"""Statistics for deciding whether a backtest result means anything.

A parameter sweep is a multiple-comparisons experiment: try enough variants of
a strategy on one price series and the best one looks profitable whether or not
any edge exists. Bailey and Lopez de Prado formalised the correction, and this
module implements the parts that matter here.

    - ``probabilistic_sharpe_ratio`` -- confidence that a true Sharpe exceeds a
      benchmark, adjusting for sample length, skew and fat tails.
    - ``expected_maximum_sharpe`` -- the Sharpe the *best of N* random trials
      would show by luck alone. This is the benchmark a sweep winner must beat.
    - ``deflated_sharpe_ratio`` -- the first evaluated against the second.
    - ``minimum_backtest_length`` -- how much data N trials actually needs.
    - ``walk_forward`` -- fit parameters on one block, score them on the next,
      never letting a fold see its own future.

References: Bailey & Lopez de Prado, "The Deflated Sharpe Ratio" (2014) and
Bailey, Borwein, Lopez de Prado & Zhu, "The Probability of Backtest
Overfitting" (2014).
"""
from dataclasses import dataclass, field
from math import e, erf, exp, log, pi, sqrt

import numpy as np

EULER_MASCHERONI = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation).

    Written out rather than pulled from scipy: the project depends only on
    numpy and pandas, and adding scipy for one function is not worth it.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between 0 and 1")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    low, high = 0.02425, 1 - 0.02425
    if p < low:
        q = sqrt(-2 * log(p))
        value = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= high:
        q, r = p - 0.5, (p - 0.5) ** 2
        value = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = sqrt(-2 * log(1 - p))
        value = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    # One Halley step; the raw approximation is only good to ~1e-9 and this is
    # cheap insurance at the tails, where N-trial corrections actually live.
    error = _norm_cdf(value) - p
    step = error * sqrt(2 * pi) * exp(value * value / 2)
    return value - step / (1 + value * step / 2)


def sharpe_ratio(returns) -> float:
    """Per-trade Sharpe. Not annualised: trade counts here are not calendar-paced."""
    values = np.asarray(returns, dtype=float)
    if len(values) < 2:
        return 0.0
    deviation = values.std(ddof=1)
    return 0.0 if deviation == 0 else float(values.mean() / deviation)


def _moments(values: np.ndarray) -> tuple[float, float]:
    """Sample skewness and (non-excess) kurtosis."""
    deviation = values.std(ddof=1)
    if deviation == 0 or len(values) < 3:
        return 0.0, 3.0
    centred = values - values.mean()
    skew = float(np.mean(centred ** 3) / deviation ** 3)
    kurtosis = float(np.mean(centred ** 4) / deviation ** 4)
    return skew, kurtosis


def probabilistic_sharpe_ratio(returns, benchmark: float = 0.0) -> float:
    """Probability the true Sharpe exceeds ``benchmark``, given this sample.

    Fat tails and negative skew make an observed Sharpe less trustworthy, and
    both are normal for stop-and-target trading, where a few large losses sit
    against many small wins.
    """
    values = np.asarray(returns, dtype=float)
    if len(values) < 3:
        return 0.0
    observed = sharpe_ratio(values)
    skew, kurtosis = _moments(values)
    denominator = 1.0 - skew * observed + (kurtosis - 1.0) / 4.0 * observed ** 2
    if denominator <= 0:
        return 0.0
    statistic = (observed - benchmark) * sqrt(len(values) - 1) / sqrt(denominator)
    return _norm_cdf(statistic)


def expected_maximum_sharpe(trial_sharpes, n_trials: int | None = None) -> float:
    """Sharpe the best of N independent trials shows by chance alone.

    Uses the dispersion of the sweep's own Sharpe ratios as the null scale, so
    a sweep over parameters that barely change anything is penalised less than
    one that ranges wildly.
    """
    values = np.asarray(trial_sharpes, dtype=float)
    trials = int(n_trials or len(values))
    if trials < 2:
        return 0.0
    spread = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    # Identical trial Sharpes leave a floating-point residue rather than an
    # exact zero, and a sweep with no dispersion has nothing to deflate.
    if spread < 1e-12:
        return 0.0
    gamma = EULER_MASCHERONI
    return spread * ((1 - gamma) * _norm_ppf(1 - 1.0 / trials)
                     + gamma * _norm_ppf(1 - 1.0 / (trials * e)))


def deflated_sharpe_ratio(returns, trial_sharpes, n_trials: int | None = None) -> dict:
    """Probability the winning strategy beats what N trials produce by luck."""
    benchmark = expected_maximum_sharpe(trial_sharpes, n_trials)
    return {
        "observed_sharpe": round(sharpe_ratio(returns), 4),
        "expected_max_sharpe": round(benchmark, 4),
        "n_trials": int(n_trials or len(np.asarray(trial_sharpes))),
        "observations": int(len(np.asarray(returns))),
        "deflated_sharpe_ratio": round(probabilistic_sharpe_ratio(returns, benchmark), 4),
        "psr_vs_zero": round(probabilistic_sharpe_ratio(returns, 0.0), 4),
    }


def minimum_backtest_length(n_trials: int, target_sharpe: float = 1.0) -> float:
    """Observations needed before a best-of-N result could be credible.

    Inverts the expected-maximum relation: if the best of ``n_trials`` shows
    ``target_sharpe`` per observation, this is roughly how many observations
    are required for that to mean more than selection luck.
    """
    if n_trials < 2 or target_sharpe <= 0:
        return float("inf")
    gamma = EULER_MASCHERONI
    expected = ((1 - gamma) * _norm_ppf(1 - 1.0 / n_trials)
                + gamma * _norm_ppf(1 - 1.0 / (n_trials * e)))
    return float((expected / target_sharpe) ** 2) + 1.0


@dataclass
class Fold:
    index: int
    train: tuple[int, int]
    test: tuple[int, int]
    chosen: dict | None = None
    train_result: dict | None = None
    test_result: dict | None = None


@dataclass
class WalkForward:
    folds: list[Fold] = field(default_factory=list)
    pooled_returns: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        scored = [f for f in self.folds if f.test_result]
        positive = sum(1 for f in scored if f.test_result.get("net", 0) > 0)
        return {
            "folds": len(self.folds),
            "folds_scored": len(scored),
            "folds_profitable_out_of_sample": positive,
            "pooled_trades": len(self.pooled_returns),
            "pooled_net": round(float(np.sum(self.pooled_returns)), 2) if self.pooled_returns else 0.0,
            "pooled_sharpe": round(sharpe_ratio(self.pooled_returns), 4),
        }


def make_folds(length: int, n_folds: int, train_fraction: float = 0.6) -> list[Fold]:
    """Contiguous, non-overlapping test blocks, each trained on the block before it.

    Blocks stay in time order and a fold is never trained on data that comes
    after its own test window, which is the whole point of walk-forward.
    """
    if n_folds < 1 or length < n_folds * 2:
        return []
    block = length // n_folds
    train_size = max(1, int(block * train_fraction))
    folds = []
    for i in range(n_folds):
        start = i * block
        end = start + block if i < n_folds - 1 else length
        split = start + train_size
        if split >= end:
            continue
        folds.append(Fold(index=i, train=(start, split), test=(split, end)))
    return folds


def walk_forward(length: int, n_folds: int, candidates: list[dict],
                 evaluate, score=None, train_fraction: float = 0.6) -> WalkForward:
    """Pick the best candidate on each training block, then score it forward.

    ``evaluate(params, lo, hi)`` runs one configuration over a bar range and
    returns a result dict (or None). ``score`` ranks training results; it
    defaults to net profit. Parameters are chosen only from the training block,
    so the pooled test returns are genuinely out of sample.
    """
    score = score or (lambda r: r.get("net", 0.0))
    result = WalkForward(folds=make_folds(length, n_folds, train_fraction))
    for fold in result.folds:
        scored = []
        for params in candidates:
            trained = evaluate(params, *fold.train)
            if trained:
                scored.append((score(trained), params, trained))
        if not scored:
            continue
        _, best_params, best_train = max(scored, key=lambda item: item[0])
        fold.chosen, fold.train_result = best_params, best_train
        tested = evaluate(best_params, *fold.test)
        if tested:
            fold.test_result = tested
            result.pooled_returns.extend(tested.get("returns", []))
    return result
