"""Statistics for comparing prompts: bootstrap intervals and paired tests.

Every test-set score is a mean of per-example scores, so uncertainty comes from
which examples were sampled. Two programs scored on the *same* examples are
compared with paired statistics, which remove the shared per-example difficulty
and are much tighter than comparing two independent intervals.

* ``bootstrap_ci``: percentile bootstrap (Efron and Tibshirani, 1993) of the
  mean, resampling examples with replacement.
* ``paired_bootstrap``: bootstrap of the mean per-example difference ``a - b``
  with a two-sided p-value from the share of resamples on the other side of 0.
* ``mcnemar``: exact McNemar test on discordant pairs for binary scores.

All functions are vectorized with numpy and seeded, so reported intervals are
reproducible.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    """A point estimate with a two-sided confidence interval."""

    mean: float
    low: float
    high: float
    n: int

    def fmt(self, scale: float = 100.0, digits: int = 1) -> str:
        return (
            f"{self.mean * scale:.{digits}f} "
            f"[{self.low * scale:.{digits}f}, {self.high * scale:.{digits}f}]"
        )


@dataclass(frozen=True)
class PairedResult:
    """Mean difference ``a - b`` with its interval and a two-sided p-value."""

    diff: Interval
    p_value: float
    wins: int
    losses: int


def _resample_means(values: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means: np.ndarray = values[idx].mean(axis=1)
    return means


def bootstrap_ci(
    scores: Sequence[float], *, level: float = 0.95, n_boot: int = 10_000, seed: int = 0
) -> Interval:
    """Percentile bootstrap interval for the mean of ``scores``."""

    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        raise ValueError("need at least one score")
    means = _resample_means(values, n_boot, seed)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return Interval(float(values.mean()), float(low), float(high), int(values.size))


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    level: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 0,
) -> PairedResult:
    """Bootstrap the mean paired difference ``a - b`` over shared examples."""

    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("paired scores need the same non-zero length")
    diff = x - y
    means = _resample_means(diff, n_boot, seed)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    observed = float(diff.mean())
    tail = float(np.mean(means <= 0.0)) if observed >= 0 else float(np.mean(means >= 0.0))
    p_value = min(1.0, 2.0 * tail)
    return PairedResult(
        diff=Interval(observed, float(low), float(high), int(diff.size)),
        p_value=p_value,
        wins=int(np.sum(diff > 0)),
        losses=int(np.sum(diff < 0)),
    )


def mcnemar(a: Sequence[float], b: Sequence[float]) -> float:
    """Exact two-sided McNemar p-value for paired binary outcomes."""

    if len(a) != len(b):
        raise ValueError("paired scores need the same length")
    wins = sum(1 for x, y in zip(a, b, strict=True) if x >= 0.5 > y)
    losses = sum(1 for x, y in zip(a, b, strict=True) if y >= 0.5 > x)
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)
