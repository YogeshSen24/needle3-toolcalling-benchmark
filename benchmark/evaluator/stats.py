"""Bootstrap statistics.

Point estimates on 5-case cells are close to meaningless, so every rate is
reported with an interval. The crossover rule in DESIGN.md §C.4 depends on
these intervals, not on eyeballing a line chart.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float
    n: int

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.mean, self.low, self.high)

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.low:.3f}, {self.high:.3f}] (n={self.n})"


def bootstrap_ci(values, n_resamples: int = 10000, alpha: float = 0.05,
                 seed: int = 20260920) -> Interval:
    data = [1.0 if v is True else 0.0 if v is False else float(v) for v in values]
    n = len(data)
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    mean = sum(data) / n
    if n == 1:
        return Interval(mean, mean, mean, 1)
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += data[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return Interval(mean, lo, hi, n)


def paired_diff_ci(a_by_case: dict, b_by_case: dict, n_resamples: int = 10000,
                   alpha: float = 0.05, seed: int = 20260920) -> Interval:
    """Paired bootstrap of (a - b) over the cases both models were given.

    Paired resampling because the two models see identical cases; treating the
    samples as independent would inflate the interval.
    """
    shared = sorted(set(a_by_case) & set(b_by_case))
    diffs = [float(a_by_case[k]) - float(b_by_case[k]) for k in shared]
    return bootstrap_ci(diffs, n_resamples=n_resamples, alpha=alpha, seed=seed)


def excludes_zero(interval: Interval) -> bool:
    return interval.low > 0 or interval.high < 0
