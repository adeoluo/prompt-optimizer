"""Tree-structured Parzen Estimator over categorical choices (Bergstra et al., 2011).

MIPROv2 searches the joint space (instruction candidate, demo-set candidate)
with Bayesian optimization; DSPy uses Optuna's TPE sampler for it. This module
implements the categorical, independent form of TPE from scratch so the search
is inspectable and has no Optuna dependency.

For a maximized objective with observations ``(x_t, y_t)``:

1. Split the observations into a *good* set (the top ``gamma`` fraction by
   ``y``, at least one) and a *bad* set (the rest).
2. For each categorical parameter with ``K`` choices, fit smoothed frequency
   estimates

       l(x) = (prior / K + #good with choice x) / (prior + n_good)
       g(x) = (prior / K + #bad  with choice x) / (prior + n_bad)

3. Draw ``n_candidates`` samples from ``l`` and keep the one maximizing
   ``l(x) / g(x)``, which is proportional to expected improvement under the
   TPE model (Bergstra et al., 2011, eq. 4).

Parameters are modelled independently (as in Optuna's default sampler). The
first ``n_startup`` suggestions are uniform random.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass
class TPESampler:
    """Suggest categorical parameter settings that are likely to score well."""

    space: Mapping[str, int]
    seed: int = 0
    n_startup: int = 4
    gamma: float = 0.25
    prior_weight: float = 1.0
    n_candidates: int = 24
    observations: list[tuple[dict[str, int], float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.space or any(k < 1 for k in self.space.values()):
            raise ValueError("every parameter needs at least one choice")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must be in (0, 1)")
        self._rng = random.Random(self.seed)

    def split(self) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
        ranked = sorted(self.observations, key=lambda obs: -obs[1])
        n_good = max(1, math.ceil(self.gamma * len(ranked)))
        good = [params for params, _ in ranked[:n_good]]
        bad = [params for params, _ in ranked[n_good:]]
        return good, bad

    def density(self, group: Sequence[Mapping[str, int]], name: str) -> list[float]:
        """Smoothed categorical density of parameter ``name`` within ``group``."""

        k = self.space[name]
        counts = [self.prior_weight / k] * k
        for params in group:
            counts[params[name]] += 1.0
        total = sum(counts)
        return [c / total for c in counts]

    def suggest(self) -> dict[str, int]:
        if len(self.observations) < self.n_startup:
            return {name: self._rng.randrange(k) for name, k in self.space.items()}
        good, bad = self.split()
        out: dict[str, int] = {}
        for name, k in self.space.items():
            l_density = self.density(good, name)
            g_density = self.density(bad, name)
            draws = self._rng.choices(range(k), weights=l_density, k=self.n_candidates)
            out[name] = max(draws, key=lambda x: (l_density[x] / g_density[x], -x))
        return out

    def observe(self, params: Mapping[str, int], value: float) -> None:
        for name, choice in params.items():
            if name not in self.space or not 0 <= choice < self.space[name]:
                raise ValueError(f"{name}={choice} is outside the search space")
        self.observations.append((dict(params), float(value)))
