"""Per-instance Pareto fronts and GEPA's candidate selection.

GEPA (Agrawal et al., 2025) keeps every accepted candidate together with its
score on each validation instance. Instead of always mutating the single best
candidate (which stalls in a local optimum), it samples a parent from the
candidates that are best on at least one instance:

1. For each instance ``i`` the front ``P*_i`` is the set of candidates that
   reach the highest score on ``i``.
2. Candidates are visited in ascending order of aggregate score; a candidate
   is *dominated* when every front that contains it also contains another
   surviving candidate. Dominated candidates are removed one at a time until
   none is left.
3. Each survivor is weighted by the number of fronts it belongs to, and the
   parent is drawn with probability proportional to that weight.

A candidate that solves a few instances nobody else solves therefore keeps a
chance to be refined, which is what lets GEPA combine complementary lessons.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence


def instance_fronts(scores: Mapping[str, Sequence[float]]) -> list[set[str]]:
    """For each instance, the candidates that reach the best score on it."""

    if not scores:
        return []
    lengths = {len(row) for row in scores.values()}
    if len(lengths) != 1:
        raise ValueError("every candidate needs a score for every instance")
    (n,) = lengths
    fronts: list[set[str]] = []
    for i in range(n):
        best = max(row[i] for row in scores.values())
        fronts.append({name for name, row in scores.items() if row[i] == best})
    return fronts


def _dominated(candidate: str, others: set[str], fronts: Sequence[set[str]]) -> bool:
    containing = [front for front in fronts if candidate in front]
    if not containing:
        return True
    return all(any(other in front for other in others) for front in containing)


def remove_dominated(scores: Mapping[str, Sequence[float]]) -> dict[str, int]:
    """Surviving candidates and the number of instance fronts each belongs to."""

    fronts = instance_fronts(scores)
    order = sorted(scores, key=lambda name: (sum(scores[name]), name))
    alive = set(order)
    changed = True
    while changed:
        changed = False
        for name in order:
            if name in alive and len(alive) > 1 and _dominated(name, alive - {name}, fronts):
                alive.discard(name)
                changed = True
                break
    return {name: sum(1 for front in fronts if name in front) for name in sorted(alive)}


def select_candidate(scores: Mapping[str, Sequence[float]], rng: random.Random) -> str:
    """Sample a parent with probability proportional to its front membership."""

    weights = remove_dominated(scores)
    names = list(weights)
    total = sum(weights.values())
    if total == 0:
        return rng.choice(names)
    return rng.choices(names, weights=[weights[n] for n in names], k=1)[0]
