"""A large-workload scale benchmark for prompt compilation.

Where ``benchmark.py`` compiles one prompt with each optimizer and ``stress.py``
drives the prediction service under concurrent load, this script measures the
core operation - :meth:`CompoundOptimizer.compile`, which searches over
instructions and demonstrations to improve a prompt - on progressively larger
batches of independent synthetic optimization tasks. It reports wall-clock time,
throughput (prompts optimized per second), peak memory (via the standard-library
:mod:`tracemalloc`), and the mean held-out improvement the optimizer delivers
over the naive baseline prompt.

Each task's intent-classification data is generated deterministically from a
seed (nothing is downloaded), so the run is reproducible and fully offline, and
every language-model call goes through the deterministic offline simulator. Run
with:

    uv run python benchmarks/scale.py
"""

from __future__ import annotations

import random
import time
import tracemalloc
from dataclasses import dataclass

from prompt_optimizer.data import CANDIDATE_INSTRUCTIONS, NAIVE_INSTRUCTION
from prompt_optimizer.metrics import evaluate, exact_match
from prompt_optimizer.optimizers import CompoundOptimizer
from prompt_optimizer.predictor import Predictor
from prompt_optimizer.types import Example, Prompt

WORKLOAD_SIZES: tuple[int, ...] = (100, 200, 400, 600)

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "refund": ("refund", "money", "charged", "overcharged", "payment", "back"),
    "shipping": ("package", "shipping", "delivery", "parcel", "delayed", "tracking", "late"),
    "cancel": ("cancel", "subscription", "plan", "recurring", "stop", "order"),
}
_LABELS: tuple[str, ...] = tuple(_KEYWORDS)


def _message(rng: random.Random, label: str) -> str:
    keywords = _KEYWORDS[label]
    count = min(rng.randint(3, 5), len(keywords))
    return " ".join(rng.sample(keywords, count))


def _make_task(seed: int) -> tuple[tuple[Example, ...], tuple[Example, ...], tuple[Example, ...]]:
    """Build one deterministic, keyword-separable optimization task."""
    rng = random.Random(seed)

    def example() -> Example:
        label = _LABELS[rng.randrange(len(_LABELS))]
        return Example(_message(rng, label), label)

    train = tuple(example() for _ in range(9))
    validation = tuple(example() for _ in range(4))
    test = tuple(example() for _ in range(6))
    return train, validation, test


@dataclass(frozen=True, slots=True)
class ScaleRow:
    """One measured row of the scale sweep."""

    size: int
    seconds: float
    throughput: float
    peak_mib: float
    mean_improvement: float


def run_workload(size: int) -> ScaleRow:
    """Optimize ``size`` independent synthetic prompts with the compound optimizer.

    The quality metric is the mean held-out improvement: the optimized prompt's
    test accuracy minus the naive baseline's, averaged over the batch.
    """
    optimizer = CompoundOptimizer(CANDIDATE_INSTRUCTIONS)
    base = Prompt(NAIVE_INSTRUCTION)
    improvement_total = 0.0
    tracemalloc.start()
    start = time.perf_counter()
    for seed in range(size):
        train, validation, test = _make_task(seed)
        result = optimizer.compile(base, train, validation)
        baseline = evaluate(Predictor(base).predict, test, exact_match)
        optimized = evaluate(Predictor(result.prompt).predict, test, exact_match)
        improvement_total += optimized - baseline
    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    throughput = size / elapsed if elapsed > 0 else float("inf")
    return ScaleRow(size, elapsed, throughput, peak / (1024 * 1024), improvement_total / size)


def main() -> None:
    print("Prompt Optimizer scale benchmark")
    print("Data is SYNTHETIC: intent-classification tasks generated deterministically.")
    print("Language-model calls use the deterministic offline simulator.\n")
    header = (
        f"{'prompts':>8}  {'seconds':>8}  {'prompts/sec':>12}  "
        f"{'peak MiB':>9}  {'mean improvement':>17}"
    )
    print(header)
    print("-" * len(header))
    for size in WORKLOAD_SIZES:
        row = run_workload(size)
        print(
            f"{row.size:>8,}  {row.seconds:>8.3f}  {row.throughput:>12,.0f}  "
            f"{row.peak_mib:>9.2f}  {row.mean_improvement:>+17.3f}"
        )


if __name__ == "__main__":
    main()
