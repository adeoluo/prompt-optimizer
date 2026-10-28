"""A micro-benchmark for the optimizers.

Measures compile time, the achieved validation score, and the number of
selected demonstrations for each optimizer, so the README can quote concrete
numbers. Deterministic and offline; run with:

    uv run python benchmarks/benchmark.py
"""

from __future__ import annotations

import time

from prompt_optimizer.data import CANDIDATE_INSTRUCTIONS, NAIVE_INSTRUCTION, TRAIN, VALIDATION
from prompt_optimizer.optimizers import (
    BootstrapFewShot,
    CompoundOptimizer,
    InstructionSearch,
    OptimizationResult,
)
from prompt_optimizer.types import Prompt


def main() -> None:
    base = Prompt(NAIVE_INSTRUCTION)
    print(f"Train: {len(TRAIN)} | Validation: {len(VALIDATION)}\n")
    print(f"{'optimizer':<22}{'compile (ms)':>14}{'score':>9}{'demos':>8}")
    print("-" * 53)

    def report(name: str, result: OptimizationResult, elapsed_ms: float) -> None:
        demos = len(result.prompt.demonstrations)
        print(f"{name:<22}{elapsed_ms:>14.1f}{result.score:>9.3f}{demos:>8}")

    start = time.perf_counter()
    instruction = InstructionSearch(CANDIDATE_INSTRUCTIONS).compile(base, VALIDATION)
    report("instruction search", instruction, (time.perf_counter() - start) * 1000)

    start = time.perf_counter()
    few_shot = BootstrapFewShot().compile(base, TRAIN, VALIDATION)
    report("few-shot bootstrap", few_shot, (time.perf_counter() - start) * 1000)

    start = time.perf_counter()
    compound = CompoundOptimizer(CANDIDATE_INSTRUCTIONS).compile(base, TRAIN, VALIDATION)
    report("compound", compound, (time.perf_counter() - start) * 1000)


if __name__ == "__main__":
    main()
