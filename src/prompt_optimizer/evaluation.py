"""An ablation that measures what each optimizer contributes.

Four prompts are compared on a held-out test set: a naive baseline, the
instruction-search result, the few-shot result, and the compound result. The
compound optimizer is expected to match or beat each component, demonstrating
that instruction and demonstration search are complementary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from prompt_optimizer.data import (
    CANDIDATE_INSTRUCTIONS,
    NAIVE_INSTRUCTION,
    TEST,
    TRAIN,
    VALIDATION,
)
from prompt_optimizer.metrics import evaluate, exact_match
from prompt_optimizer.optimizers import (
    BootstrapFewShot,
    CompoundOptimizer,
    InstructionSearch,
)
from prompt_optimizer.predictor import Predictor
from prompt_optimizer.types import Example, Metric, Prompt


@dataclass(frozen=True)
class Row:
    """One row of the ablation: a method name and its held-out score."""

    method: str
    test_score: int
    demonstrations: int

    @property
    def score(self) -> float:
        return self.test_score / 1000


def _test_score(prompt: Prompt, test: Sequence[Example], metric: Metric) -> int:
    return round(evaluate(Predictor(prompt).predict, test, metric) * 1000)


def run_ablation(
    trainset: Sequence[Example] = TRAIN,
    validation: Sequence[Example] = VALIDATION,
    test: Sequence[Example] = TEST,
    candidates: Sequence[str] = CANDIDATE_INSTRUCTIONS,
    metric: Metric = exact_match,
) -> list[Row]:
    """Compile prompts four ways and score each on the held-out test set."""

    base = Prompt(NAIVE_INSTRUCTION)

    instruction = InstructionSearch(candidates, metric).compile(base, validation).prompt
    few_shot = BootstrapFewShot(metric=metric).compile(base, trainset, validation).prompt
    compound = (
        CompoundOptimizer(candidates, metric=metric).compile(base, trainset, validation).prompt
    )

    prompts = {
        "baseline (naive, zero-shot)": base,
        "instruction search": instruction,
        "few-shot bootstrap": few_shot,
        "compound (instruction + few-shot)": compound,
    }
    return [
        Row(name, _test_score(prompt, test, metric), len(prompt.demonstrations))
        for name, prompt in prompts.items()
    ]


def format_table(rows: Sequence[Row]) -> str:
    """Render the ablation rows as an aligned text table."""

    header = f"{'method':<36}{'demonstrations':>16}{'test accuracy':>16}"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(f"{row.method:<36}{row.demonstrations:>16}{row.score:>16.3f}")
    return "\n".join(lines)
