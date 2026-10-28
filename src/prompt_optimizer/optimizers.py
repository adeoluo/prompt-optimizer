"""Optimizers that compile a prompt against a metric.

Each optimizer takes a starting prompt, a training set it may draw
demonstrations from, and a validation set it scores candidates on, and returns
an improved prompt. The search is deterministic: given the same inputs it
always returns the same prompt.

Three strategies are provided, mirroring how prompt programs are optimized in
practice:

* ``BootstrapFewShot`` performs greedy forward selection of demonstrations.
* ``InstructionSearch`` chooses the best instruction from a candidate set.
* ``CompoundOptimizer`` alternates the two by coordinate ascent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from prompt_optimizer.metrics import evaluate, exact_match
from prompt_optimizer.predictor import Predictor
from prompt_optimizer.types import Example, Metric, Prompt


@dataclass(frozen=True)
class OptimizationResult:
    """The outcome of an optimization run."""

    prompt: Prompt
    score: float
    trace: tuple[str, ...]


def _score(prompt: Prompt, validation: Sequence[Example], metric: Metric) -> float:
    return evaluate(Predictor(prompt).predict, validation, metric)


class BootstrapFewShot:
    """Greedily select demonstrations that maximize the validation metric.

    Starting from no demonstrations, repeatedly add the training example whose
    inclusion most improves the validation score, stopping when no addition
    helps or the budget ``max_demonstrations`` is reached. This is the engine
    behind few-shot prompt compilation: it finds a small, high-value set of
    in-context examples rather than using them all.
    """

    def __init__(self, max_demonstrations: int = 4, metric: Metric = exact_match) -> None:
        if max_demonstrations < 1:
            raise ValueError("max_demonstrations must be at least 1")
        self.max_demonstrations = max_demonstrations
        self.metric = metric

    def compile(
        self,
        prompt: Prompt,
        trainset: Sequence[Example],
        validation: Sequence[Example],
    ) -> OptimizationResult:
        chosen: list[Example] = []
        remaining = list(trainset)
        best_score = _score(prompt, validation, self.metric)
        trace = [f"start score={best_score:.3f}"]

        while remaining and len(chosen) < self.max_demonstrations:
            best_candidate: Example | None = None
            best_candidate_score = best_score
            for candidate in remaining:
                trial = prompt.with_demonstrations((*chosen, candidate))
                trial_score = _score(trial, validation, self.metric)
                if trial_score > best_candidate_score:
                    best_candidate_score = trial_score
                    best_candidate = candidate
            if best_candidate is None:
                break
            chosen.append(best_candidate)
            remaining.remove(best_candidate)
            best_score = best_candidate_score
            trace.append(f"add demo -> score={best_score:.3f}")

        optimized = prompt.with_demonstrations(tuple(chosen))
        return OptimizationResult(optimized, best_score, tuple(trace))


class InstructionSearch:
    """Choose the instruction, from a candidate set, with the best metric."""

    def __init__(self, candidates: Sequence[str], metric: Metric = exact_match) -> None:
        if not candidates:
            raise ValueError("at least one candidate instruction is required")
        self.candidates = tuple(candidates)
        self.metric = metric

    def compile(self, prompt: Prompt, validation: Sequence[Example]) -> OptimizationResult:
        best_prompt = prompt
        best_score = _score(prompt, validation, self.metric)
        trace = [f"start score={best_score:.3f}"]
        for instruction in self.candidates:
            trial = prompt.with_instruction(instruction)
            trial_score = _score(trial, validation, self.metric)
            if trial_score > best_score:
                best_score = trial_score
                best_prompt = trial
                trace.append(f"select instruction -> score={best_score:.3f}")
        return OptimizationResult(best_prompt, best_score, tuple(trace))


class CompoundOptimizer:
    """Alternate instruction search and few-shot selection by coordinate ascent."""

    def __init__(
        self,
        candidates: Sequence[str],
        max_demonstrations: int = 4,
        rounds: int = 2,
        metric: Metric = exact_match,
    ) -> None:
        if rounds < 1:
            raise ValueError("rounds must be at least 1")
        self.instruction_search = InstructionSearch(candidates, metric)
        self.few_shot = BootstrapFewShot(max_demonstrations, metric)
        self.rounds = rounds
        self.metric = metric

    def compile(
        self,
        prompt: Prompt,
        trainset: Sequence[Example],
        validation: Sequence[Example],
    ) -> OptimizationResult:
        current = prompt
        best_score = _score(current, validation, self.metric)
        trace = [f"start score={best_score:.3f}"]
        for round_index in range(self.rounds):
            instruction_result = self.instruction_search.compile(current, validation)
            current = instruction_result.prompt
            few_shot_result = self.few_shot.compile(current, trainset, validation)
            current = few_shot_result.prompt
            best_score = few_shot_result.score
            trace.append(f"round {round_index + 1} -> score={best_score:.3f}")
            if best_score >= 1.0:
                break
        return OptimizationResult(current, best_score, tuple(trace))
