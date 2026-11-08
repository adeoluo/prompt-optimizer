"""Task abstraction: examples, a fixed output contract, scoring and feedback.

A task owns everything that must stay constant while the prompt is optimized:
how an example is shown to the model, the output contract appended to every
system message, how the final answer is extracted, how it is scored, and the
natural-language feedback that feedback-driven optimizers (TextGrad, GEPA) read.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from prompt_optimizer.program import Demo

_EMPTY: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True)
class TaskExample:
    """One benchmark item.

    ``question`` is the full user message. ``answer`` is the canonical gold
    answer. ``rationale`` is a gold worked solution when the dataset has one
    (GSM8K, MATH); labeled demonstrations use it. ``group`` names a subtask
    (BIG-Bench Hard) so demonstrations can be kept per subtask.
    """

    id: str
    question: str
    answer: str
    rationale: str = ""
    group: str = ""
    meta: Mapping[str, Any] = field(default=_EMPTY, hash=False, compare=False)


_ANSWER_LINE = re.compile(r"answer\s*(?:is)?\s*[:=]\s*(.+)", re.IGNORECASE)


def last_answer_line(text: str) -> str | None:
    """Content after the last ``Answer:`` marker, or None when there is none."""

    matches = list(_ANSWER_LINE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


class Task(ABC):
    """Base class for a benchmark task."""

    name: str = "task"
    contract: str = "End your response with a final line of the form 'Answer: <answer>'."
    seed_instruction: str = "Answer the question."
    description: str = ""
    max_tokens: int = 512

    @abstractmethod
    def extract(self, output: str) -> str:
        """Pull the final answer out of a model response."""

    @abstractmethod
    def score(self, example: TaskExample, output: str) -> float:
        """Score in [0, 1] for one response."""

    def metrics(self, example: TaskExample, output: str) -> dict[str, float]:
        """Extra per-example metrics reported next to the score (none by default)."""

        return {}

    def feedback(self, example: TaskExample, output: str) -> str:
        """Natural-language feedback for reflective optimizers."""

        predicted = self.extract(output) or "(no answer found)"
        if self.score(example, output) >= 1.0:
            return f"Correct. The final answer {predicted!r} matches the reference."
        return (
            f"Incorrect. The final answer was {predicted!r} but the reference answer is "
            f"{example.answer!r}."
        )

    def gold_output(self, example: TaskExample) -> str:
        """A labeled assistant turn built from the gold rationale and answer."""

        body = example.rationale.strip()
        final = f"Answer: {example.answer}"
        return f"{body}\n{final}" if body else final

    def gold_demo(self, example: TaskExample) -> Demo:
        return Demo(example.question, self.gold_output(example), example.id, example.group)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
