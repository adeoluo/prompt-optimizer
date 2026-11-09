"""GSM8K: grade school math word problems (Cobbe et al., 2021).

The test split has 1,319 problems and is scored in full. Training and
validation examples are drawn, disjointly and with a fixed seed, from the
7,473-problem training split. Gold solutions end with ``#### <answer>`` and
carry calculator annotations ``<<48/2=24>>``, which are removed when a
solution is used as a labeled demonstration.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from prompt_optimizer.tasks.base import Task, TaskExample, last_answer_line

_CALC = re.compile(r"<<[^>]*>>")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def normalize_number(text: str) -> str:
    """Canonical string for a numeric answer: no commas, no trailing zeros."""

    cleaned = text.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        value = float(cleaned)
    except ValueError:
        return cleaned
    if value == int(value):
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def extract_number(output: str) -> str:
    """Last number after the final ``Answer:`` marker, else the last number at all."""

    line = last_answer_line(output)
    for source in (line, output):
        if source is None:
            continue
        numbers = _NUMBER.findall(source)
        if numbers:
            return normalize_number(numbers[-1] if source is output else numbers[0])
    return ""


class GSM8K(Task):
    """Exact numeric match on the final answer."""

    name = "gsm8k"
    contract = (
        "Show your reasoning in a few short steps (no more than about 120 words), then end "
        "with a final line of the form 'Answer: <number>'."
    )
    seed_instruction = "Solve the math word problem."
    description = (
        "Grade school math word problems that need two to eight steps of basic arithmetic; "
        "the answer is a single number."
    )
    max_tokens = 384

    def extract(self, output: str) -> str:
        return extract_number(output)

    def score(self, example: TaskExample, output: str) -> float:
        predicted = self.extract(output)
        if not predicted:
            return 0.0
        try:
            return 1.0 if abs(float(predicted) - float(example.answer)) < 1e-6 else 0.0
        except ValueError:
            return 1.0 if predicted == example.answer else 0.0


def parse_record(record: dict[str, str], index: int, split: str) -> TaskExample:
    solution, _, final = record["answer"].rpartition("####")
    rationale = _CALC.sub("", solution).strip()
    return TaskExample(
        id=f"gsm8k-{split}-{index}",
        question=record["question"].strip(),
        answer=normalize_number(final),
        rationale=rationale,
    )


def load(root: Path, split: str) -> list[TaskExample]:
    path = root / "gsm8k" / f"{split}.jsonl"
    with open(path, encoding="utf-8") as handle:
        return [parse_record(json.loads(line), i, split) for i, line in enumerate(handle)]
