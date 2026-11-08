"""A synthetic arithmetic task used with the fake model in tests and demos."""

from __future__ import annotations

import random
import re

from prompt_optimizer.tasks.base import Task, TaskExample, last_answer_line

_INT = re.compile(r"-?\d+")


class ArithmeticTask(Task):
    """Integer arithmetic questions such as ``What is 17 + 25?``."""

    name = "arithmetic"
    contract = "End your response with a final line of the form 'Answer: <integer>'."
    seed_instruction = "Answer the arithmetic question."
    description = "Short integer arithmetic questions."
    max_tokens = 64

    def extract(self, output: str) -> str:
        line = last_answer_line(output)
        source = line if line is not None else output
        numbers = _INT.findall(source)
        return numbers[-1] if numbers else ""

    def score(self, example: TaskExample, output: str) -> float:
        return 1.0 if self.extract(output) == example.answer else 0.0

    @staticmethod
    def make(n: int, seed: int = 0, prefix: str = "q") -> list[TaskExample]:
        rng = random.Random(seed)
        out: list[TaskExample] = []
        for index in range(n):
            left, right = rng.randint(2, 99), rng.randint(2, 99)
            op = rng.choice("+-*")
            value = left + right if op == "+" else left - right if op == "-" else left * right
            out.append(
                TaskExample(
                    id=f"{prefix}{seed}-{index}",
                    question=f"What is {left} {op} {right}?",
                    answer=str(value),
                    rationale=f"Compute {left} {op} {right} = {value}.",
                )
            )
        return out
