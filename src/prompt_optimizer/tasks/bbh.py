"""BIG-Bench Hard: the 27 tasks of Suzgun et al. (2022).

Each task file holds up to 250 examples (6,511 in total). Every task has a
one-line description at the top of its official chain-of-thought prompt file;
it is prepended to each question so one shared instruction can serve all 27
tasks while demonstrations stay per task (``TaskExample.group``).

Answers come in seven shapes: multiple-choice letters ``(A)``, True/False,
Yes/No, valid/invalid, integers, sorted word lists and Dyck bracket
sequences. ``normalize_answer`` maps a free-form final answer onto the shape of
the gold target before an exact comparison.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from prompt_optimizer.tasks.base import Task, TaskExample, last_answer_line

BBH_TASKS: tuple[str, ...] = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
)

_OPTION = re.compile(r"\(([A-R])\)")
_BARE_OPTION = re.compile(r"^\(?([A-R])\)?(?:[\s.:]|$)")
_SO_ANSWER = re.compile(r"the answer is\s*:?\s*(.+)", re.IGNORECASE)
_WORDS = {"true", "false", "yes", "no", "valid", "invalid"}
_INT = re.compile(r"-?\d+")


def _final_answer(output: str) -> str:
    line = last_answer_line(output)
    if line is None:
        matches = list(_SO_ANSWER.finditer(output))
        if matches:
            line = matches[-1].group(1)
        else:
            lines = output.strip().splitlines()
            line = lines[-1] if lines else ""
    return line.strip().strip("*").strip()


def normalize_answer(raw: str, target: str) -> str:
    """Map a model's final answer onto the canonical form of ``target``."""

    text = raw.strip().strip("`*").strip().rstrip(".").strip()
    gold = target.strip()
    if _OPTION.fullmatch(gold):
        match = _BARE_OPTION.match(text) or _OPTION.search(text)
        return f"({match.group(1)})" if match else text
    if gold.lower() in _WORDS:
        words: list[str] = re.findall(r"[a-zA-Z]+", text.lower())
        for word in words:
            if word in _WORDS:
                return word
        return text.lower()
    if _INT.fullmatch(gold):
        numbers: list[str] = _INT.findall(text.replace(",", ""))
        return numbers[0] if numbers else text
    return " ".join(text.replace(",", " ").split())


def canonical_target(target: str) -> str:
    gold = target.strip()
    if gold.lower() in _WORDS:
        return gold.lower()
    return " ".join(gold.split())


class BBH(Task):
    """Exact match after shape-aware normalization, over all 27 tasks."""

    name = "bbh"
    contract = (
        "End your response with a final line of the form 'Answer: <answer>', where <answer> is "
        "only the final answer in the form the question asks for: an option letter such as (A), "
        "True or False, Yes or No, valid or invalid, a number, or a space-separated sequence."
    )
    seed_instruction = "Answer the question."
    description = (
        "BIG-Bench Hard: 27 reasoning tasks (logic, arithmetic, dates, tracking objects, "
        "language understanding) with short exact answers."
    )
    max_tokens = 512

    def extract(self, output: str) -> str:
        return _final_answer(output)

    def score(self, example: TaskExample, output: str) -> float:
        predicted = normalize_answer(self.extract(output), example.answer)
        return 1.0 if predicted == canonical_target(example.answer) else 0.0


def task_description(cot_prompt: str) -> str:
    """First non-empty line of an official CoT prompt file, skipping the canary."""

    for line in cot_prompt.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("BENCHMARK DATA", "canary", "-----")):
            return stripped
    return ""


def load(root: Path, tasks: tuple[str, ...] = BBH_TASKS) -> list[TaskExample]:
    out: list[TaskExample] = []
    for name in tasks:
        data = json.loads((root / "bbh" / f"{name}.json").read_text(encoding="utf-8"))
        prompt_file = root / "bbh" / "cot-prompts" / f"{name}.txt"
        header = task_description(prompt_file.read_text(encoding="utf-8"))
        for index, item in enumerate(data["examples"]):
            question = f"{header}\n\n{item['input'].strip()}" if header else item["input"].strip()
            out.append(
                TaskExample(
                    id=f"bbh-{name}-{index}",
                    question=question,
                    answer=str(item["target"]).strip(),
                    group=name,
                )
            )
    return out
