"""MATH-500: the 500-problem MATH test subset of Lightman et al. (2023).

The test split is the 500 problems released with PRM800K (``math_splits``);
training and validation problems are drawn with a fixed seed from the
12,000-problem training split of the same release, so no test problem can be
used as a demonstration.

Answers are LaTeX. Scoring reimplements the string normalization of Hendrycks
et al. (2021) (fractions, ``\\left``/``\\right``, units, degrees, percent signs,
``a/b`` and ``0.5`` forms, short ``x =`` prefixes, spaces) and then compares the
normalized strings, falling back to a numeric comparison when both sides parse
as numbers. There is no symbolic (sympy) equivalence check, so mathematically
equal but differently written answers can be marked wrong; that bias is the
same for every optimizer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from prompt_optimizer.tasks.base import Task, TaskExample, last_answer_line


def last_boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` (or ``\\fbox{...}``), braces balanced."""

    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    brace = text.find("{", start)
    if brace < 0:
        rest = text[start + len("\\boxed") :].strip()
        return rest.split("$")[0].strip() or None
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : index]
    return None


def _fix_fracs(text: str) -> str:
    parts = text.split("\\frac")
    out = parts[0]
    for part in parts[1:]:
        out += "\\frac"
        if part.startswith("{") or len(part) < 2:
            out += part
            continue
        a, b, rest = part[0], part[1], part[2:]
        if b == "{":
            out += "{" + a + "}" + b + rest
        else:
            out += "{" + a + "}{" + b + "}" + rest
    return out


def _fix_a_slash_b(text: str) -> str:
    if text.count("/") != 1:
        return text
    a, b = text.split("/")
    if a.lstrip("-").isdigit() and b.isdigit():
        return "\\frac{" + a + "}{" + b + "}"
    return text


def _fix_sqrt(text: str) -> str:
    return re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", text)


def _remove_right_units(text: str) -> str:
    if "\\text{ " in text:
        return text.split("\\text{ ")[0]
    return text


def strip_string(text: str) -> str:
    """Hendrycks et al. (2021) answer normalization."""

    text = text.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    text = text.replace("tfrac", "frac").replace("dfrac", "frac")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("^{\\circ}", "").replace("^\\circ", "")
    text = text.replace("\\$", "").replace("$", "")
    text = _remove_right_units(text)
    text = text.replace("\\%", "").replace("%", "")
    text = text.replace(" .", " 0.").replace("{.", "{0.")
    if not text:
        return text
    if text.startswith("."):
        text = "0" + text
    if text.count("=") == 1 and len(text.split("=")[0].strip()) <= 2:
        text = text.split("=")[1]
    text = _fix_sqrt(text)
    text = text.replace(" ", "")
    text = _fix_fracs(text)
    if text == "0.5":
        text = "\\frac{1}{2}"
    text = _fix_a_slash_b(text)
    return text


def _as_number(text: str) -> float | None:
    cleaned = text.replace(",", "")
    match = re.fullmatch(r"\\frac\{(-?\d+)\}\{(\d+)\}", cleaned)
    if match and int(match.group(2)) != 0:
        return int(match.group(1)) / int(match.group(2))
    try:
        return float(cleaned)
    except ValueError:
        return None


def is_equiv(predicted: str, gold: str) -> bool:
    a, b = strip_string(predicted), strip_string(gold)
    if a == b:
        return True
    x, y = _as_number(a), _as_number(b)
    return x is not None and y is not None and abs(x - y) <= 1e-6 * max(1.0, abs(y))


class MATH500(Task):
    """Competition mathematics with LaTeX answers."""

    name = "math500"
    contract = "End your response with the final answer written as \\boxed{<answer>}."
    seed_instruction = "Solve the math problem."
    description = (
        "Competition mathematics (algebra, counting and probability, geometry, intermediate "
        "algebra, number theory, prealgebra, precalculus) with LaTeX answers."
    )
    max_tokens = 1024

    def extract(self, output: str) -> str:
        boxed = last_boxed(output)
        if boxed is not None:
            return boxed.strip()
        line = last_answer_line(output)
        return (line or "").strip().strip("$").strip()

    def score(self, example: TaskExample, output: str) -> float:
        predicted = self.extract(output)
        return 1.0 if predicted and is_equiv(predicted, example.answer) else 0.0

    def gold_output(self, example: TaskExample) -> str:
        body = example.rationale.strip()
        if "\\boxed" in body:
            return body
        return f"{body}\n\\boxed{{{example.answer}}}".strip()


def load(root: Path, split: str) -> list[TaskExample]:
    out: list[TaskExample] = []
    with open(root / "math" / f"{split}.jsonl", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            record = json.loads(line)
            out.append(
                TaskExample(
                    id=f"math-{split}-{record.get('unique_id', index)}",
                    question=record["problem"].strip(),
                    answer=str(record["answer"]).strip(),
                    rationale=record["solution"].strip(),
                    meta={"subject": record.get("subject", ""), "level": record.get("level", 0)},
                )
            )
    return out
