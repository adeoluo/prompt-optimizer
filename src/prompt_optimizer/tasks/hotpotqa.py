"""HotpotQA, distractor setting, dev split (Yang et al., 2018).

Each of the 7,405 dev questions comes with ten Wikipedia paragraphs: the two
gold paragraphs and eight retrieved distractors. The model reads all ten and
answers with a short span (or yes/no). Scoring follows the official evaluation
script: answers are lower-cased, stripped of punctuation and the articles
a/an/the, whitespace-normalized, then compared by exact match and token F1.
The optimization score is F1, which gives a smoother signal than exact match
on small validation sets; both are reported.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from prompt_optimizer.tasks.base import Task, TaskExample, last_answer_line

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize(text: str) -> str:
    """The official HotpotQA / SQuAD answer normalization."""

    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(predicted: str, gold: str) -> float:
    return 1.0 if normalize(predicted) == normalize(gold) else 0.0


def f1(predicted: str, gold: str) -> float:
    """Official token F1, including the yes/no/noanswer special case."""

    pred, ref = normalize(predicted), normalize(gold)
    special = {"yes", "no", "noanswer"}
    if (pred in special or ref in special) and pred != ref:
        return 0.0
    pred_tokens, ref_tokens = pred.split(), ref.split()
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


class HotpotQA(Task):
    """Multi-hop reading comprehension over ten paragraphs."""

    name = "hotpotqa"
    contract = (
        "End your response with a final line of the form 'Answer: <answer>', where <answer> is "
        "only the short answer: a few words copied from the paragraphs, or yes or no."
    )
    seed_instruction = "Answer the question using the paragraphs."
    description = (
        "Multi-hop questions over ten Wikipedia paragraphs, two of which hold the needed "
        "facts; answers are short spans or yes/no."
    )
    max_tokens = 384

    def extract(self, output: str) -> str:
        line = last_answer_line(output)
        if line is None:
            lines = [ln for ln in output.strip().splitlines() if ln.strip()]
            line = lines[-1] if lines else ""
        return line.strip().strip("*").strip().rstrip(".")

    def score(self, example: TaskExample, output: str) -> float:
        return f1(self.extract(output), example.answer)

    def metrics(self, example: TaskExample, output: str) -> dict[str, float]:
        predicted = self.extract(output)
        return {"em": exact_match(predicted, example.answer), "f1": f1(predicted, example.answer)}


def render(record: Mapping[str, Any]) -> str:
    context = record["context"]
    blocks = []
    for index, (title, sentences) in enumerate(
        zip(context["title"], context["sentences"], strict=True), start=1
    ):
        blocks.append(f"[{index}] {title}: {''.join(sentences).strip()}")
    return "Paragraphs:\n" + "\n".join(blocks) + f"\n\nQuestion: {record['question'].strip()}"


def load(root: Path) -> list[TaskExample]:
    path = root / "hotpotqa" / "distractor_validation.jsonl"
    out: list[TaskExample] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            out.append(
                TaskExample(
                    id=f"hotpot-{record['id']}",
                    question=render(record),
                    answer=record["answer"].strip(),
                    meta={"type": record["type"], "level": record["level"]},
                )
            )
    return out
