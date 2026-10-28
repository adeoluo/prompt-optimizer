"""Evaluation metrics and dataset scoring helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from prompt_optimizer.text import tokenize
from prompt_optimizer.types import Example, Metric

# A callable that maps an input text to a predicted label.
Predict = Callable[[str], str]


def exact_match(predicted: str, expected: str) -> float:
    """Return 1.0 when the normalized labels match exactly, else 0.0."""

    return 1.0 if predicted.strip().lower() == expected.strip().lower() else 0.0


def token_f1(predicted: str, expected: str) -> float:
    """Token-level F1, useful for free-form rather than single-label outputs."""

    predicted_tokens = set(tokenize(predicted))
    expected_tokens = set(tokenize(expected))
    if not predicted_tokens and not expected_tokens:
        return 1.0
    if not predicted_tokens or not expected_tokens:
        return 0.0
    overlap = len(predicted_tokens & expected_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate(predict: Predict, dataset: Sequence[Example], metric: Metric = exact_match) -> float:
    """Average ``metric`` of ``predict`` over ``dataset`` (0.0 for an empty set)."""

    if not dataset:
        return 0.0
    total = sum(metric(predict(example.text), example.label) for example in dataset)
    return total / len(dataset)
