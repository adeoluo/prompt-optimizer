"""Deterministic text utilities shared by the language model and metrics."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "to",
        "of",
        "in",
        "on",
        "for",
        "and",
        "or",
        "my",
        "i",
        "you",
        "it",
        "this",
        "that",
        "with",
        "please",
        "can",
        "could",
        "would",
        "will",
        "do",
        "did",
        "have",
        "has",
        "me",
        "we",
        "they",
        "he",
        "she",
        "at",
        "be",
        "been",
        "as",
        "but",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric characters, and drop stopwords."""

    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS]


def jaccard(left: str, right: str) -> float:
    """Jaccard similarity of the token sets of two strings."""

    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return intersection / union
