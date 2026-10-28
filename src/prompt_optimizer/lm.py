"""A deterministic, offline simulated language model.

The optimizers need a model whose output genuinely depends on the prompt, so
that searching over instructions and demonstrations has a real signal, yet is
reproducible and free of any network dependency. This simulated model captures
the two behaviours that make prompt optimization work in practice:

* **In-context learning.** Given demonstrations, it answers a query by copying
  the label of the most similar demonstration (nearest neighbour over token
  overlap). Better and more relevant demonstrations therefore raise accuracy.
* **Instruction following.** It reads the label vocabulary out of the
  instruction and snaps its answer to the closest valid label. Without the
  vocabulary it must guess from keywords in the query, which is weaker.

Neither behaviour is hand-tuned to the bundled dataset; both are generic.
"""

from __future__ import annotations

import re

from prompt_optimizer.text import jaccard, tokenize
from prompt_optimizer.types import Prompt

_LABELS_CLAUSE = re.compile(r"labels?\s*:\s*(.+)", re.IGNORECASE)
_UNKNOWN = "unknown"


def parse_label_vocabulary(instruction: str) -> tuple[str, ...]:
    """Extract an ordered label vocabulary from an instruction, if present.

    Recognizes a clause such as ``Valid labels: refund, shipping, cancel``.
    Returns an empty tuple when the instruction does not declare a vocabulary.
    """

    match = _LABELS_CLAUSE.search(instruction)
    if not match:
        return ()
    raw = match.group(1)
    labels = [item.strip().lower() for item in re.split(r"[,/]", raw) if item.strip()]
    deduped: list[str] = []
    for label in labels:
        if label not in deduped:
            deduped.append(label)
    return tuple(deduped)


class SimulatedLanguageModel:
    """A deterministic stand-in for a real language model."""

    def generate(self, prompt: Prompt, query: str) -> str:
        vocabulary = parse_label_vocabulary(prompt.instruction)
        candidate = self._raw_answer(prompt, query, vocabulary)
        return self._snap_to_vocabulary(candidate, vocabulary)

    def _raw_answer(self, prompt: Prompt, query: str, vocabulary: tuple[str, ...]) -> str:
        if prompt.demonstrations:
            best_label = ""
            best_score = -1.0
            for demo in prompt.demonstrations:
                score = jaccard(query, demo.text)
                # Deterministic tie-break: earlier demonstration wins.
                if score > best_score:
                    best_score = score
                    best_label = demo.label
            if best_score > 0.0:
                return best_label
        # No demonstrations, or none overlapped: guess from query keywords.
        return self._keyword_guess(query, vocabulary)

    def _keyword_guess(self, query: str, vocabulary: tuple[str, ...]) -> str:
        tokens = tokenize(query)
        for token in tokens:
            if token in vocabulary:
                return token
        return tokens[0] if tokens else _UNKNOWN

    def _snap_to_vocabulary(self, candidate: str, vocabulary: tuple[str, ...]) -> str:
        if not vocabulary:
            return candidate
        if candidate in vocabulary:
            return candidate
        best_label = _UNKNOWN
        best_score = 0.0
        for label in vocabulary:
            score = jaccard(candidate, label)
            if score > best_score:
                best_score = score
                best_label = label
        return best_label
