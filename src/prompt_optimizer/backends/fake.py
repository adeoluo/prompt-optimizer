"""A deterministic fake language model for tests and offline demos.

The fake has to give prompt optimizers a real, prompt-dependent signal while
staying fully deterministic, so every optimizer can be unit tested without a
model. It answers two kinds of request.

**Task rollouts** (``purpose == "task"``). The query is an arithmetic question
such as ``What is 17 + 25?`` (see ``tasks.synthetic``). Each question has a
fixed difficulty ``u`` in [0, 1) drawn from a hash of the question text. The
prompt has a skill ``p`` computed from its system message and demonstrations:

    p = 0.25 + 0.15 * (good phrases in the instruction, at most 3)
             + 0.06 * (correct demonstrations, at most 4)
             - 0.10 * (incorrect demonstrations)
             - 0.20 * [any harmful phrase present]

clipped to [0.02, 0.98]. The answer is correct exactly when ``u < p``. So
accuracy is monotone in ``p``: better instructions and better demonstrations
lift accuracy, wrong demonstrations hurt it, and the optimum is reachable.

**Optimizer-side calls** (proposals, mutations, reflections, critiques,
summaries). The reply is a new instruction wrapped in ``<instruction>`` tags,
built from a phrase pool. Phrases that appear in the request (for example in a
critique or a reflection on failures) are preferred, so feedback-driven
optimizers can genuinely use the feedback. Choices are a pure function of the
request content, seed and temperature.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from prompt_optimizer.backends.base import LMRequest, LMResponse

GOOD_PHRASES: tuple[str, ...] = (
    "step by step",
    "verify",
    "double-check",
    "carefully",
    "units",
)
BAD_PHRASES: tuple[str, ...] = ("guess", "skip the work", "ignore")
NEUTRAL_PHRASES: tuple[str, ...] = ("be concise", "be polite", "use plain words", "be friendly")

_QUESTION = re.compile(r"What is (-?\d+) ([+\-*]) (-?\d+)\?")
_ANSWER = re.compile(r"Answer:\s*(-?\d+)")


def _unit(*parts: object) -> float:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def solve(question: str) -> int | None:
    """Exact answer to a synthetic arithmetic question, or None if unparsable."""

    match = _QUESTION.search(question)
    if not match:
        return None
    left, op, right = int(match.group(1)), match.group(2), int(match.group(3))
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    return left * right


def _count_tokens(text: str) -> int:
    return max(1, len(text.split()))


class FakeLM:
    """Deterministic stand-in for a chat model (see module docstring)."""

    def __init__(self, model_id: str = "fake-lm", *, difficulty_seed: int = 0) -> None:
        self._model_id = model_id
        self.difficulty_seed = difficulty_seed
        self.requests: list[LMRequest] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    def skill(self, request: LMRequest) -> float:
        system = " ".join(m.content.lower() for m in request.messages if m.role == "system")
        good = sum(1 for phrase in GOOD_PHRASES if phrase in system)
        harmful = any(phrase in system for phrase in BAD_PHRASES)
        correct_demos = wrong_demos = 0
        pending_question: str | None = None
        for message in request.messages[:-1]:
            if message.role == "user":
                pending_question = message.content
            elif message.role == "assistant" and pending_question is not None:
                truth = solve(pending_question)
                given = _ANSWER.search(message.content)
                if truth is not None and given is not None and int(given.group(1)) == truth:
                    correct_demos += 1
                else:
                    wrong_demos += 1
                pending_question = None
        p = 0.25 + 0.15 * min(good, 3) + 0.06 * min(correct_demos, 4) - 0.10 * wrong_demos
        if harmful:
            p -= 0.20
        return min(0.98, max(0.02, p))

    def difficulty(self, question: str) -> float:
        return _unit("difficulty", self.difficulty_seed, question)

    def _task_answer(self, request: LMRequest) -> str:
        question = request.messages[-1].content
        truth = solve(question)
        if truth is None:
            return "I cannot parse this question.\nAnswer: 0"
        correct = self.difficulty(question) < self.skill(request)
        value = truth if correct else truth + 1
        return f"Work it out step by step: the result is {value}.\nAnswer: {value}"

    def _instruction(self, request: LMRequest) -> str:
        content = " ".join(m.content.lower() for m in request.messages)
        # Only the latest user turn counts as feedback; the system text of a
        # meta-prompt lists every phrase and would otherwise bias every choice.
        feedback = request.messages[-1].content.lower()
        pool = GOOD_PHRASES + BAD_PHRASES + NEUTRAL_PHRASES
        chosen: list[str] = []
        for index, phrase in enumerate(pool):
            draw = _unit("phrase", content, request.seed, request.temperature, index)
            threshold = 0.65 if phrase in feedback else 0.22
            if draw < threshold:
                chosen.append(phrase)
        if not chosen:
            chosen.append(pool[int(_unit("fallback", content, request.seed) * len(pool))])
        body = "Solve the arithmetic problem; " + ", ".join(chosen) + "."
        return f"<instruction>{body}</instruction>"

    def _critique(self, request: LMRequest) -> str:
        content = " ".join(m.content.lower() for m in request.messages)
        missing = [phrase for phrase in GOOD_PHRASES if phrase not in content]
        if not missing:
            return "The instruction already covers the important behaviours."
        pick = missing[int(_unit("critique", content, request.seed) * len(missing))]
        return f"The answers are wrong because the instruction does not say to {pick}."

    def answer(self, request: LMRequest) -> str:
        if request.purpose == "task":
            return self._task_answer(request)
        if request.purpose == "critique":
            return self._critique(request)
        if request.purpose == "summarize":
            return "The data are short arithmetic questions with integer answers."
        return self._instruction(request)

    def generate(self, requests: Sequence[LMRequest]) -> list[LMResponse]:
        out: list[LMResponse] = []
        for request in requests:
            self.requests.append(request)
            text = self.answer(request)
            prompt_tokens = sum(_count_tokens(m.content) for m in request.messages)
            out.append(
                LMResponse(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=_count_tokens(text),
                )
            )
        return out
