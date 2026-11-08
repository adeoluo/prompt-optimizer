"""Shared machinery for the optimizers: evaluation, budgets, trials, proposals.

Every optimizer in ``prompt_optimizer.optim`` follows one contract:

* It receives a seed program, a training set (a pool for demonstrations,
  minibatches and reflection) and a validation set (for selection), plus a
  ``Budget`` on task rollouts.
* It spends task rollouts only through ``Evaluator.evaluate``, which memoizes
  (program, example) pairs within a run, charges only new pairs to the budget,
  and refuses a batch that would overrun it (``BudgetExhaustedError``).
* It returns the candidate with the best full-validation score it measured,
  with the full trial history.

The test split is never visible to an optimizer; it is scored afterwards by the
experiment runner. Comparing the validation score of the returned program with
its test score is the overfitting analysis in the evaluation report.
"""

from __future__ import annotations

import random
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from prompt_optimizer.backends.base import LanguageModel, LMRequest, Message, Purpose
from prompt_optimizer.program import PromptProgram
from prompt_optimizer.tasks.base import Task, TaskExample, mean


class BudgetExhaustedError(RuntimeError):
    """Raised before a batch that would exceed the rollout budget."""


@dataclass
class Budget:
    """A cap on task rollouts (model calls on train and validation examples)."""

    max_rollouts: int | None = None
    used: int = 0

    @property
    def remaining(self) -> float:
        return float("inf") if self.max_rollouts is None else self.max_rollouts - self.used

    def can_afford(self, n: int) -> bool:
        return n <= self.remaining

    def charge(self, n: int) -> None:
        if not self.can_afford(n):
            raise BudgetExhaustedError(f"needs {n} rollouts, {self.remaining} left")
        self.used += n


@dataclass(frozen=True)
class Rollout:
    """One task rollout: the response to one example under one program."""

    example: TaskExample
    output: str
    score: float
    truncated: bool = False


@dataclass(frozen=True)
class EvalResult:
    """Rollouts of one program over a list of examples."""

    program: PromptProgram
    rollouts: tuple[Rollout, ...]

    @property
    def scores(self) -> list[float]:
        return [r.score for r in self.rollouts]

    @property
    def mean(self) -> float:
        return mean(self.scores)

    def by_id(self) -> dict[str, float]:
        return {r.example.id: r.score for r in self.rollouts}


class Evaluator:
    """Batched, memoized, budgeted scoring of programs on examples."""

    def __init__(
        self,
        task: Task,
        lm: LanguageModel,
        budget: Budget | None = None,
        *,
        max_tokens: int | None = None,
    ) -> None:
        self.task = task
        self.lm = lm
        self.budget = budget if budget is not None else Budget()
        self.max_tokens = max_tokens or task.max_tokens
        self._memo: dict[tuple[str, str], Rollout] = {}

    def request(self, program: PromptProgram, example: TaskExample) -> LMRequest:
        return program.request(
            self.task.contract,
            example.question,
            group=example.group,
            max_tokens=self.max_tokens,
        )

    def evaluate(
        self, program: PromptProgram, examples: Sequence[TaskExample], *, charge: bool = True
    ) -> EvalResult:
        return self.evaluate_many([(program, examples)], charge=charge)[0]

    def pending(self, program: PromptProgram, examples: Sequence[TaskExample]) -> int:
        """Rollouts ``evaluate`` would still have to run (and charge)."""

        fingerprint = program.fingerprint()
        return sum(1 for ex in dict.fromkeys(examples) if (fingerprint, ex.id) not in self._memo)

    def evaluate_many(
        self,
        jobs: Sequence[tuple[PromptProgram, Sequence[TaskExample]]],
        *,
        charge: bool = True,
    ) -> list[EvalResult]:
        """Score several programs with one model batch, so the server's slots stay full."""

        todo: dict[tuple[str, str], tuple[PromptProgram, TaskExample]] = {}
        for program, examples in jobs:
            fingerprint = program.fingerprint()
            for ex in examples:
                key = (fingerprint, ex.id)
                if key not in self._memo and key not in todo:
                    todo[key] = (program, ex)
        if todo:
            if charge:
                self.budget.charge(len(todo))
            responses = self.lm.generate([self.request(p, ex) for p, ex in todo.values()])
            for key, (_, ex), resp in zip(todo, todo.values(), responses, strict=True):
                self._memo[key] = Rollout(
                    ex, resp.text, self.task.score(ex, resp.text), resp.truncated
                )
        return [
            EvalResult(
                program, tuple(self._memo[(program.fingerprint(), ex.id)] for ex in examples)
            )
            for program, examples in jobs
        ]

    def known(self, program: PromptProgram, example: TaskExample) -> bool:
        return (program.fingerprint(), example.id) in self._memo


@dataclass(frozen=True)
class Trial:
    """One scored candidate in an optimizer's history."""

    step: int
    fingerprint: str
    instruction: str
    n_demos: int
    score: float
    n_examples: int
    split: str
    rollouts_used: int
    note: str = ""


@dataclass
class OptimizerResult:
    """The selected program, its validation score and the search history."""

    program: PromptProgram
    val_score: float
    trials: list[Trial]
    rollouts_used: int
    seconds: float
    info: dict[str, Any] = field(default_factory=dict)


_TAGGED = re.compile(r"<instruction>(.*?)(?:</instruction>|$)", re.IGNORECASE | re.DOTALL)
_FENCED = re.compile(r"```(?:[a-zA-Z]*\n)?(.*?)```", re.DOTALL)
_LEADING = re.compile(
    r"^(?:here is|here's)?\s*(?:the|a)?\s*(?:new|improved|revised|updated)?\s*"
    r"(?:instruction|prompt)\s*:\s*",
    re.IGNORECASE,
)
MAX_INSTRUCTION_CHARS = 2000


def extract_instruction(text: str) -> str | None:
    """Pull a proposed instruction out of a proposer response.

    Order of preference: ``<instruction>`` tags, a fenced block, then the whole
    reply with a leading label such as ``New instruction:`` removed. Returns None
    for an empty result. Long proposals are truncated at a sentence boundary.
    """

    match = _TAGGED.search(text)
    if match:
        body = match.group(1)
    else:
        fenced = _FENCED.search(text)
        body = fenced.group(1) if fenced else text
    body = _LEADING.sub("", body.strip()).strip().strip('"').strip()
    if not body:
        return None
    if len(body) > MAX_INSTRUCTION_CHARS:
        cut = body[:MAX_INSTRUCTION_CHARS]
        body = cut[: cut.rfind(".") + 1] or cut
    return body


class OptimizerContext:
    """Everything one optimization run needs, threaded through the search."""

    def __init__(
        self,
        task: Task,
        lm: LanguageModel,
        trainset: Sequence[TaskExample],
        valset: Sequence[TaskExample],
        *,
        budget: Budget | None = None,
        seed: int = 0,
        proposer_temperature: float = 0.9,
        proposer_max_tokens: int = 400,
    ) -> None:
        if not trainset or not valset:
            raise ValueError("trainset and valset must be non-empty")
        self.task = task
        self.lm = lm
        self.trainset = list(trainset)
        self.valset = list(valset)
        self.budget = budget if budget is not None else Budget()
        self.evaluator = Evaluator(task, lm, self.budget)
        self.rng = random.Random(seed)
        self.seed = seed
        self.proposer_temperature = proposer_temperature
        self.proposer_max_tokens = proposer_max_tokens
        self.trials: list[Trial] = []
        self.step = 0
        self._best: tuple[float, PromptProgram] | None = None
        self._val_scores: dict[str, float] = {}
        self._programs: dict[str, PromptProgram] = {}

    # -- scoring ---------------------------------------------------------------------------
    def score_val(self, program: PromptProgram, note: str = "") -> float:
        """Full-validation score, recorded as a trial and tracked for selection."""

        fingerprint = program.fingerprint()
        if fingerprint in self._val_scores:
            return self._val_scores[fingerprint]
        result = self.evaluator.evaluate(program, self.valset)
        self._val_scores[fingerprint] = result.mean
        self._record(program, result.mean, len(self.valset), "val", note)
        if self._best is None or result.mean > self._best[0]:
            self._best = (result.mean, program)
        return result.mean

    def score_val_many(self, programs: Sequence[PromptProgram], note: str = "") -> list[float]:
        """Full-validation scores for several programs, run as one batch."""

        fresh = list(
            {
                p.fingerprint(): p for p in programs if p.fingerprint() not in self._val_scores
            }.values()
        )
        if fresh:
            results = self.evaluator.evaluate_many([(p, self.valset) for p in fresh])
            for program, result in zip(fresh, results, strict=True):
                self._val_scores[program.fingerprint()] = result.mean
                self._record(program, result.mean, len(self.valset), "val", note)
                if self._best is None or result.mean > self._best[0]:
                    self._best = (result.mean, program)
        return [self._val_scores[p.fingerprint()] for p in programs]

    def score_finalists(self, programs: Sequence[PromptProgram], note: str = "finalist") -> None:
        """Score, in order, as many distinct programs on validation as the budget allows."""

        chosen: dict[str, PromptProgram] = {}
        need = 0
        for program in programs:
            if program.fingerprint() in chosen:
                continue
            cost = self.unseen_val_cost(program)
            if self.affordable(need + cost):
                chosen[program.fingerprint()] = program
                need += cost
        self.score_val_many(list(chosen.values()), note)

    def val_rollouts(self, program: PromptProgram, note: str = "") -> EvalResult:
        """The memoized validation rollouts of a program (scores it first if needed)."""

        self.score_val(program, note=note)
        return self.evaluator.evaluate(program, self.valset, charge=False)

    def score_batch(
        self,
        program: PromptProgram,
        batch: Sequence[TaskExample],
        note: str = "",
        *,
        split: str = "train",
    ) -> EvalResult:
        """Minibatch score (training examples unless ``split`` says otherwise), recorded."""

        return self.score_batch_many([program], batch, note=note, split=split)[0]

    def score_batch_many(
        self,
        programs: Sequence[PromptProgram],
        batch: Sequence[TaskExample],
        note: str = "",
        *,
        split: str = "train",
    ) -> list[EvalResult]:
        """Minibatch scores for several programs, run as one batch and recorded as trials."""

        results = self.evaluator.evaluate_many([(p, batch) for p in programs])
        for program, result in zip(programs, results, strict=True):
            self._record(program, result.mean, len(batch), split, note)
        return results

    def _record(self, program: PromptProgram, score: float, n: int, split: str, note: str) -> None:
        self._programs.setdefault(program.fingerprint(), program)
        self.trials.append(
            Trial(
                step=self.step,
                fingerprint=program.fingerprint(),
                instruction=program.instruction,
                n_demos=len(program.demos),
                score=score,
                n_examples=n,
                split=split,
                rollouts_used=self.budget.used,
                note=note,
            )
        )

    def known_val_score(self, program: PromptProgram) -> float | None:
        return self._val_scores.get(program.fingerprint())

    @property
    def best(self) -> tuple[float, PromptProgram] | None:
        return self._best

    # -- proposer calls --------------------------------------------------------------------
    def ask(
        self,
        system: str,
        user: str,
        *,
        n: int = 1,
        purpose: Purpose = "propose",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> list[str]:
        """Sample ``n`` replies from the optimizer-side model with distinct seeds."""

        temp = self.proposer_temperature if temperature is None else temperature
        requests = [
            LMRequest(
                messages=(Message("system", system), Message("user", user)),
                max_tokens=max_tokens or self.proposer_max_tokens,
                temperature=temp,
                seed=self.rng.randrange(2**31),
                purpose=purpose,
            )
            for _ in range(n)
        ]
        return [resp.text for resp in self.lm.generate(requests)]

    def ask_many(
        self,
        system: str,
        users: Sequence[str],
        *,
        purpose: Purpose = "propose",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> list[str]:
        """One reply per user message, sent to the model as a single batch."""

        temp = self.proposer_temperature if temperature is None else temperature
        requests = [
            LMRequest(
                messages=(Message("system", system), Message("user", user)),
                max_tokens=max_tokens or self.proposer_max_tokens,
                temperature=temp,
                seed=self.rng.randrange(2**31),
                purpose=purpose,
            )
            for user in users
        ]
        return [resp.text for resp in self.lm.generate(requests)]

    def propose_many(
        self,
        system: str,
        users: Sequence[str],
        *,
        purpose: Purpose = "propose",
        temperature: float | None = None,
    ) -> list[str | None]:
        """Batched proposals, parsed; position ``i`` answers ``users[i]`` (None if empty)."""

        texts = self.ask_many(system, users, purpose=purpose, temperature=temperature)
        return [extract_instruction(text) for text in texts]

    def propose_instructions(
        self,
        system: str,
        user: str,
        *,
        n: int = 1,
        purpose: Purpose = "propose",
        temperature: float | None = None,
    ) -> list[str]:
        """Sample replies and parse each into an instruction; drop empty ones."""

        texts = self.ask(system, user, n=n, purpose=purpose, temperature=temperature)
        out: list[str] = []
        for text in texts:
            instruction = extract_instruction(text)
            if instruction and instruction not in out:
                out.append(instruction)
        return out

    def sample(self, pool: Sequence[TaskExample], k: int) -> list[TaskExample]:
        k = min(k, len(pool))
        return self.rng.sample(list(pool), k)

    # -- budget helpers --------------------------------------------------------------------
    @property
    def val_cost(self) -> int:
        """Rollouts one full-validation score costs."""

        return len(self.valset)

    def affordable(self, n: int, reserve: int = 0) -> bool:
        """True when ``n`` more rollouts still leave ``reserve`` for later steps."""

        return self.budget.remaining - reserve >= n

    def cap(self, n: int, reserve: int = 0) -> int:
        """The largest count up to ``n`` that still leaves ``reserve`` rollouts."""

        return int(max(0.0, min(float(n), self.budget.remaining - reserve)))

    def unseen_val_cost(self, program: PromptProgram) -> int:
        """Validation rollouts not yet memoized for ``program`` (0 if already scored)."""

        if program.fingerprint() in self._val_scores:
            return 0
        return sum(1 for ex in self.valset if not self.evaluator.known(program, ex))

    def scored_programs(self) -> list[tuple[float, PromptProgram]]:
        """Every program with a full-validation score, best first."""

        seen: dict[str, PromptProgram] = {}
        for trial in self.trials:
            if trial.split == "val" and trial.fingerprint not in seen:
                seen[trial.fingerprint] = self._programs[trial.fingerprint]
        ranked = [(self._val_scores[fp], prog) for fp, prog in seen.items()]
        return sorted(ranked, key=lambda pair: -pair[0])


class Optimizer(ABC):
    """Base class: subclasses implement ``_search``; ``compile`` handles budgets."""

    name: str = "optimizer"

    @abstractmethod
    def _search(self, ctx: OptimizerContext, seed_program: PromptProgram) -> None:
        """Explore candidates, scoring finalists with ``ctx.score_val``."""

    def compile(
        self,
        task: Task,
        lm: LanguageModel,
        trainset: Sequence[TaskExample],
        valset: Sequence[TaskExample],
        *,
        seed_program: PromptProgram | None = None,
        budget: Budget | None = None,
        seed: int = 0,
    ) -> OptimizerResult:
        ctx = OptimizerContext(task, lm, trainset, valset, budget=budget, seed=seed)
        program = seed_program or PromptProgram(task.seed_instruction)
        start = time.perf_counter()
        stopped = "completed"
        try:
            ctx.score_val(program, note="seed")
            self._search(ctx, program)
        except BudgetExhaustedError:
            stopped = "budget"
        best = ctx.best
        if best is None:  # the seed itself could not be afforded
            raise BudgetExhaustedError("budget too small to score the seed program on validation")
        score, chosen = best
        return OptimizerResult(
            program=chosen,
            val_score=score,
            trials=ctx.trials,
            rollouts_used=ctx.budget.used,
            seconds=time.perf_counter() - start,
            info={"stopped": stopped, "optimizer": self.name, "seed": seed},
        )


def format_examples(examples: Sequence[TaskExample], task: Task, limit: int = 3) -> str:
    """Render a few gold input/output pairs for a proposer prompt."""

    blocks = []
    for ex in examples[:limit]:
        blocks.append(f"Input:\n{ex.question}\n\nReference output:\n{task.gold_output(ex)}")
    return "\n\n---\n\n".join(blocks)


def clip(text: str, limit: int = 1200) -> str:
    """Shorten long inputs (HotpotQA contexts, long BBH tables) inside meta-prompts."""

    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 16].rstrip() + " ... [truncated]"


def format_rollouts(
    rollouts: Sequence[Rollout], task: Task, *, limit: int = 4, input_chars: int = 1200
) -> str:
    """Render rollouts with the task's feedback for critique and reflection prompts."""

    blocks = []
    for index, rollout in enumerate(rollouts[:limit], start=1):
        blocks.append(
            f"### Example {index}\n"
            f"Input:\n{clip(rollout.example.question, input_chars)}\n\n"
            f"Model output:\n{clip(rollout.output, 900)}\n\n"
            f"Feedback: {task.feedback(rollout.example, rollout.output)}"
        )
    return "\n\n".join(blocks)
