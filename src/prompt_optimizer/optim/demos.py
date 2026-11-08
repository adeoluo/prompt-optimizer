"""Demonstration bootstrapping and demo-set sampling shared by several optimizers.

Bootstrapping (Khattab et al., 2023, DSPy) runs a *teacher* program on training
examples and keeps the full outputs of the rollouts the metric marks correct.
Those traces become demonstrations: they show the model its own successful
reasoning in its own style, which is why bootstrapped demos usually beat gold
rationales written by humans in a different style. A trace is kept only when

* the score is 1 (exactly right), and
* the response was not cut off by ``max_tokens`` (a truncated trace teaches the
  model to stop mid-solution).

Demo sets are sampled per group. For grouped tasks (the 27 BIG-Bench Hard
subtasks) a set holds ``k`` demos for each group and a query only sees the demos
of its own group (``PromptProgram.demos_for``), so one program carries
per-subtask demonstrations next to one shared instruction.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from prompt_optimizer.optim.base import OptimizerContext
from prompt_optimizer.program import Demo, PromptProgram
from prompt_optimizer.tasks.base import TaskExample


def bootstrap_traces(
    ctx: OptimizerContext,
    teacher: PromptProgram,
    pool: Sequence[TaskExample],
    *,
    note: str = "bootstrap",
) -> list[Demo]:
    """Run ``teacher`` on ``pool`` (charged to the budget); return correct traces as demos."""

    if not pool:
        return []
    result = ctx.score_batch(teacher, pool, note=note)
    return [
        Demo(r.example.question, r.output.strip(), r.example.id, r.example.group)
        for r in result.rollouts
        if r.score >= 1.0 and not r.truncated and r.output.strip()
    ]


def labeled_demos(ctx: OptimizerContext, pool: Sequence[TaskExample]) -> list[Demo]:
    """Gold demonstrations built from dataset rationales (empty when there are none)."""

    return [ctx.task.gold_demo(ex) for ex in pool if ex.rationale.strip()]


def group_demos(demos: Sequence[Demo]) -> dict[str, list[Demo]]:
    grouped: dict[str, list[Demo]] = defaultdict(list)
    for demo in demos:
        grouped[demo.group].append(demo)
    return dict(grouped)


def sample_demo_set(rng: random.Random, demos: Sequence[Demo], k: int) -> tuple[Demo, ...]:
    """Up to ``k`` demos per group, drawn without replacement, in a stable group order."""

    if k <= 0:
        return ()
    chosen: list[Demo] = []
    for group, members in sorted(group_demos(demos).items()):
        del group
        take = min(k, len(members))
        chosen.extend(rng.sample(members, take))
    return tuple(chosen)


def first_demo_set(demos: Sequence[Demo], k: int) -> tuple[Demo, ...]:
    """The first ``k`` demos per group in pool order (DSPy's unshuffled candidate)."""

    chosen: list[Demo] = []
    for _, members in sorted(group_demos(demos).items()):
        chosen.extend(members[:k])
    return tuple(chosen)


def teacher_pool(ctx: OptimizerContext, size: int) -> list[TaskExample]:
    """A seeded sample of training examples for teacher rollouts, balanced by group."""

    by_group: dict[str, list[TaskExample]] = defaultdict(list)
    for ex in ctx.trainset:
        by_group[ex.group].append(ex)
    if len(by_group) <= 1:
        return ctx.sample(ctx.trainset, size)
    per_group = max(1, size // len(by_group))
    out: list[TaskExample] = []
    for _, members in sorted(by_group.items()):
        out.extend(ctx.rng.sample(members, min(per_group, len(members))))
    return out
