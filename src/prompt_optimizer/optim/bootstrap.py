"""Bootstrapped few-shot with random search (Khattab et al., 2023; DSPy).

The instruction is left unchanged; only the demonstrations are optimized.

1. **Bootstrap.** Run the seed program (the teacher) on ``n_teacher`` training
   examples and keep the correct, untruncated outputs as candidate demos
   (``optim.demos.bootstrap_traces``).
2. **Candidates.** Build demo sets in DSPy's order: the unshuffled first ``k``
   traces, labeled gold demos when the dataset has rationales, then
   ``n_candidates`` random sets whose size per group is drawn uniformly from
   ``[1, max_demos]``.
3. **Select.** Score each candidate on the full validation set while the budget
   allows and return the best (the seed itself is always a candidate).

The search is pure random search, so its cost is easy to state: one teacher
rollout per training example used plus one validation pass per candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

from prompt_optimizer.optim.base import Optimizer, OptimizerContext
from prompt_optimizer.optim.demos import (
    bootstrap_traces,
    first_demo_set,
    labeled_demos,
    sample_demo_set,
    teacher_pool,
)
from prompt_optimizer.program import Demo, PromptProgram


@dataclass
class BootstrapFewShotRS(Optimizer):
    """Bootstrapped demonstrations chosen by random search on validation."""

    max_demos: int = 4
    n_teacher: int = 40
    n_candidates: int = 6
    use_labeled: bool = True
    name: str = "bootstrap_fewshot"

    def candidate_sets(
        self, ctx: OptimizerContext, traces: list[Demo], labeled: list[Demo]
    ) -> list[tuple[str, tuple[Demo, ...]]]:
        out: list[tuple[str, tuple[Demo, ...]]] = []
        if traces:
            out.append(("bootstrap-first", first_demo_set(traces, self.max_demos)))
        if self.use_labeled and labeled:
            out.append(("labeled", sample_demo_set(ctx.rng, labeled, self.max_demos)))
        for index in range(self.n_candidates):
            if not traces:
                break
            k = ctx.rng.randint(1, self.max_demos)
            out.append((f"bootstrap-random-{index}", sample_demo_set(ctx.rng, traces, k)))
        unique: dict[tuple[Demo, ...], str] = {}
        for note, demos in out:
            if demos and demos not in unique:
                unique[demos] = note
        return [(note, demos) for demos, note in unique.items()]

    def _search(self, ctx: OptimizerContext, seed_program: PromptProgram) -> None:
        reserve = ctx.val_cost
        pool = teacher_pool(ctx, self.n_teacher)
        pool = pool[: ctx.cap(len(pool), reserve)]
        traces = bootstrap_traces(ctx, seed_program, pool)
        labeled = labeled_demos(ctx, ctx.sample(ctx.trainset, 4 * self.max_demos))
        ctx.step += 1
        for note, demos in self.candidate_sets(ctx, traces, labeled):
            candidate = seed_program.with_demos(demos)
            if not ctx.affordable(ctx.unseen_val_cost(candidate)):
                break
            ctx.score_val(candidate, note=note)
            ctx.step += 1
