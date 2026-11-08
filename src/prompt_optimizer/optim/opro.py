"""OPRO: Optimization by PROmpting (Yang et al., 2023).

The proposer model is the optimizer. Each step it reads a *meta-prompt* with

* the optimization trajectory: up to ``max_history`` previous instructions with
  their scores (0 to 100), sorted in **ascending** order so the best come last,
  where the model's attention is strongest;
* a few task exemplars showing where the instruction goes (``<INS>``) and the
  expected output;
* the request to write an instruction different from all previous ones that
  scores higher.

It samples ``per_step`` candidates at temperature 1.0 (the paper's setting),
scores each on one fixed training subset so scores are comparable across
steps, and appends them to the trajectory. After ``steps`` rounds (or when the
budget for steps runs out) the ``n_finalists`` best by training score are
scored on validation.
"""

from __future__ import annotations

from dataclasses import dataclass

from prompt_optimizer.optim.base import Optimizer, OptimizerContext, clip
from prompt_optimizer.program import PromptProgram
from prompt_optimizer.tasks.base import TaskExample

_SYSTEM = (
    "You are an optimizer that writes better instructions for a language model. Reply with "
    "the new instruction inside <instruction></instruction> tags."
)


@dataclass
class OPRO(Optimizer):
    """LLM-as-optimizer over a scored trajectory of instructions."""

    steps: int = 8
    per_step: int = 3
    train_subset: int = 15
    max_history: int = 20
    n_exemplars: int = 3
    n_finalists: int = 2
    temperature: float = 1.0
    name: str = "opro"

    def meta_prompt(
        self, ctx: OptimizerContext, history: dict[str, float], exemplars: list[TaskExample]
    ) -> str:
        top = sorted(history.items(), key=lambda kv: kv[1])[-self.max_history :]
        trajectory = "\n\n".join(
            f"text:\n{instruction}\nscore:\n{round(100 * score)}" for instruction, score in top
        )
        shots = "\n\n".join(
            f"input:\nQ: <INS>\n{clip(ex.question, 600)}\nA:\noutput:\n"
            f"{clip(ctx.task.gold_output(ex), 400)}"
            for ex in exemplars
        )
        return (
            "Your task is to generate the instruction <INS>. Below are some previous "
            "instructions with their scores. The score ranges from 0 to 100.\n\n"
            f"{trajectory}\n\n"
            "Below are some problems. The instruction <INS> is placed before each problem.\n\n"
            f"{shots}\n\n"
            "Generate an instruction that is different from all the instructions above, and "
            "has a higher score than all the instructions above. The instruction should be "
            "concise, effective, and generally applicable to all problems above. The answer "
            f"line format is fixed and is not part of the instruction: {ctx.task.contract}"
        )

    def _search(self, ctx: OptimizerContext, seed_program: PromptProgram) -> None:
        reserve = self.n_finalists * ctx.val_cost
        subset = ctx.sample(ctx.trainset, self.train_subset)
        history: dict[str, float] = {}
        seed_score = ctx.score_batch(seed_program, subset, note="seed").mean
        history[seed_program.instruction] = seed_score
        for _ in range(self.steps):
            ctx.step += 1
            if not ctx.affordable(len(subset), reserve):
                break
            exemplars = ctx.sample(ctx.trainset, self.n_exemplars)
            prompt = self.meta_prompt(ctx, history, exemplars)
            texts = ctx.propose_many(
                _SYSTEM, [prompt] * self.per_step, temperature=self.temperature
            )
            fresh = [t for t in dict.fromkeys(texts) if t and t not in history]
            fresh = fresh[: ctx.cap(len(fresh) * len(subset), reserve) // len(subset)]
            if not fresh:
                continue
            programs = [seed_program.with_instruction(t) for t in fresh]
            for text, result in zip(
                fresh, ctx.score_batch_many(programs, subset, note="opro"), strict=True
            ):
                history[text] = result.mean
        ranked = sorted(history, key=lambda s: -history[s])[: self.n_finalists]
        ctx.score_finalists([seed_program.with_instruction(s) for s in ranked])
