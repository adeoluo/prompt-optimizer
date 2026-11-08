"""GEPA-style reflective prompt evolution with Pareto selection (Agrawal et al., 2025).

GEPA keeps a pool of candidate programs, each with its score on every
validation instance, and repeats:

1. **Select a parent** by Pareto sampling (``optim.pareto.select_candidate``):
   candidates that are best on at least one validation instance survive, and
   are drawn in proportion to how many instances they win.
2. **Reflect.** Run the parent on a small training minibatch (``minibatch``
   examples, 3 in the paper) and show the reflection model the instruction, the
   inputs, the full outputs and the task's feedback. It diagnoses what went
   wrong and writes an improved instruction that keeps the lessons that apply
   generally (the reflective mutation of the paper).
3. **Gate on the minibatch.** Score the child on the same minibatch; only a
   strict improvement over the parent earns a full validation pass.
4. **Grow the pool.** Accepted children join the pool with their per-instance
   validation scores, which updates the Pareto fronts.

The returned program is the pool member with the best aggregate validation
score. GEPA's *merge* operator recombines modules from different lineages of a
multi-module program; a ``PromptProgram`` has one instruction, so merge is
replaced here by an optional *crossover* reflection that asks the model to
combine the two parents' instructions (``crossover_every``).
"""

from __future__ import annotations

from dataclasses import dataclass

from prompt_optimizer.optim.base import Optimizer, OptimizerContext, format_rollouts
from prompt_optimizer.optim.pareto import remove_dominated, select_candidate
from prompt_optimizer.program import PromptProgram

_REFLECT = (
    "You improve the instruction given to an assistant. You read examples of the assistant's "
    "work with feedback, find what went wrong and what worked, and write a better instruction. "
    "Reply with the new instruction inside <instruction></instruction> tags."
)


@dataclass
class GEPA(Optimizer):
    """Reflective mutation, minibatch gating and per-instance Pareto selection."""

    iterations: int = 30
    minibatch: int = 3
    crossover_every: int = 0
    name: str = "gepa"

    def reflection_prompt(self, instruction: str, examples: str, contract: str) -> str:
        return (
            "I provided an assistant with the following instruction to perform a task:\n"
            f"```\n{instruction}\n```\n\n"
            f"(Every response must also follow this fixed format line: {contract})\n\n"
            "The following are examples of task inputs, the assistant's responses and "
            f"feedback on how they could be better:\n\n{examples}\n\n"
            "Your task is to write a new instruction for the assistant. Read the inputs "
            "carefully and identify the input format and the task. Read the feedback and "
            "identify the mistakes and any domain-specific facts or strategies that would "
            "help; include them in the instruction, because the assistant will not see these "
            "examples again. Keep the instruction general enough for all inputs of this kind."
        )

    def crossover_prompt(self, first: str, second: str) -> str:
        return (
            "Two instructions for the same task each solve some problems the other misses.\n\n"
            f"Instruction A:\n```\n{first}\n```\n\nInstruction B:\n```\n{second}\n```\n\n"
            "Write one instruction that combines the useful guidance of both without "
            "contradictions."
        )

    def _search(self, ctx: OptimizerContext, seed_program: PromptProgram) -> None:
        pool: dict[str, PromptProgram] = {}
        per_instance: dict[str, list[float]] = {}

        def admit(program: PromptProgram, note: str = "accepted") -> None:
            rollouts = ctx.val_rollouts(program, note=note)
            pool[program.fingerprint()] = program
            per_instance[program.fingerprint()] = rollouts.scores

        admit(seed_program)
        for iteration in range(1, self.iterations + 1):
            ctx.step += 1
            if not ctx.affordable(2 * self.minibatch):
                break
            batch = ctx.sample(ctx.trainset, self.minibatch)
            survivors = remove_dominated(per_instance)
            if (
                self.crossover_every
                and iteration % self.crossover_every == 0
                and len(survivors) > 1
            ):
                first, second = ctx.rng.sample(sorted(survivors), 2)
                parent = pool[first]
                prompt = self.crossover_prompt(parent.instruction, pool[second].instruction)
                parent_score = max(
                    ctx.score_batch(pool[f], batch, note="parent").mean for f in (first, second)
                )
                note = "crossover"
            else:
                parent = pool[select_candidate(per_instance, ctx.rng)]
                result = ctx.score_batch(parent, batch, note="parent")
                parent_score = result.mean
                prompt = self.reflection_prompt(
                    parent.instruction,
                    format_rollouts(result.rollouts, ctx.task, limit=self.minibatch),
                    ctx.task.contract,
                )
                note = "reflect"
            texts = ctx.propose_many(_REFLECT, [prompt], purpose="reflect")
            if not texts or not texts[0] or texts[0] == parent.instruction:
                continue
            child = parent.with_instruction(texts[0])
            if child.fingerprint() in pool:
                continue
            child_score = ctx.score_batch(child, batch, note=note).mean
            if child_score <= parent_score:
                continue
            if not ctx.affordable(ctx.unseen_val_cost(child)):
                break
            admit(child)
