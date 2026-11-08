"""MIPROv2-style joint instruction and demonstration search (Opsahl-Ong et al., 2024).

Three stages, following the paper and DSPy's ``MIPROv2``:

1. **Bootstrap demo-set candidates.** Teacher rollouts of the seed program on
   training data give correct traces; ``n_demo_sets`` candidate sets are built
   from them (set 0 is always the empty set, set 1 the labeled gold demos when
   the dataset has rationales, the rest random bootstrapped subsets).
2. **Grounded instruction proposals.** The proposer sees a dataset summary
   (written by the model from training examples), the task description, a few
   bootstrapped traces, the instructions proposed so far, and one randomly
   drawn *tip* ("be concise", "describe the persona", ...). The tips diversify
   the candidates; grounding keeps them on task. The seed instruction is
   candidate 0.
3. **Bayesian search.** A TPE sampler (``optim.tpe``) picks
   ``(instruction index, demo-set index)`` per trial; each trial is scored on a
   random validation minibatch (as DSPy does), and the TPE objective is that
   minibatch score. Every ``full_eval_every`` trials, and at the end, the
   configuration with the best running mean minibatch score is scored on the
   full validation set, and the best full score is returned.

Minibatch scores of the same configuration are averaged across trials, so a
lucky single minibatch does not decide the full evaluations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from prompt_optimizer.optim.base import Optimizer, OptimizerContext, clip
from prompt_optimizer.optim.demos import (
    bootstrap_traces,
    labeled_demos,
    sample_demo_set,
    teacher_pool,
)
from prompt_optimizer.optim.tpe import TPESampler
from prompt_optimizer.program import Demo, PromptProgram

TIPS: tuple[str, ...] = (
    "Keep the instruction clear and concise.",
    "Make the instruction specific and informative about how to reach the answer.",
    "Describe the expert persona that would solve these problems best.",
    "Mention the most common mistakes to avoid on this kind of problem.",
    "Ask for the reasoning to be checked before the final answer is given.",
    "Write the instruction as a short numbered procedure.",
    "Stress that the problem has a single correct answer and accuracy matters most.",
    "",
)

_SYSTEM = (
    "You propose instructions for a language model program. Reply with one instruction "
    "inside <instruction></instruction> tags."
)


@dataclass
class MIPROv2(Optimizer):
    """Bootstrapped demos, grounded instruction proposals and TPE search."""

    n_instructions: int = 6
    n_demo_sets: int = 5
    max_demos: int = 3
    n_teacher: int = 40
    n_trials: int = 12
    minibatch: int = 20
    full_eval_every: int = 6
    n_summary_examples: int = 8
    name: str = "miprov2"

    def summarize(self, ctx: OptimizerContext) -> str:
        examples = ctx.sample(ctx.trainset, self.n_summary_examples)
        listing = "\n\n".join(
            f"Input: {clip(ex.question, 400)}\nAnswer: {ex.answer}" for ex in examples
        )
        replies = ctx.ask(
            "You describe datasets in two or three sentences.",
            "Here are examples from a dataset:\n\n"
            f"{listing}\n\nSummarize the dataset: what the inputs are, what the answers look "
            "like, and what makes the problems hard.",
            purpose="summarize",
            temperature=0.0,
            max_tokens=200,
        )
        return replies[0].strip()

    def proposal_prompt(
        self,
        ctx: OptimizerContext,
        summary: str,
        traces: list[Demo],
        previous: list[str],
        tip: str,
    ) -> str:
        shown = ctx.rng.sample(traces, min(2, len(traces))) if traces else []
        demo_text = "\n\n".join(
            f"Input:\n{clip(d.input, 500)}\nOutput:\n{clip(d.output, 500)}" for d in shown
        )
        prior = "\n".join(f"- {p}" for p in previous[-6:])
        parts = [
            f"Dataset summary: {summary}",
            f"Task description: {ctx.task.description}",
            f"The fixed answer format (not part of the instruction): {ctx.task.contract}",
        ]
        if demo_text:
            parts.append(f"Examples of correct solutions:\n{demo_text}")
        if prior:
            parts.append(f"Instructions already proposed:\n{prior}")
        if tip:
            parts.append(f"Tip: {tip}")
        parts.append("Propose a new instruction that will make the model answer correctly.")
        return "\n\n".join(parts)

    def _search(self, ctx: OptimizerContext, seed_program: PromptProgram) -> None:
        trial_cost = self.minibatch
        reserve = 2 * ctx.val_cost
        pool = teacher_pool(ctx, self.n_teacher)
        traces = bootstrap_traces(ctx, seed_program, pool[: ctx.cap(len(pool), reserve)])
        demo_sets: list[tuple[Demo, ...]] = [()]
        labeled = labeled_demos(ctx, ctx.sample(ctx.trainset, 4 * self.max_demos))
        if labeled:
            demo_sets.append(sample_demo_set(ctx.rng, labeled, self.max_demos))
        while traces and len(demo_sets) < self.n_demo_sets:
            k = ctx.rng.randint(1, self.max_demos)
            demo_sets.append(sample_demo_set(ctx.rng, traces, k))
        ctx.step += 1

        summary = self.summarize(ctx)
        instructions = [seed_program.instruction]
        prompts = [
            self.proposal_prompt(ctx, summary, traces, instructions, ctx.rng.choice(TIPS))
            for _ in range(self.n_instructions - 1)
        ]
        for text in ctx.propose_many(_SYSTEM, prompts):
            if text and text not in instructions:
                instructions.append(text)
        ctx.step += 1

        def program(choice: dict[str, int]) -> PromptProgram:
            return PromptProgram(instructions[choice["instruction"]], demo_sets[choice["demos"]])

        sampler = TPESampler(
            {"instruction": len(instructions), "demos": len(demo_sets)}, seed=ctx.seed
        )
        sums: dict[tuple[int, int], float] = defaultdict(float)
        counts: dict[tuple[int, int], int] = defaultdict(int)

        def best_config() -> dict[str, int]:
            key = max(sums, key=lambda k: (sums[k] / counts[k], counts[k]))
            return {"instruction": key[0], "demos": key[1]}

        for trial in range(1, self.n_trials + 1):
            if not ctx.affordable(trial_cost, reserve):
                break
            choice = sampler.suggest()
            batch = ctx.sample(ctx.valset, self.minibatch)
            score = ctx.score_batch(program(choice), batch, "tpe", split="val-minibatch").mean
            ctx.step += 1
            sampler.observe(choice, score)
            key = (choice["instruction"], choice["demos"])
            sums[key] += score
            counts[key] += 1
            if trial % self.full_eval_every == 0:
                ctx.score_finalists([program(best_config())], note=f"trial-{trial}")
        if sums:
            ctx.score_finalists([program(best_config())], note="final")
