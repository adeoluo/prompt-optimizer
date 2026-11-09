"""APE: Automatic Prompt Engineer (Zhou et al., 2022).

APE treats the instruction as a latent variable and searches it by
generate, score, resample:

1. **Forward generation.** Show the proposer a few input/output pairs from the
   training set and ask which instruction produces them. Each of
   ``n_proposals`` proposals sees a different random set of pairs.
2. **Score.** Execution accuracy of each proposal on one shared training
   minibatch (the same minibatch for every candidate, so scores are comparable).
3. **Iterative Monte Carlo search.** Keep the top ``top_k`` and ask the
   proposer for ``n_variants`` semantically equivalent paraphrases of each;
   score them on the same minibatch.
4. **Select.** The ``n_finalists`` best by minibatch score are scored on the full
   validation set; the best validation score wins.

The paper's resampling prompt ("Generate a variation of the following
instruction while keeping the semantic meaning") is used verbatim in spirit:
the variant is a paraphrase, not a new idea, which is the difference from OPRO.
"""

from __future__ import annotations

from dataclasses import dataclass

from prompt_optimizer.optim.base import Optimizer, OptimizerContext, clip, reference_output
from prompt_optimizer.program import PromptProgram

_SYSTEM = (
    "You write instructions for a language model. Reply with one instruction inside "
    "<instruction></instruction> tags and nothing else."
)


@dataclass
class APE(Optimizer):
    """Instruction induction from demonstrations plus Monte Carlo resampling."""

    n_proposals: int = 8
    pairs_per_proposal: int = 4
    minibatch: int = 16
    top_k: int = 2
    n_variants: int = 2
    n_finalists: int = 2
    name: str = "ape"

    def induction_prompt(self, ctx: OptimizerContext) -> str:
        pairs = ctx.sample(ctx.trainset, self.pairs_per_proposal)
        blocks = [
            f"Input:\n{clip(ex.question, 700)}\n"
            f"Output:\n{clip(reference_output(ctx.task, ex), 500)}"
            for ex in pairs
        ]
        return (
            "I gave a friend an instruction. Based on the instruction they produced the "
            "following input/output pairs:\n\n"
            + "\n\n".join(blocks)
            + "\n\nThe outputs must also end with this line format, which is fixed and not part "
            f"of the instruction: {ctx.task.contract}\n\nWhat was the instruction? Write it "
            "so that another model following it would solve new inputs of this kind correctly."
        )

    def variant_prompt(self, instruction: str) -> str:
        return (
            "Generate a variation of the following instruction while keeping the semantic "
            f"meaning.\n\nInput: {instruction}\n\nOutput the variation only."
        )

    def _search(self, ctx: OptimizerContext, seed_program: PromptProgram) -> None:
        reserve = self.n_finalists * ctx.val_cost
        batch = ctx.sample(ctx.trainset, self.minibatch)
        scored: dict[str, float] = {}

        def score(instructions: list[str], note: str) -> None:
            fresh = [s for s in dict.fromkeys(instructions) if s not in scored]
            fresh = fresh[: ctx.cap(len(fresh) * len(batch), reserve) // max(1, len(batch))]
            programs = [seed_program.with_instruction(s) for s in fresh]
            for text, result in zip(
                fresh, ctx.score_batch_many(programs, batch, note), strict=True
            ):
                scored[text] = result.mean

        prompts = [self.induction_prompt(ctx) for _ in range(self.n_proposals)]
        proposals = [p for p in ctx.propose_many(_SYSTEM, prompts) if p]
        score([seed_program.instruction, *proposals], "induced")
        ctx.step += 1
        parents = sorted(scored, key=lambda s: -scored[s])[: self.top_k]
        prompts = [self.variant_prompt(p) for p in parents for _ in range(self.n_variants)]
        score([p for p in ctx.propose_many(_SYSTEM, prompts, purpose="mutate") if p], "variant")
        ctx.step += 1
        finalists = sorted(scored, key=lambda s: -scored[s])[: self.n_finalists]
        ctx.score_finalists([seed_program.with_instruction(s) for s in finalists])
