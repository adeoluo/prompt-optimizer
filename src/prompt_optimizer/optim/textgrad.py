"""TextGrad-style optimization with textual gradients (Yuksekgonul et al., 2024).

The instruction is the optimized variable. One iteration is a forward pass,
a textual backward pass and a textual gradient-descent step:

* **Forward.** Run the current program on a training minibatch. The *loss* is
  the task's natural-language feedback on each wrong answer
  (``Task.feedback``), the textual analogue of a per-example loss.
* **Backward.** The critic model reads the instruction and the failed
  examples with their feedback and writes a *gradient*: specific criticism of
  how the instruction caused the errors. It is asked not to rewrite the
  instruction, only to explain what to change (TextGrad's separation of
  ``backward`` from ``step``).
* **Step (TGD).** The optimizer model applies the gradient to the instruction.
  Like TextGrad's momentum option, the last ``momentum`` accepted versions and
  their scores are shown so the step does not oscillate.
* **Validation gate.** The new instruction is scored on the same minibatch;
  only if it is at least as good there is it scored on the full validation set,
  and it becomes the current variable only if its validation score improves
  (the validation-gated loop of the TextGrad prompt-optimization experiments).
"""

from __future__ import annotations

from dataclasses import dataclass

from prompt_optimizer.optim.base import Optimizer, OptimizerContext, format_rollouts
from prompt_optimizer.program import PromptProgram

_CRITIC = (
    "You are part of an optimization system that improves the system prompt of a language "
    "model. You give feedback (a textual gradient) on the system prompt. Do not write a new "
    "prompt; explain concisely how the current prompt led to the errors and what should change."
)
_STEP = (
    "You are part of an optimization system that improves text (a variable) given feedback. "
    "Reply with the improved variable inside <instruction></instruction> tags."
)


@dataclass
class TextGrad(Optimizer):
    """Textual gradient descent on the instruction with a validation gate."""

    iterations: int = 8
    minibatch: int = 10
    max_failures: int = 4
    momentum: int = 2
    name: str = "textgrad"

    def gradient_prompt(self, instruction: str, failures: str, contract: str) -> str:
        return (
            f"System prompt (the variable):\n<VARIABLE>\n{instruction}\n</VARIABLE>\n\n"
            f"The system prompt is always followed by this fixed line: {contract}\n\n"
            "The model answered the following examples incorrectly:\n\n"
            f"{failures}\n\n"
            "Give feedback on the system prompt: what about it led to these mistakes, and how "
            "should it change so the model gets such examples right?"
        )

    def step_prompt(self, instruction: str, gradient: str, past: list[tuple[str, float]]) -> str:
        history = "\n\n".join(
            f"Earlier version (validation accuracy {score:.2f}):\n{text}" for text, score in past
        )
        memory = f"\n\nPast versions of the variable:\n{history}" if history else ""
        return (
            f"Variable to improve (a system prompt):\n<VARIABLE>\n{instruction}\n</VARIABLE>\n\n"
            f"Feedback on the variable:\n<FEEDBACK>\n{gradient}\n</FEEDBACK>{memory}\n\n"
            "Rewrite the variable to address the feedback. Keep what works, keep it general "
            "(it must work for any problem of this kind, not only these examples), and do not "
            "include the answer line format, which is added separately."
        )

    def _search(self, ctx: OptimizerContext, seed_program: PromptProgram) -> None:
        current = seed_program
        current_val = ctx.score_val(current)
        accepted: list[tuple[str, float]] = [(current.instruction, current_val)]
        for _ in range(self.iterations):
            ctx.step += 1
            if not ctx.affordable(2 * self.minibatch):
                break
            batch = ctx.sample(ctx.trainset, self.minibatch)
            forward = ctx.score_batch(current, batch, note="forward")
            failures = [r for r in forward.rollouts if r.score < 1.0]
            if not failures:
                continue
            critique = ctx.ask(
                _CRITIC,
                self.gradient_prompt(
                    current.instruction,
                    format_rollouts(failures, ctx.task, limit=self.max_failures),
                    ctx.task.contract,
                ),
                purpose="critique",
            )[0]
            past = accepted[-self.momentum - 1 : -1] if self.momentum else []
            proposals = ctx.propose_many(
                _STEP, [self.step_prompt(current.instruction, critique, past)]
            )
            if not proposals or not proposals[0] or proposals[0] == current.instruction:
                continue
            candidate = current.with_instruction(proposals[0])
            after = ctx.score_batch(candidate, batch, note="step").mean
            if after < forward.mean:
                continue
            if not ctx.affordable(ctx.unseen_val_cost(candidate)):
                break
            score = ctx.score_val(candidate, note="accepted-on-batch")
            if score > current_val:
                current, current_val = candidate, score
                accepted.append((current.instruction, score))
