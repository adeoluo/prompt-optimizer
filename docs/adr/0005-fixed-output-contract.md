# ADR-0005: The output contract belongs to the task, not to the optimized prompt

- **Status:** Accepted
- **Date:** 2026-11-08

## Context

Every benchmark is scored by extracting a final answer from free-form model output. If an
optimizer could rewrite the answer format, a large share of any measured gain could come
from a higher parse rate rather than better reasoning, and methods would not be comparable.

## Decision

Each task owns a fixed *contract* (for example "End your response with a final line of the
form 'Answer: <number>'") that is appended to the system message after the optimized
instruction. Optimizers edit only the instruction and the demonstrations. Proposer prompts
show the contract and say it is fixed, so proposals do not restate or contradict it.
Answer extraction is shared by all methods, including the DSPy rows, whose `answer` field
is scored by the same `Task.score`.

## Alternatives considered

- **Let the optimizer own the whole system prompt.** Closer to some papers, but conflates
  format compliance with task skill and makes small-model results noisy.
- **Constrained decoding of the answer.** Removes parse failures entirely but changes the
  generation process in a way none of the compared methods assume.

## Consequences

- Score differences between methods measure reasoning and instruction quality.
- DSPy formats prompts with its own adapter and field markers; its rows are scored on the
  same extracted answer, and its own zero-shot program is reported as a separate baseline.
