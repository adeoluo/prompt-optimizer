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

### Length clause (added after the first real runs)

The GSM8K and BIG-Bench Hard contracts also ask for brief reasoning ("in a few short steps
(no more than about 120 words)") with a 384-token cap. Without it, Qwen3.5-2B wrote 295
(GSM8K) and 355 (BIG-Bench Hard) completion tokens per answer on average, and 39% of
BIG-Bench Hard answers hit a 512-token cap before reaching an answer, so the comparison
would have mostly measured verbosity, and every run cost about three times as much on the
shared machine. Measured on the seed prompt and the same validation examples:

| Benchmark | Contract | Accuracy | Completion tokens per answer | Truncated |
|---|---|---:|---:|---:|
| GSM8K (36 validation problems) | answer line only, 512 tokens | 86.1% | 295 | 2 |
| GSM8K (same 36) | with length clause, 384 tokens | 80.6% | 111 | 0 |
| BIG-Bench Hard (54 validation examples) | answer line only, 512 tokens | 44.4% | 355 | 18 |
| BIG-Bench Hard (same 54) | with length clause, 384 tokens | 61.1% | 117 | 4 |

The clause is part of the fixed contract for every method; for DSPy it is the description of
the `reasoning` field, which DSPy optimizers do not rewrite. Reproduce the table with
`python benchmarks/contract_probe.py` (replays from the response cache).

## Alternatives considered

- **Let the optimizer own the whole system prompt.** Closer to some papers, but conflates
  format compliance with task skill and makes small-model results noisy.
- **A larger token cap instead of a length clause.** Removes truncation but multiplies the
  cost of every rollout; the measured runs could not have covered seven methods.
- **Constrained decoding of the answer.** Removes parse failures entirely but changes the
  generation process in a way none of the compared methods assume.

## Consequences

- Score differences between methods measure reasoning and instruction quality.
- DSPy formats prompts with its own adapter and field markers; its rows are scored on the
  same extracted answer, and its own zero-shot program is reported as a separate baseline.
