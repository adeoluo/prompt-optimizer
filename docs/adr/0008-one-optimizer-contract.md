# ADR-0008: Seven optimizers behind one contract, each faithful to its paper's core loop

- **Status:** Accepted
- **Date:** 2026-11-08

## Context

The literature methods (APE, OPRO, MIPROv2, TextGrad, GEPA, PromptBreeder and bootstrapped
few-shot) were published with different program representations, model APIs and budgets.
A fair, readable comparison needs them in one code base with one interface.

## Decision

All optimizers subclass `Optimizer` and implement `_search(ctx, seed_program)`. The shared
`OptimizerContext` provides budgeted, batched scoring (`score_batch_many`,
`score_val_many`, `score_finalists`), batched proposer calls and a seeded RNG. Each method
keeps its paper's characteristic loop and states its scaled-down choices in its docstring:

- **APE:** forward instruction induction from demonstrations, shared-minibatch scoring,
  one round of paraphrase resampling (iterative Monte Carlo), finalists on validation.
- **OPRO:** meta-prompt with the scored trajectory in ascending order and task exemplars,
  several samples per step at temperature 1, fixed training subset for comparable scores.
- **MIPROv2:** bootstrapped demo-set candidates, grounded proposals (dataset summary, task
  description, traces, previous proposals, random tip), a from-scratch categorical TPE
  over (instruction, demo set) on validation minibatches, periodic full evaluations.
- **TextGrad:** forward on a minibatch, textual gradient from failures with feedback,
  gradient step with momentum over accepted versions, validation-gated acceptance.
- **GEPA:** per-instance Pareto pool, frequency-weighted parent sampling after removing
  dominated candidates, reflective mutation on 3-example minibatches, strict minibatch
  gate before a full validation pass. Module-wise merge does not apply to a one-module
  program; an optional crossover reflection replaces it.
- **PromptBreeder:** units of task prompt plus mutation prompt, binary tournament, the
  paper's operator classes (zero and first order, EDA, lineage, hypermutation, Lamarckian,
  crossover), with original short thinking-style and mutation-prompt lists. One task
  prompt per unit instead of two; few-shot context evolution left to other methods.
- **Bootstrapped few-shot:** teacher traces kept when correct and untruncated, DSPy's
  candidate order (unshuffled, labeled, random sizes), selection on validation.

The original three optimizers on the simulated classifier stay as the readable reference.

## Alternatives considered

- **Wrap the authors' code.** Each library assumes its own program model and client;
  wrapping would compare integrations, not algorithms. DSPy is compared directly instead.

## Consequences

- One test suite with a deterministic fake model checks every method against the same
  contract (budget, determinism, selection by validation, test split never seen).
- Faithfulness is to the algorithmic core; prompts are paraphrased, not copied.
