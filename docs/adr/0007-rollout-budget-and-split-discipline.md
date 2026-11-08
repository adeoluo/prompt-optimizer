# ADR-0007: Equal rollout budgets and strict train, validation and test splits

- **Status:** Accepted
- **Date:** 2026-11-08

## Context

Prompt optimizers differ in how much they spend: some score many candidates on small
minibatches, others few candidates on the full validation set, and some also spend model
calls on proposals and critiques. Comparisons are only meaningful at equal spend, and a
validation score that selected the prompt is an optimistic estimate of its accuracy.

## Decision

- The budget is a cap on *task rollouts* (one model call on one train or validation
  example under one candidate program). Every method, including DSPy GEPA through
  `max_metric_calls`, gets the same cap (400 for the GSM8K and BIG-Bench Hard runs).
  Proposer, critique and reflection calls are not capped but are counted and reported.
- `Evaluator` memoizes (program, example) pairs within a run and charges only new pairs;
  a batch that would overrun the cap is refused before it is sent.
- Splits are fixed by seed: train (demonstration pool, minibatches, reflection), validation
  (selection), test (reporting only; no optimizer ever receives it). BIG-Bench Hard is split
  per subtask so every subtask appears in every split.
- Test examples are scored in one fixed seeded order (subtasks interleaved) and a run may
  score a prefix, so a smaller test result is always a subset of the larger one.
- The overfitting analysis compares each selected program's validation score with its test
  score.

## Alternatives considered

- **Equal wall-clock time.** Depends on server load from other jobs on the shared machine.
- **Equal total tokens.** Penalizes methods whose prompts make the model write longer
  solutions, which is a property of the prompt, not of the search.

## Consequences

- Budgets are small next to the papers (which often use thousands of rollouts); the
  evaluation report states this and shows where methods stopped on budget.
- Test-set sizes are stated per table; intervals come from the same examples (ADR-0010).
