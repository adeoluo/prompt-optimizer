# ADR-0003: Compile prompts against a held-out validation metric

- **Status:** Accepted
- **Date:** 2026-01-01

## Context

The optimizers must decide which instruction and which demonstrations to keep. Scoring
candidates on the same data they are drawn from would reward memorization rather than
generalization.

## Decision

Score every candidate prompt on a held-out validation split, and report final numbers on a
separate test split. Few-shot selection is greedy forward selection against the validation
metric; instruction search picks the best-scoring candidate; the compound optimizer
alternates the two by coordinate ascent.

## Alternatives considered

- **Score on the training set.** Simpler, but overfits and does not predict held-out
  accuracy.
- **Exhaustive search over demonstration subsets.** Optimal but exponential; greedy forward
  selection is a standard, near-linear approximation.

## Consequences

- Compiled prompts are selected for generalization and reported on truly held-out data.
- Greedy selection can stop early when validation saturates; the validation split is sized
  to exercise the vocabulary the test split needs.
- The whole procedure is deterministic and reproducible.
