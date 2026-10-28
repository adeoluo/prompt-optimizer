# Evaluation Report - Prompt Optimizer

> Reproduce with `python -m prompt_optimizer evaluate`. Deterministic and offline.

## Setup

- **Task:** three-class intent classification (refund, shipping, cancel).
- **Splits:** disjoint training (9), validation (4), test (6), sharing vocabulary.
- **Metric:** exact-match accuracy.
- **Methods:** a naive zero-shot baseline, instruction search, few-shot bootstrap, and the
  compound optimizer; all compiled on training and validation, reported on test.

## Results

| Method | demonstrations | test accuracy |
|---|---:|---:|
| baseline (naive, zero-shot) | 0 | 0.333 |
| instruction search | 0 | 0.833 |
| few-shot bootstrap | 3 | 0.833 |
| compound (instruction + few-shot) | 1 | 1.000 |

## Interpretation

The naive baseline, with no label vocabulary and no demonstrations, must guess from keywords
and scores 0.333. Declaring the vocabulary (instruction search) or supplying demonstrations
(few-shot) each lifts accuracy to 0.833. The compound optimizer combines a vocabulary-bearing
instruction with a single well-chosen demonstration and reaches 1.000, beating either
component. The two search dimensions are complementary - the central claim of the project,
measured on held-out data.

## Reproducibility

Everything is offline and deterministic; the numbers regenerate exactly.
