# Data Card - Prompt Optimizer

The project bundles one small, synthetic intent-classification task. There is no training of
a model and nothing is downloaded.

## bundled task (`src/prompt_optimizer/data.py`)

- **Provenance:** hand-authored for this project. Short customer messages labeled as
  ``refund``, ``shipping``, or ``cancel``.
- **Splits:** disjoint training (9), validation (4), and test (6) sets that share vocabulary,
  so a prompt compiled on training and validation generalizes to test.
- **Candidate instructions:** four hand-authored instructions, two of which declare the label
  vocabulary.
- **Licensing:** authored by this repository; MIT.
- **Known gaps:** tiny, English, three classes; a demonstration set, not a benchmark.

## Ethical considerations

The task is synthetic and contains no personal, sensitive, or copyrighted content.
