# Threat Model - Prompt Optimizer

## Assets

- Integrity of the compiled prompt and the reported metrics.

## Trust boundaries

- **Instructions, demonstrations, and queries are untrusted text.** They are rendered and
  compared, never executed; there is no path from text to execution.

## Threats and mitigations

| Threat | Vector | Impact | Mitigation |
|---|---|---|---|
| Overfitting the metric | Selection on data used for reporting | Inflated accuracy | Disjoint validation and test splits |
| Nondeterministic results | A networked model | Irreproducible numbers | Deterministic, offline simulated model |
| Poisoned demonstrations | Malicious training examples | Skewed prompt | Selection is read-only; callers control the training set |
| Supply-chain compromise | Malicious dependency | Code execution | Zero runtime dependencies; Dependabot and CodeQL in continuous integration |

## Residual risk

The simulated model is not a real model; absolute numbers do not transfer. The reproducible,
meaningful quantity is the relative improvement from optimization.
