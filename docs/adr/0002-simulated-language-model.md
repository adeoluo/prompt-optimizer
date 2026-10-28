# ADR-0002: A deterministic simulated language model

- **Status:** Accepted
- **Date:** 2026-01-01

## Context

Optimizing a prompt requires a model whose output depends on the prompt, so that searching
over instructions and demonstrations has a real signal. A real model would add a network
dependency and nondeterminism, making the test suite flaky and the results impossible to
reproduce.

## Decision

Ship a deterministic, offline simulated language model that reproduces the two behaviours
prompt optimization relies on: in-context learning (copy the nearest demonstration's label)
and instruction following (read the declared label vocabulary and snap to it). Both
behaviours are generic, not tuned to the bundled task.

## Alternatives considered

- **Call a hosted model.** Realistic, but nondeterministic, networked, and not reproducible.
- **A static lookup table.** Deterministic, but its output would not respond to the prompt,
  so optimization would have nothing to optimize.

## Consequences

- The pipeline is fully offline, deterministic, and reproducible.
- A real model drops in behind the predictor without changing the optimizers.
- Absolute accuracy numbers reflect the simulation, not any particular hosted model; the
  meaningful quantity is the relative lift from optimization.
