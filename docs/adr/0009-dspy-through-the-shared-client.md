# ADR-0009: Run DSPy through the same cached, metered client

- **Status:** Accepted
- **Date:** 2026-11-08

## Context

The head-to-head with DSPy must use the same model, server, splits and budget, and report
cost in the same units. DSPy normally calls models through LiteLLM with its own cache.

## Decision

`benchmarks/dspy_compare.py` defines `RoutedLM`, a `dspy.BaseLM` whose `forward` turns
DSPy's chat messages into `LMRequest`s and answers them through this repository's
`CachedLM`. DSPy's task, proposal and reflection models are three `RoutedLM` instances
with different accounting labels. DSPy programs are `ChainOfThought` with the same seed
instruction; their `answer` field is scored by the same `Task.score`. GEPA gets
`max_metric_calls` equal to the in-repo budget; MIPROv2 and random search are configured
to comparable trial counts and their actual task calls are reported.

## Alternatives considered

- **LiteLLM to the same server with DSPy's cache.** Simpler, but cost accounting and
  resumability would differ from the in-repo methods.

## Consequences

- DSPy and in-repo rows share one cost ledger and one cache.
- DSPy 3.4 marks the `forward` subclass interface as deprecated (removal planned for 3.5);
  the adapter will need the engine interface when the comparison is rerun on DSPy 3.5.
