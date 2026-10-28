# System Card - Prompt Optimizer

## Overview

- **What it is:** an automated prompt optimizer - it compiles a prompt (instruction plus
  demonstrations) against a metric by searching over instructions and demonstrations.
- **What it is not:** a language model. It ships a deterministic simulated model to make the
  optimization reproducible; a real model drops in behind the predictor.
- **Version:** 0.1.0.

## Intended use

- **Primary use cases:** learning and demonstrating prompt and few-shot optimization; a
  from-scratch reference for metric-driven prompt compilation.
- **Primary users:** engineers and researchers building prompt programs.

## Out-of-scope / prohibited use

- Absolute accuracy reflects the simulated model, not any hosted model.
- The bundled task is a demonstration, not a benchmark.

## Performance

See the README ablation: baseline 0.333, instruction search 0.833, few-shot 0.833, compound
1.000 on held-out test data. The meaningful quantity is the relative lift.

## Risks and limitations

- Greedy few-shot selection can overfit a small validation split.
- The instruction candidate set is enumerated, not generated.

## Safety mitigations

- Deterministic, offline, reproducible; held-out validation and test splits.

## Environmental / cost footprint

Negligible; all in-memory with no model calls.
