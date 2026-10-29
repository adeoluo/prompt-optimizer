# Scaling and Design Tradeoffs - Prompt Optimizer

How the reference design grows, and the tradeoffs taken.

## Reference architecture (today)

A single process compiles the prompt once at startup and serves predictions from it over
a standard-library HTTP service. Shared-nothing and stateless per request.

## Scaling path

| Stage | Bottleneck | Change |
|---|---|---|
| One process | Startup compile time | Compile offline; ship the prompt as config |
| One host | Prediction CPU (already cheap) | Replicas for availability, not throughput |
| Larger search | Compile cost grows with candidates and training size | Cache validation scores; parallelize candidate evaluation |
| Real model | Per-prediction latency | Batch and cache model calls; keep the prompt compiled |

## Substantial tradeoffs

- **Compile-time search vs runtime prompting.** Compiling once makes prediction nearly
  free and the served prompt auditable, at the cost of recompiling when data changes.
  Prompting a model per request is more flexible but pays a model call every time. The
  reference chooses the compiled path and exposes `/compile` for inspection.
- **Simulated model vs a hosted model.** The deterministic simulated model keeps the whole
  pipeline reproducible and dependency-free; a hosted model generalizes but adds latency,
  cost, and nondeterminism. The predictor is the single seam to swap it in.
- **Greedy few-shot selection vs exhaustive search.** Greedy forward selection is near
  linear and good enough; exhaustive subset search is optimal but exponential. The budget
  bounds compile cost explicitly.
- **Validation-driven selection vs training fit.** Scoring candidates on held-out data
  costs a validation pass per candidate but prevents overfitting; the reference pays that
  cost for generalization.

## What would change under real load

Move compilation offline into a build step that emits the prompt as versioned
configuration, run stateless replicas for availability, and, if a real model replaces the
simulator, batch and cache its calls behind the predictor. The optimizers are unchanged.
