# Operations Guide - Prompt Optimizer

The operational contract of the prediction service.

## Running the service

```bash
uv run python -m prompt_optimizer serve --host 0.0.0.0 --port 8000
```

A standard-library HTTP server with **zero runtime dependencies**. The prompt is
**compiled once at startup** (the expensive step) and then serves cheap predictions.

## Endpoints

| Method | Path | Body | Success |
|---|---|---|---|
| GET | `/health` | none | status, instruction, demonstration count |
| POST | `/predict` | `{"text"}` | `{"text", "label"}` |
| POST | `/compile` | none | the compiled instruction and demonstrations |

Errors return JSON `{"error": ...}` with a 4xx status.

## Service level objectives

Measured by `benchmarks/stress.py` on a developer laptop:

| Objective | Target | Observed |
|---|---|---|
| Compile time (startup) | < 50 ms | a few milliseconds |
| Predict latency p99 | < 1 ms | about 0.01 ms |
| Throughput | > 10000 predictions/second/core | about 100000 predictions/second |

Illustrative targets for the reference implementation.

## Operational constraints

- **Compilation is a startup cost, not a request cost.** The optimized prompt is built
  once; prediction is a nearest-neighbour lookup over a handful of demonstrations, so it
  is effectively free. Recompilation (the `/compile` endpoint or a restart) is the only
  expensive path and should be rate limited or run offline.
- **The prediction path is CPU bound and holds the global interpreter lock.** Throughput
  per process is already high; scale with processes if a single core saturates.
- **The compiled prompt is immutable after startup.** New training data requires a
  recompile and a restart or blue/green swap.
- **Inputs are bounded** to 2000 characters. **No persistence, no authentication**; front
  the service with a gateway.

## Failure modes and responses

| Symptom | Likely cause | Response |
|---|---|---|
| Slow startup | Large candidate or training set | Compile offline; ship the prompt as config |
| Unexpected label | Query far from any demonstration | Expected; widen the training set and recompile |
| `400` on input | Empty or overlong text | Inspect the `error` field |

## Capacity planning

A single core serves on the order of 100000 predictions per second. Provision replicas
for availability and redundancy rather than throughput; the service is stateless.
