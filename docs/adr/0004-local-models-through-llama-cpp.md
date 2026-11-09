# ADR-0004: Run real models locally through llama.cpp with batched async requests

- **Status:** Accepted
- **Date:** 2026-11-08

## Context

The deeper study needs real language models for every optimizer and benchmark, with no API
keys and an 8 GB Apple M1 shared with other work. Prompt optimization is dominated by many
short, independent rollouts (one per example per candidate prompt), so throughput matters
far more than single-request latency.

## Decision

- Serve Qwen3.5 0.8B, 2B and 4B instruction models (Unsloth Q4_K_M GGUF builds) with
  `llama-server` from llama.cpp, one model in memory at a time.
- Talk to it through the OpenAI-compatible chat endpoint with an `httpx.AsyncClient`;
  `LlamaCppLM.generate` sends a whole batch concurrently (bounded by `concurrency`) and the
  server's continuous batching packs the in-flight sequences into shared forward passes.
- Disable Qwen3.5 thinking at the server (`--reasoning off`): every method then pays for
  the same visible reasoning, and hidden thinking tokens do not inflate the cost columns.
- Size the server to the machine: 8 slots sharing one unified 16,384-token KV pool (`-kvu`;
  each request still fits the longest prompt plus answer measured, 2,172 tokens), a
  128-token micro-batch (`-ub 128`), flash attention, a q8_0 KV cache and no host-RAM
  prompt cache (`--cache-ram 0`). Earlier runs used 8 slots of 4,096 tokens each. The 2B server then needs about 0.5 GB beyond
  its memory-mapped weights, which keeps the shared machine out of swap. The first runs
  used 12 to 24 slots of 3,072 tokens; the slot count changes speed, and only through
  llama.cpp's batch-composition nondeterminism could it change an answer.
- Optimize with the 2B model; score the final prompts on 0.8B for transfer. The 4B model is
  supported, and its transfer runs are left for a later release because they did not fit
  this release's compute budget.

## Alternatives considered

- **transformers on MPS.** Simpler packaging, but no continuous batching and higher memory
  per sequence; measured throughput was far lower for many concurrent short generations.
- **Thinking on.** Qwen3.5 reasons longer with thinking enabled, which would multiply the
  cost of every rollout by several times on hardware that is already the bottleneck.
- **A larger optimization model.** The 4B model is about twice as slow per token; using it
  for every rollout would have cut the number of methods and benchmarks covered.

## Consequences

- Aggregate throughput on the shared M1 was 30 to 80 completion tokens per second for the
  2B model depending on what else ran; this, not code speed, bounds every experiment size,
  and the evaluation report states the scaled-down sizes.
- The backend is a narrow protocol (`LanguageModel.generate`), so the fake model, DSPy and
  any OpenAI-compatible server plug in without changes to the optimizers.
