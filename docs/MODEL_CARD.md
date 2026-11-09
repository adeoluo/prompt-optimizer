# Model Card: Prompt Optimizer

This project does not train model weights. It optimizes *prompts* (an instruction plus
in-context demonstrations) for existing instruction models, and its artifacts are optimized
prompts with their search histories (`benchmarks/results/runs/`).

## Models used

| Model | Build | File size | Role |
|---|---|---:|---|
| Qwen/Qwen3.5-0.8B (Feb 2026) | unsloth/Qwen3.5-0.8B-GGUF, Q4_K_M | 533 MB | transfer evaluation |
| Qwen/Qwen3.5-2B (Feb 2026) | unsloth/Qwen3.5-2B-GGUF, Q4_K_M | 1.28 GB | optimization and evaluation |
| Qwen/Qwen3.5-4B (Feb 2026) | unsloth/Qwen3.5-4B-GGUF, Q4_K_M | 2.74 GB | supported for transfer; not run in this release |

- **License:** Apache 2.0 (model and GGUF builds).
- **Serving:** llama.cpp `llama-server` (Homebrew build 9.0, b12110), Metal, flash attention,
  q8_0 KV cache, thinking disabled (`--reasoning off`), greedy decoding (temperature 0) for
  task rollouts, temperature 0.9 to 1.0 with per-request seeds for proposer calls.
- **Hardware:** Apple M1, 8 cores, 8 GB unified memory, shared with other jobs.

## Intended use

- Research and engineering comparison of prompt-optimization methods under equal budgets.
- A readable, tested reference implementation of APE, OPRO, MIPROv2-style, TextGrad-style,
  GEPA-style, PromptBreeder-style and bootstrapped few-shot optimization.

## Out of scope

- Optimized prompts are selected for one model and one benchmark split; the transfer
  results show how much carries over to other model sizes, and it is not guaranteed.
- Absolute accuracies are for 4-bit quantized small models without thinking and are not
  comparable to published numbers for full-precision or larger models.

## Risks

- Prompt optimization can overfit the validation set; the evaluation report shows the
  validation to test gap for every selected prompt.
- Proposer models can write instructions that leak answers seen in feedback. Proposers see
  training and validation examples only; the test split is never passed to an optimizer.

## Reference pipeline

The original simulated language model (`lm.py`) and its three optimizers remain as a
deterministic, offline reference with their own small classification task.
