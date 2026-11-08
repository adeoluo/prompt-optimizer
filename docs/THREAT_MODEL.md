# Threat Model: Prompt Optimizer

Mapped to the OWASP Top 10 for LLM Applications 2026 (IDs LLM01:2026 to LLM10:2026).

## Assets

- Integrity of the optimized prompts and of every reported number (results files, tables).
- The local model server and the machine it shares with other work.
- The response cache, which is the store every result can be replayed from.

## Trust boundaries

- **Benchmark text, model outputs and proposer outputs are untrusted text.** They are
  rendered into prompts, parsed with regular expressions and compared; nothing is executed
  or evaluated, and no tool calls are made.
- **The model server** (`llama-server`) binds to 127.0.0.1 without an API key; it must not be
  exposed beyond the local host.
- **Downloads** (datasets, GGUF weights, Python packages) cross the network boundary once.

## Threats and mitigations

| OWASP 2026 | Threat | Vector | Mitigation |
|---|---|---|---|
| LLM01:2026 Prompt Injection | Injected text in a training example or HotpotQA paragraph steers the proposer into writing a harmful or answer-leaking instruction | Train examples and rollouts appear inside meta-prompts (APE, OPRO, TextGrad, GEPA, PromptBreeder) | Proposals are plain text that only enters the system message; the output contract is fixed (ADR-0005); proposals are capped at 2,000 characters; a proposal is kept only if it scores better on validation |
| LLM04:2026 Supply Chain | Tampered dataset, model weights or package | Downloads | Dataset manifest with SHA-256 digests; GGUF builds from named Hugging Face repositories with recorded revisions; locked dependencies (`uv.lock`); CodeQL on main |
| LLM05:2026 Data and Model Poisoning | Poisoned training examples become bootstrapped demonstrations | Teacher traces | Only traces the metric scores as correct (and untruncated) are kept; validation decides selection; the test split is never visible to optimizers |
| LLM06:2026 Unbounded Consumption | A search loop or long generations exhaust the shared machine | Proposer loops, runaway outputs | Hard rollout budget checked before every batch; `max_tokens` per task and per proposer call; bounded concurrency; one model in memory at a time |
| LLM07:2026 Misinformation | Overstated results | Selecting the best of many noisy validation scores | Separate test split; 95% bootstrap intervals and paired tests; validation to test gap reported for every method (ADR-0007, ADR-0010) |
| LLM10:2026 Improper Output Handling | Model output treated as code or markup | Answer extraction | Outputs are only parsed by regular expressions and string normalization; no `eval`, no shell, no HTML rendering |

LLM02 (sensitive information), LLM03 (excessive agency), LLM08 (hidden context exposure)
and LLM09 (vector and embedding weaknesses) have no meaningful surface here: there are no
secrets or personal data in prompts, no tools, no hidden system context beyond the published
prompts, and no retrieval index.

## Residual risk

- An optimized instruction can memorize validation-specific facts; this shows up as a larger
  validation to test gap and is reported, not prevented.
- Contamination of public benchmarks in the models' pretraining cannot be ruled out; relative
  comparisons between prompts on the same model are the reported quantity.
