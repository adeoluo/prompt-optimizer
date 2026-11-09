# Evaluation Report: Prompt Optimizer

Seven prompt optimizers and two DSPy optimizers, compared on real local models and public
benchmarks under equal rollout budgets. Every number below is recomputed from the committed
per-example result files by `python -m prompt_optimizer report`.

## 1. Question

Given a small instruction model, a seed prompt and a fixed budget of task rollouts, how much
does each prompt-optimization method improve held-out accuracy, what does the search cost,
how much of the validation gain survives on test, and do the optimized prompts transfer to
smaller and larger models of the same family?

## 2. Setup

- **Hardware:** Apple M1 (8 cores, 8 GB unified memory), shared with another agent's
  PyTorch jobs during every run. Measured 2B throughput was about 45 completion tokens per second across 8 parallel requests (about 5.5 per request).
- **Serving:** llama.cpp `llama-server` 9.0 (build 12110), Metal, flash attention, q8_0 KV
  cache, no host-RAM prompt cache, thinking disabled. The seed-prompt runs used 12 slots of
  3,072 tokens; later runs used 8 slots, of 4,096 tokens each and then sharing one unified
  16,384-token cache, to leave memory for the other job (ADR-0004). Decoding is greedy, so
  slot settings change speed, not answers.
- **Models:** Qwen3.5-2B (optimization and evaluation); Qwen3.5-0.8B (transfer); Unsloth Q4_K_M GGUF builds. Task rollouts are greedy (temperature 0);
  proposer calls use temperature 0.9 (1.0 for OPRO, as in the paper) with per-request seeds.
- **Software:** Python 3.12, numpy 2.5, httpx 0.28, DSPy 3.4.1 (the latest release when run).
- **Benchmarks and splits** (seed 20260211; the test split is never given to an optimizer):

| Benchmark | Train | Validation | Test available | Test scored | Real-model results |
|---|---:|---:|---:|---:|---|
| GSM8K | 200 | 60 | 1,319 | 300 (seeded prefix) | yes: eight methods, DSPy, transfer |
| BIG-Bench Hard | 216 | 54 | 405 | 270 (10 per task) | yes: eight methods, transfer |
| MATH-500 | 200 | 100 | 500 |  | not yet (loader, scoring and tests) |
| HotpotQA (distractor) | 150 | 100 | 500 |  | not yet (loader, scoring and tests) |
| IFEval | 100 | 100 | 341 |  | not yet (loader, 25 checkers and tests) |

- **Budget:** 400 task rollouts per optimization run, counted on train and validation
  examples, memoized within a run, enforced before every batch. Proposer calls are counted
  and reported but not capped. The seed program's validation pass counts against the budget.
- **Seed prompt:** the task's one-line instruction (for example "Solve the math word
  problem.") plus the task's fixed output contract (ADR-0005).
- **Selection:** each method returns the program with its best full-validation score.
- **Test scoring:** a fixed seeded order of the test split (subtasks interleaved for
  BIG-Bench Hard); a table reports only the examples every row scored.
- **Statistics:** 95% percentile bootstrap intervals (10,000 resamples); paired bootstrap
  differences against the seed prompt with two-sided p-values (ADR-0010).

## 3. Method settings

| Method | Settings (defaults in the module) |
|---|---|
| Bootstrapped few-shot | 40 teacher rollouts, up to 4 demos per group, unshuffled, labeled and 6 random candidate sets |
| APE | 8 induction proposals from 4 pairs each, 16-example shared minibatch, 2 paraphrases of the top 2, 2 finalists |
| OPRO | 8 steps of 3 samples at temperature 1.0, 15-example fixed training subset, history of 20, 3 exemplars, 2 finalists |
| MIPROv2-style | 40 teacher rollouts, 5 demo sets (empty, labeled, bootstrapped), 6 instructions, 12 TPE trials on 20-example validation minibatches, full evaluation every 6 trials |
| TextGrad-style | up to 8 iterations, 10-example minibatch, 4 failures per gradient, momentum 2, validation gate |
| GEPA-style | up to 30 iterations, 3-example reflection minibatch, strict minibatch gate, Pareto parent sampling |
| PromptBreeder-style | population 6, 6 generations of 3 tournaments, 12-example fitness minibatch, crossover rate 0.1, 2 finalists |
| DSPy BootstrapFewShotWithRandomSearch | 4 bootstrapped and 4 labeled demos, 3 candidate programs |
| DSPy MIPROv2 | `auto=None`, 6 candidates, 12 trials, minibatch 20, full evaluation every 6, 3 bootstrapped and 3 labeled demos |

## 4. Results

On GSM8K with Qwen3.5-2B, TextGrad-style search raises test accuracy from 64.7% (seed
prompt) to 75.0%, +10.3 points [4.7, 16.0], p < 0.001, using 399 task rollouts and 0.13M
tokens; it is the only method whose gain survives a correction for the seven comparisons.
OPRO (+6.0), GEPA-style (+5.3) and PromptBreeder-style (+5.0) gain about half as much, and
MIPROv2-style, despite tying for the best validation score, tests below the seed prompt. On
BIG-Bench Hard, where one prompt serves 27 tasks, no method beats the seed prompt.

### 4.1 GSM8K, Qwen3.5-2B (300 test problems)

| Method | Val acc | Test acc [95% CI] | Diff vs seed [95% CI] | p | Truncated | Task rollouts | Proposer calls | Opt. tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Seed prompt (zero-shot) | 76.7 | 64.7 [59.3, 70.0] |  |  | 1.3% | 60 | 0 | 0.01M |
| Bootstrapped few-shot (random search) | 81.7 | 68.0 [62.7, 73.3] | 3.3 [-2.3, 9.0] | 0.274 | 1.7% | 400 | 0 | 0.27M |
| APE | 81.7 | 65.0 [59.7, 70.3] | 0.3 [-5.3, 6.0] | 0.945 | 1.3% | 388 | 12 | 0.13M |
| OPRO | 83.3 | 70.7 [65.3, 75.7] | 6.0 [1.0, 11.0] | 0.019 | 1.0% | 390 | 18 | 0.16M |
| MIPROv2-style | 86.7 | 64.0 [58.7, 69.3] | -0.7 [-6.0, 4.7] | 0.839 | 0.0% | 320 | 6 | 0.15M |
| TextGrad-style | 86.7 | 75.0 [69.7, 79.7] | 10.3 [4.7, 16.0] | 0.000 | 3.0% | 399 | 10 | 0.13M |
| GEPA-style | 85.0 | 70.0 [64.7, 75.0] | 5.3 [-0.3, 11.0] | 0.072 | 1.3% | 356 | 20 | 0.18M |
| PromptBreeder-style | 86.7 | 69.7 [64.3, 74.7] | 5.0 [-0.3, 10.3] | 0.088 | 4.7% | 348 | 19 | 0.14M |

Test examples: 300 shared by every row (of 1319 available). Accuracy in percent; intervals are 95% percentile bootstrap over examples (10,000 resamples); differences and p-values are paired bootstrap against the seed prompt on the same examples. Truncated is the share of test answers cut off by the task's max_tokens (scored wrong when no final answer was reached).

TextGrad-style search is the only method whose gain survives a correction for the seven
comparisons with the seed prompt (Holm, family-wise 5%): 75.0% against 64.7%, a paired
difference of +10.3 points [4.7, 16.0], p < 0.001. OPRO (+6.0, p = 0.019) is significant on
its own but not after the correction; GEPA-style (+5.3, p = 0.07) and PromptBreeder-style
(+5.0, p = 0.09) are borderline; bootstrapped few-shot (+3.3) and APE (+0.3) are within
noise. MIPROv2-style tied for the best validation score and tested below the seed prompt.

Every winning program is an instruction without demonstrations. TextGrad-style rewrote the
instruction from critiques of failed examples into an ordering rule ("perform all arithmetic
operations first ... state the scenario logic only after all calculations are complete");
the rewrite step's `<solution>` wrapper was kept verbatim in the instruction, which the
model ignores. OPRO converged on restating the output contract step by step, and GEPA-style
and PromptBreeder-style on longer role and rule lists. Demonstrations did not help
here: the two programs that use them (bootstrapped few-shot with two, MIPROv2-style with
one) are not significantly better than the seed prompt.

### 4.2 BIG-Bench Hard, Qwen3.5-2B (27 tasks, 270 test examples)

| Method | Val acc | Test acc [95% CI] | Diff vs seed [95% CI] | p | Truncated | Task rollouts | Proposer calls | Opt. tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Seed prompt (zero-shot) | 61.1 | 49.6 [43.7, 55.6] |  |  | 6.7% | 54 | 0 | 0.02M |
| Bootstrapped few-shot (random search) | 61.1 | 49.6 [43.7, 55.6] | 0.0 [0.0, 0.0] | 1.000 | 6.7% | 378 | 0 | 0.18M |
| APE | 61.1 | 49.6 [43.7, 55.6] | 0.0 [0.0, 0.0] | 1.000 | 6.7% | 316 | 12 | 0.14M |
| OPRO | 61.1 | 49.6 [43.7, 55.6] | 0.0 [0.0, 0.0] | 1.000 | 6.7% | 333 | 21 | 0.19M |
| MIPROv2-style | 61.1 | 49.6 [43.7, 55.6] | 0.0 [0.0, 0.0] | 1.000 | 6.7% | 274 | 6 | 0.14M |
| TextGrad-style | 64.8 | 51.1 [45.2, 57.0] | 1.5 [-4.1, 7.0] | 0.653 | 7.0% | 392 | 14 | 0.18M |
| GEPA-style | 61.1 | 49.6 [43.7, 55.6] | 0.0 [0.0, 0.0] | 1.000 | 6.7% | 398 | 22 | 0.24M |
| PromptBreeder-style | 61.1 | 49.6 [43.7, 55.6] | 0.0 [0.0, 0.0] | 1.000 | 6.7% | 258 | 14 | 0.16M |

Test examples: 270 shared by every row (of 405 available). Accuracy in percent; intervals are 95% percentile bootstrap over examples (10,000 resamples); differences and p-values are paired bootstrap against the seed prompt on the same examples. Truncated is the share of test answers cut off by the task's max_tokens (scored wrong when no final answer was reached).

One instruction is shared by all 27 tasks, which mix logical deduction, date arithmetic,
word sorting and boolean evaluation, and the seed instruction ("Answer the question.")
relies on each example's own task framing. Against that seed, six of seven methods found no
candidate that beat it on the 54-example validation set and returned it unchanged, so their
test rows equal the seed's. Their searches did run: GEPA-style accepted four candidates on
three-example minibatches (scores up to 1.0) that then scored 38.9 to 57.4 on full
validation, and MIPROv2-style's best minibatch trial (75.0 on 20 examples) scored 51.9 on
full validation. TextGrad-style's selected program is the seed instruction wrapped in
`<system_prompt>` tags; it gained 3.7 points on validation and 1.5 on test (p = 0.65), which
is noise. Per-task prompts, the usual setup for BIG-Bench Hard, are on the roadmap.

### 4.3 Against DSPy 3.4.1 (GSM8K)

| Method | Implementation | Test acc [95% CI] | Diff vs own zero-shot [95% CI] | In-repo minus DSPy [95% CI] | p | Task rollouts | Proposer calls | Opt. tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot | Seed prompt (zero-shot) (this repository) | 64.7 [59.3, 70.0] |  |  |  | 60 | 0 | 0.01M |
| Zero-shot | DSPy ChainOfThought (zero-shot) (dspy 3.4.1) | 67.0 [61.7, 72.3] |  | -2.3 [-7.3, 2.7] | 0.406 | 60 | 0 | 0.02M |
| Bootstrapped few-shot | Bootstrapped few-shot (random search) (this repository) | 68.0 [62.7, 73.3] | 3.3 [-2.3, 9.0] |  |  | 400 | 0 | 0.27M |
| Bootstrapped few-shot | DSPy BootstrapFewShotWithRandomSearch (dspy 3.4.1) | 69.7 [64.3, 74.7] | 2.7 [-2.7, 8.0] | -1.7 [-7.0, 3.7] | 0.559 | 440 | 0 | 0.41M |
| MIPROv2 | MIPROv2-style (this repository) | 64.0 [58.7, 69.3] | -0.7 [-6.0, 4.7] |  |  | 320 | 6 | 0.15M |
| MIPROv2 | DSPy MIPROv2 (dspy 3.4.1) | 69.7 [64.3, 74.7] | 2.7 [-2.3, 7.7] | -5.7 [-11.0, -0.3] | 0.046 | 499 | 29 | 0.45M |

Test examples: 300 shared by every row. Accuracy in percent with 95% bootstrap intervals; the gain column is paired against the same implementation's zero-shot program, the head-to-head column is paired in-repo minus DSPy.

DSPy 3.4.1 ran on the same model, server, splits and test prefix, through this
repository's cached and metered client, so calls and tokens are counted the same way. DSPy
formats prompts differently (field markers and a separate reasoning field), so each
implementation is compared with its own zero-shot program as well as head to head. DSPy's
zero-shot ChainOfThought scores 67.0 against the in-repo seed prompt's 64.7, and its
BootstrapFewShotWithRandomSearch 69.7 against the in-repo bootstrapped few-shot's 68.0;
neither difference is significant (p = 0.41 and 0.56). DSPy's random search has no rollout
cap: with three candidate programs it used 440 task rollouts, 10% more than the in-repo
methods' budget. DSPy's MIPROv2 also has no hard cap (its trial count and minibatch size set the cost): it
used 499 task rollouts, 29 proposer calls and 0.45M tokens, against 320 rollouts, 6 proposer
calls and 0.15M tokens for the in-repo MIPROv2-style run. It scores 69.7, which is 5.7
points [0.3, 11.0] above the in-repo MIPROv2-style program (p = 0.046), the one significant
head-to-head difference, and 2.7 points [-2.3, 7.7] above DSPy's own zero-shot program.
Both MIPROv2 runs scored well on validation (83.3 and 86.7); the in-repo run lost more of it
on test (22.7 points against 13.6), so with these budgets DSPy's longer search selected a
program that generalized better.

### 4.4 Validation versus test

GSM8K:

| Method | Candidates scored on val | Val acc | Test acc | Val minus test | Minus the seed's gap |
|---|---:|---:|---:|---:|---:|
| Seed prompt (zero-shot) | 1 | 76.7 | 64.7 | +12.0 | +0.0 |
| Bootstrapped few-shot (random search) | 6 | 81.7 | 68.0 | +13.7 | +1.7 |
| APE | 3 | 81.7 | 65.0 | +16.7 | +4.7 |
| OPRO | 3 | 83.3 | 70.7 | +12.7 | +0.7 |
| MIPROv2-style | 3 | 86.7 | 64.0 | +22.7 | +10.7 |
| TextGrad-style | 5 | 86.7 | 75.0 | +11.7 | -0.3 |
| GEPA-style | 4 | 85.0 | 70.0 | +15.0 | +3.0 |
| PromptBreeder-style | 3 | 86.7 | 69.7 | +17.0 | +5.0 |

BIG-Bench Hard:

| Method | Candidates scored on val | Val acc | Test acc | Val minus test | Minus the seed's gap |
|---|---:|---:|---:|---:|---:|
| Seed prompt (zero-shot) | 1 | 61.1 | 49.6 | +11.5 | +0.0 |
| Bootstrapped few-shot (random search) | 6 | 61.1 | 49.6 | +11.5 | +0.0 |
| APE | 2 | 61.1 | 49.6 | +11.5 | +0.0 |
| OPRO | 2 | 61.1 | 49.6 | +11.5 | +0.0 |
| MIPROv2-style | 2 | 61.1 | 49.6 | +11.5 | +0.0 |
| TextGrad-style | 5 | 64.8 | 51.1 | +13.7 | +2.2 |
| GEPA-style | 5 | 61.1 | 49.6 | +11.5 | +0.0 |
| PromptBreeder-style | 3 | 61.1 | 49.6 | +11.5 | +0.0 |

The seed prompt itself scores 12.0 points lower on test than on validation, so the
validation problems are easier than the test prefix for this model; every method inherits
that gap, and the last column removes it. MIPROv2-style (+10.7) and PromptBreeder-style (+5.0) lost the most:
three methods reached the same 86.7% on 60 validation problems and then tested at 75.0,
69.7 and 64.0, so a validation set this small cannot separate candidates within a few
points. TextGrad-style kept its validation gain on test (-0.3).

### 4.5 Anytime performance

Best full-validation accuracy each method had reached within a given number of task
rollouts (read from each run's trial history; no extra model calls).

GSM8K:

| Method | <= 100 rollouts | <= 200 rollouts | <= 300 rollouts | <= 400 rollouts |
|---|---:|---:|---:|---:|
| Bootstrapped few-shot (random search) | 76.7 | 80.0 | 81.7 | 81.7 |
| APE | 76.7 | 76.7 | 76.7 | 81.7 |
| OPRO | 76.7 | 76.7 | 76.7 | 83.3 |
| MIPROv2-style | 76.7 | 76.7 | 78.3 | 86.7 |
| TextGrad-style | 76.7 | 76.7 | 76.7 | 86.7 |
| GEPA-style | 76.7 | 85.0 | 85.0 | 85.0 |
| PromptBreeder-style | 76.7 | 76.7 | 76.7 | 86.7 |

BIG-Bench Hard:

| Method | <= 100 rollouts | <= 200 rollouts | <= 300 rollouts | <= 400 rollouts |
|---|---:|---:|---:|---:|
| Bootstrapped few-shot (random search) | 61.1 | 61.1 | 61.1 | 61.1 |
| APE | 61.1 | 61.1 | 61.1 | 61.1 |
| OPRO | 61.1 | 61.1 | 61.1 | 61.1 |
| MIPROv2-style | 61.1 | 61.1 | 61.1 | 61.1 |
| TextGrad-style | 61.1 | 61.1 | 61.1 | 64.8 |
| GEPA-style | 61.1 | 61.1 | 61.1 | 61.1 |
| PromptBreeder-style | 61.1 | 61.1 | 61.1 | 61.1 |

### 4.6 Transfer across model sizes

The programs optimized on Qwen3.5-2B, scored unchanged with Qwen3.5-0.8B on the same test
examples (paired differences against the seed prompt on 0.8B):

GSM8K (300 problems):

| Method | qwen3.5-0.8b | qwen3.5-2b |
|---|---:|---:|
| Seed prompt (zero-shot) | 40.3 [34.7, 46.0] | 64.7 [59.3, 70.0] |
| MIPROv2-style | 33.3 [28.0, 38.7] | 64.0 [58.7, 69.3] |
| TextGrad-style | 45.0 [39.3, 50.7] | 75.0 [69.7, 79.7] |
| PromptBreeder-style | 44.7 [39.0, 50.0] | 69.7 [64.3, 74.7] |

Programs optimized on qwen3.5-2b. Shared test examples per model: qwen3.5-0.8b: 300, qwen3.5-2b: 300.

BIG-Bench Hard (270 examples):

| Method | qwen3.5-0.8b | qwen3.5-2b |
|---|---:|---:|
| Seed prompt (zero-shot) | 36.3 [30.7, 42.2] | 49.6 [43.7, 55.6] |
| TextGrad-style | 34.4 [28.9, 40.0] | 51.1 [45.2, 57.0] |

Programs optimized on qwen3.5-2b. Shared test examples per model: qwen3.5-0.8b: 270, qwen3.5-2b: 270.

The two instruction rewrites that helped on 2B also help on 0.8B, by about half as much and
no longer significantly: TextGrad-style +4.7 points [-1.0, 10.3] (p = 0.12) and
PromptBreeder-style +4.3 [-1.3, 10.0] (p = 0.16). MIPROv2-style's program, level with the
seed prompt on 2B, hurts on 0.8B: -7.0 points [-13.0, -0.7] (p = 0.03). On BIG-Bench Hard,
TextGrad-style's 2B gain does not carry over (-1.9 points, p = 0.57).

### 4.7 Cost

Task rollouts are capped at 400 per run and counted before every batch; proposer calls
(instruction proposals, critiques, reflections, mutations) are counted but not capped and
are cheap next to the rollouts: 0 to 22 calls per run. Optimization tokens range from
0.13M to 0.27M per run. On GSM8K bootstrapped few-shot is the most expensive (0.27M),
because its candidate programs carry demonstrations into every rollout, and TextGrad-style,
the best method there, is among the cheapest (0.13M); on BIG-Bench Hard GEPA-style is the
most expensive (0.24M). Each run took 10 to 67 minutes of wall time on the shared M1 at about
45 completion tokens per second across 8 parallel requests; wall time also depends on how
many calls were already in the response cache, so the token counts are the comparable
measure.

### 4.8 Benchmarks without real-model results yet

MATH-500, HotpotQA (distractor) and IFEval are implemented end to end (download with
checksums, seeded splits, answer extraction, scoring, the 25 IFEval instruction checkers,
proposer feedback) and covered by offline tests, and every optimizer runs on them with
`--task math500`, `--task hotpotqa` or `--task ifeval`. They have no real-model results in
this release: on the shared M1 the GSM8K and BIG-Bench Hard comparisons used the compute
budget, and the README reports only what was run.

## 5. Threats to validity

- **Budget and scale.** 400 rollouts per run is small next to the papers, which often use
  thousands; methods that need many iterations (OPRO, PromptBreeder) are the most limited.
  The ranking at larger budgets may differ.
- **Single optimization seed.** Bootstrap intervals cover test-sample variance, not search
  variance; another seed can reorder methods whose intervals overlap.
- **Validation size.** 60 (GSM8K) and 54 (BIG-Bench Hard) validation examples make
  selection noisy; the validation versus test table shows the resulting optimism.
- **Test subsets.** GSM8K uses a seeded 300-problem prefix of the 1,319-problem test split
  and BIG-Bench Hard 10 examples per task (270 of the 405 held-out examples); the sizes are
  in every table. A 300-example test resolves differences of about 5 points at 95%
  confidence for paired comparisons, so smaller true effects can be missed.
- **Multiple comparisons.** Seven methods are each compared with the seed prompt; the
  p-values are per comparison and not corrected, so one of seven comparisons can fall
  below 0.05 by chance more often than 5% of the time.
- **Model and quantization.** 4-bit quantized small models without thinking; results may not
  carry over to larger models or to thinking mode.
- **DSPy formatting.** DSPy's chat adapter formats prompts with field markers and parses a
  separate answer field, so its zero-shot baseline differs from the in-repo seed prompt; the
  DSPy rows are compared with DSPy's own zero-shot program as well. DSPy's demonstrations are
  shared across BIG-Bench Hard subtasks, while the in-repo optimizers can keep them per task.
- **Contamination.** All benchmarks are public and older than the models.
- **Shared machine.** Throughput varied with other jobs; it changes wall-clock time, not
  results, because decoding is greedy and every response is cached.

## 6. Reproduce

```bash
uv run --with pyarrow python scripts/download_data.py
python benchmarks/run_experiments.py --task gsm8k --budget 400 --test-limit 300 \
    --optimizers seed bootstrap_fewshot gepa miprov2 opro textgrad ape promptbreeder
python benchmarks/run_experiments.py --task bbh --budget 400 --test-limit 270 \
    --optimizers seed bootstrap_fewshot gepa miprov2 opro textgrad ape promptbreeder
uv run --group compare python benchmarks/dspy_compare.py --task gsm8k --budget 400 \
    --test-limit 300 --optimizers seed bootstrap_rs miprov2
python benchmarks/run_experiments.py --task gsm8k --optimizers seed textgrad miprov2 promptbreeder \
    --eval-model qwen3.5-0.8b --test-limit 300 --transfer-only
python benchmarks/run_experiments.py --task bbh --optimizers seed textgrad \
    --eval-model qwen3.5-0.8b --test-limit 270 --transfer-only
python -m prompt_optimizer report
```

Replaying from the response cache reproduces every number exactly; a fresh run on other
hardware reproduces them up to llama.cpp's batch-composition nondeterminism.

## 7. Reference pipeline (simulated model)

The original optimizers (greedy bootstrap few-shot, instruction search and their compound)
run on a deterministic simulated model and a three-class intent task with 9 training, 4
validation and 6 test messages. Reproduce with `python -m prompt_optimizer evaluate`.

| Method | Demonstrations | Test accuracy |
|---|---:|---:|
| Baseline (naive, zero-shot) | 0 | 0.333 |
| Instruction search | 0 | 0.833 |
| Few-shot bootstrap | 3 | 0.833 |
| Compound (instruction plus few-shot) | 1 | 1.000 |

The naive baseline has no label vocabulary and no demonstrations and must guess from
keywords. Declaring the vocabulary or adding demonstrations each lifts accuracy to 0.833,
and the compound optimizer reaches 1.000 with one demonstration. The task is tiny and the
model simulated, so this shows the mechanics, not a benchmark result.
