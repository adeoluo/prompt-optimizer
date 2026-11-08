# Data Card: Prompt Optimizer

The benchmarks are public datasets downloaded from their authors' official locations by
`scripts/download_data.py` into `.bench-data/prompt-optimizer/data/` (outside the
repository; nothing is committed). The script writes a manifest with the SHA-256 digest and
size of every file. The reference pipeline also bundles one tiny synthetic task.

## Benchmarks

| Benchmark | Source | Size downloaded | Splits used (seed 20260211) | Score |
|---|---|---:|---|---|
| GSM8K (Cobbe et al., 2021) | openai/grade-school-math, `train.jsonl`, `test.jsonl` | 7,473 train, 1,319 test | train 200 and validation 60 drawn from the official train split; test is the official 1,319 (scored in a fixed seeded order, prefix size stated per table) | exact numeric match |
| BIG-Bench Hard (Suzgun et al., 2022) | suzgunmirac/BIG-Bench-Hard, 27 task files and CoT prompt headers | 6,511 examples, 27 tasks | per task: 8 train, 2 validation, 15 test (216, 54, 405) | exact match after answer-shape normalization |
| MATH-500 (Lightman et al., 2023) | openai/prm800k `math_splits` | 12,000 train, 500 test | train 200 and validation 100 from the train split; test is the 500 | normalized string match of the boxed answer (Hendrycks et al. rules) with a numeric fallback |
| HotpotQA distractor dev (Yang et al., 2018) | hotpotqa/hotpot_qa on Hugging Face (the original host no longer resolves) | 7,405 questions | seeded train 150, validation 100, test 500 | token F1 (exact match also reported) |
| IFEval (Zhou et al., 2023) | google-research `instruction_following_eval/data/input_data.jsonl` | 541 prompts | seeded train 100, validation 100, test 341 | prompt-level strict accuracy |

- **Licensing:** GSM8K MIT; BIG-Bench Hard MIT; PRM800K MATH splits MIT (MATH problems from
  Hendrycks et al., MIT); HotpotQA CC BY-SA 4.0; IFEval Apache 2.0. Only scores, ids and
  short extracted answers are committed under `benchmarks/results/`; full model outputs stay
  in the local response cache.
- **Contamination:** all five benchmarks are public and predate the Qwen3.5 models used
  here, so absolute accuracies may be inflated by training-set exposure. Comparisons between
  prompts on the same model are much less affected, which is what this project reports.
- **Known gaps:** English only; BIG-Bench Hard and HotpotQA use seeded subsets of their
  evaluation data (sizes above) because of compute on the shared Apple M1.

## Bundled synthetic tasks

- `src/prompt_optimizer/data.py`: the original nine train, four validation and six test
  customer messages labeled refund, shipping or cancel, used by the reference optimizers.
- `tasks/synthetic.py`: integer arithmetic questions generated from a seed, used with the
  deterministic fake model in tests and offline demos.

## Ethical considerations

The benchmarks contain no personal data beyond public Wikipedia text in HotpotQA. The
synthetic tasks are generated or hand-authored and contain no personal or sensitive content.
