"""Measure the effect of the length clause in the GSM8K and BIG-Bench Hard contracts.

Scores the seed prompt on the same validation examples under the original contract
(answer line only, 512 tokens) and the current one (length clause, 384 tokens) and prints
the table in ADR-0005. Every call is cached, so a rerun replays the recorded answers.

    python benchmarks/contract_probe.py
"""

from __future__ import annotations

from prompt_optimizer.backends import CachedLM, LlamaCppLM, ResponseCache
from prompt_optimizer.optim.base import Evaluator
from prompt_optimizer.program import PromptProgram
from prompt_optimizer.tasks.registry import DATA_ROOT, load_task

ORIGINAL = {
    "gsm8k": "End your response with a final line of the form 'Answer: <number>'.",
    "bbh": (
        "End your response with a final line of the form 'Answer: <answer>', where <answer> is "
        "only the final answer in the form the question asks for: an option letter such as (A), "
        "True or False, Yes or No, valid or invalid, a number, or a space-separated sequence."
    ),
}
EXAMPLES = {"gsm8k": 36, "bbh": 54}


def main() -> None:
    cache = ResponseCache(DATA_ROOT.parent / "cache" / "lm.sqlite")
    lm = CachedLM(
        LlamaCppLM("http://127.0.0.1:8091", "qwen3.5-2b"), cache, model_id="qwen3.5-2b-q4km"
    )
    print("| Benchmark | Contract | Accuracy | Completion tokens per answer | Truncated |")
    print("|---|---|---:|---:|---:|")
    for name, n in EXAMPLES.items():
        task, splits = load_task(name)
        examples = list(splits.val[:n])
        current = (task.contract, task.max_tokens)
        for label, (contract, max_tokens) in (
            ("answer line only, 512 tokens", (ORIGINAL[name], 512)),
            ("with length clause, 384 tokens", current),
        ):
            task.contract, task.max_tokens = contract, max_tokens
            result = Evaluator(task, lm).evaluate(PromptProgram(task.seed_instruction), examples)
            requests = [
                PromptProgram(task.seed_instruction).request(
                    contract, ex.question, group=ex.group, max_tokens=max_tokens
                )
                for ex in examples
            ]
            answers = lm.generate(requests)
            tokens = sum(a.completion_tokens for a in answers) / n
            truncated = sum(1 for r in result.rollouts if r.truncated)
            print(
                f"| {name} ({n} validation) | {label} | {100 * result.mean:.1f}% | "
                f"{tokens:.0f} | {truncated} |"
            )
        task.contract, task.max_tokens = current


if __name__ == "__main__":
    main()
