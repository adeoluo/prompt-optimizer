"""Head-to-head against DSPy optimizers on the same model, splits and budget.

DSPy (``dspy==3.4.1``, the latest release at the time of the runs) compiles a
``ChainOfThought`` program whose signature carries the same seed instruction as
the in-repo optimizers. Every DSPy model call is routed through this
repository's ``CachedLM`` (``RoutedLM`` below), so DSPy and the in-repo
optimizers share one content-hash cache, one llama.cpp server and one cost
meter: calls and tokens are counted identically.

Optimizers and budgets (task rollouts are metric calls on train or validation):

* ``dspy-bootstrap_rs``: ``BootstrapFewShotWithRandomSearch`` (4 bootstrapped
  demos, 3 candidate programs), validation set as given;
* ``dspy-miprov2``: ``MIPROv2`` with ``auto=None``, 6 instruction candidates,
  12 trials on minibatches of 20, full evaluation every 6 trials;
* ``dspy-gepa``: ``GEPA`` with ``max_metric_calls`` equal to the in-repo budget
  and reflection minibatches of 3; the same model reflects.

The compiled program is then scored on the same ordered test prefix as every
other method, and result files use the same format, so the report tables put
DSPy rows next to the in-repo rows. ``dspy-seed`` is DSPy's own zero-shot
``ChainOfThought`` with the seed instruction (DSPy formats prompts differently,
so its starting point is reported separately).

Run (needs the ``compare`` dependency group and a running llama-server):

    uv run --group compare python benchmarks/dspy_compare.py --task gsm8k \\
        --optimizers seed bootstrap_rs miprov2 gepa --budget 400 --test-limit 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import dspy

from prompt_optimizer.backends import CachedLM, CostMeter, LlamaCppLM, ResponseCache
from prompt_optimizer.backends.base import LMRequest, Message, Purpose
from prompt_optimizer.experiments import RESULTS_ROOT, RunSpec, ordered_examples
from prompt_optimizer.tasks.base import Task, TaskExample
from prompt_optimizer.tasks.registry import DATA_ROOT, load_task

DSPY_VERSION = dspy.__version__
CACHE = DATA_ROOT.parent / "cache" / "lm.sqlite"

ANSWER_FIELD = {
    "gsm8k": "the final answer as a single number",
    "bbh": (
        "only the final answer in the form the question asks for: an option letter such as "
        "(A), True or False, Yes or No, valid or invalid, a number, or a space-separated sequence"
    ),
    "math500": "the final answer in simplest LaTeX form, without \\boxed",
    "hotpotqa": "a short answer span (a few words) or yes/no",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


class RoutedLM(dspy.BaseLM):  # type: ignore[misc]
    """A DSPy LM that answers through this repository's cached, metered client."""

    def __init__(self, lm: CachedLM, purpose: Purpose, max_tokens: int, temperature: float = 0.0):
        super().__init__(
            model=f"routed/{lm.model_id}",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=False,
        )
        self.lm = lm
        self.purpose = purpose

    def forward(
        self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> SimpleNamespace:
        merged = {**self.kwargs, **kwargs}
        raw = messages or [{"role": "user", "content": prompt or ""}]
        chat = tuple(Message(m["role"], _text(m["content"])) for m in raw)
        n = int(merged.get("n", 1) or 1)
        rollout = merged.get("rollout_id")
        base_seed = int(rollout) if isinstance(rollout, int) else 0
        temperature = float(merged.get("temperature", 0.0) or 0.0)
        requests = [
            LMRequest(
                messages=chat,
                max_tokens=int(merged.get("max_tokens") or 512),
                temperature=temperature,
                seed=base_seed + i,
                purpose=self.purpose,
            )
            for i in range(n)
        ]
        responses = self.lm.generate(requests)
        choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=r.text, tool_calls=None),
                finish_reason=r.finish_reason,
                index=i,
            )
            for i, r in enumerate(responses)
        ]
        usage = {
            "prompt_tokens": sum(r.prompt_tokens for r in responses),
            "completion_tokens": sum(r.completion_tokens for r in responses),
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return SimpleNamespace(choices=choices, usage=usage, model=self.model)


#: The brevity clause of the task contracts (ADR-0005), carried by DSPy's reasoning field,
#: which DSPy optimizers do not rewrite (they rewrite the signature instructions).
REASONING_FIELD = {
    "gsm8k": "your reasoning in a few short steps (no more than about 120 words)",
    "bbh": "your reasoning in a few short sentences (no more than about 120 words)",
}


def make_program(task: Task) -> dspy.Module:
    signature = dspy.Signature("question -> answer", task.seed_instruction)
    signature = signature.with_updated_fields("answer", desc=ANSWER_FIELD.get(task.name, "answer"))
    rationale = dspy.OutputField(desc=REASONING_FIELD.get(task.name, "${reasoning}"))
    return dspy.ChainOfThought(signature, rationale_field=rationale)


def as_output(prediction: Any) -> str:
    reasoning = getattr(prediction, "reasoning", "") or ""
    answer = getattr(prediction, "answer", "") or ""
    return f"{reasoning}\nAnswer: {answer}"


def to_examples(examples: list[TaskExample]) -> list[dspy.Example]:
    return [
        dspy.Example(question=ex.question, answer=ex.answer, ex_id=ex.id).with_inputs("question")
        for ex in examples
    ]


def program_dict(program: dspy.Module) -> dict[str, Any]:
    predictor = program.predictors()[0]
    demos = [
        {
            k: str(v)[:4000]
            for k, v in dict(demo).items()
            if k in {"question", "reasoning", "answer"}
        }
        for demo in predictor.demos
    ]
    return {"instruction": predictor.signature.instructions, "demos": demos}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--optimizers", nargs="+", default=["seed", "bootstrap_rs", "miprov2", "gepa"]
    )
    parser.add_argument("--model", default="qwen3.5-2b", help="optimization model")
    parser.add_argument("--eval-model", default=None, help="test model (default: --model)")
    parser.add_argument("--transfer-only", action="store_true", help="score saved programs only")
    parser.add_argument("--budget", type=int, default=400)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--root", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args(argv)

    eval_model = args.eval_model or args.model
    task, splits = load_task(args.task)
    index = {ex.id: ex for ex in (*splits.train, *splits.val, *splits.test)}
    trainset, valset = to_examples(list(splits.train)), to_examples(list(splits.val))
    cache = ResponseCache(CACHE)

    def score(gold: dspy.Example, prediction: Any) -> float:
        return task.score(index[gold.ex_id], as_output(prediction))

    def metric(gold: dspy.Example, prediction: Any, trace: Any = None) -> float:
        return score(gold, prediction)

    def feedback_metric(
        gold: dspy.Example,
        prediction: Any,
        trace: Any = None,
        pred_name: str | None = None,
        pred_trace: Any = None,
    ) -> dspy.Prediction:
        output = as_output(prediction)
        example = index[gold.ex_id]
        return dspy.Prediction(
            score=task.score(example, output), feedback=task.feedback(example, output)
        )

    for name in args.optimizers:
        meter = CostMeter()
        inner = LlamaCppLM(args.url, args.model, concurrency=args.threads)
        client = CachedLM(inner, cache, meter, model_id=f"{args.model}-q4km")
        task_lm = RoutedLM(client, "task", task.max_tokens)
        prompt_lm = RoutedLM(client, "propose", 1024, temperature=1.0)
        reflect_lm = RoutedLM(client, "reflect", 2048, temperature=1.0)
        dspy.configure(lm=task_lm)
        spec = RunSpec(args.task, f"dspy-{name}", args.model, args.seed, args.budget)
        run_path = spec.run_path(args.root)
        start = time.perf_counter()
        log(f"dspy {DSPY_VERSION} {name} on {args.task}")
        student = make_program(task)
        if args.transfer_only and not run_path.exists():
            log(f"skip {spec.run_id}: no run file")
            continue
        if run_path.exists():
            run = json.loads(run_path.read_text())
            compiled = make_program(task)
            compiled.load(str(run_path.with_suffix(".dspy.json")))
        else:
            with meter.phase("optimize"):
                if name == "seed":
                    compiled = student
                elif name == "bootstrap_rs":
                    optimizer = dspy.BootstrapFewShotWithRandomSearch(
                        metric=metric,
                        max_bootstrapped_demos=4,
                        max_labeled_demos=4,
                        num_candidate_programs=3,
                        num_threads=args.threads,
                    )
                    compiled = optimizer.compile(student, trainset=trainset, valset=valset)
                elif name == "miprov2":
                    optimizer = dspy.MIPROv2(
                        metric=metric,
                        prompt_model=prompt_lm,
                        task_model=task_lm,
                        max_bootstrapped_demos=3,
                        max_labeled_demos=3,
                        auto=None,
                        num_candidates=6,
                        num_threads=args.threads,
                        seed=args.seed,
                        verbose=False,
                    )
                    compiled = optimizer.compile(
                        student,
                        trainset=trainset,
                        valset=valset,
                        num_trials=12,
                        minibatch_size=20,
                        minibatch_full_eval_steps=6,
                        requires_permission_to_run=False,
                    )
                elif name == "gepa":
                    optimizer = dspy.GEPA(
                        metric=feedback_metric,
                        max_metric_calls=args.budget,
                        reflection_minibatch_size=3,
                        reflection_lm=reflect_lm,
                        num_threads=args.threads,
                        seed=args.seed,
                        track_stats=False,
                    )
                    compiled = optimizer.compile(student, trainset=trainset, valset=valset)
                else:
                    raise SystemExit(f"unknown DSPy optimizer {name}")
                val_result = dspy.Evaluate(
                    devset=valset,
                    metric=metric,
                    num_threads=args.threads,
                    display_progress=False,
                    max_errors=len(valset),
                )(compiled)
                val_scores = [float(value) for _, _, value in val_result.results]
            compiled.save(str(run_path.with_suffix(".dspy.json")))
            run = {
                "task": args.task,
                "optimizer": spec.optimizer,
                "library": f"dspy {DSPY_VERSION}",
                "model": args.model,
                "seed": args.seed,
                "budget": args.budget,
                "splits": splits.sizes(),
                "val_score": sum(val_scores) / len(val_scores),
                "rollouts_used": meter.total("optimize", "task").calls,
                "program": program_dict(compiled),
                "cost": meter.breakdown().get("optimize", {}),
                "wall_seconds": round(time.perf_counter() - start, 1),
            }
            run_path.parent.mkdir(parents=True, exist_ok=True)
            run_path.write_text(json.dumps(run, indent=1, ensure_ascii=False) + "\n")
        log(f"{name}: val={run['val_score']:.3f} task calls={run['rollouts_used']}")

        if eval_model != args.model:
            from run_experiments import ensure_server

            ensure_server(eval_model, int(args.url.rsplit(":", 1)[1]))
            eval_client = CachedLM(
                LlamaCppLM(args.url, eval_model, concurrency=args.threads),
                cache,
                meter,
                model_id=f"{eval_model}-q4km",
            )
            dspy.configure(lm=RoutedLM(eval_client, "task", task.max_tokens))
        test_path = spec.test_path(eval_model, args.root)
        ordered = ordered_examples(splits.test)
        wanted = ordered if args.test_limit is None else ordered[: args.test_limit]
        if test_path.exists() and json.loads(test_path.read_text())["n"] >= len(wanted):
            continue
        records: list[dict[str, Any]] = []
        test_start = time.perf_counter()
        with meter.phase(f"test:{eval_model}"):
            evaluator = dspy.Evaluate(
                devset=to_examples(wanted),
                metric=metric,
                num_threads=args.threads,
                display_progress=False,
                provide_traceback=False,
                max_errors=len(wanted),
            )
            result = evaluator(compiled)
            for example, prediction, value in result.results:
                output = as_output(prediction)
                records.append(
                    {
                        "id": example.ex_id,
                        "score": float(value),
                        "pred": task.extract(output)[:200],
                        "truncated": False,
                    }
                )
        accuracy = sum(r["score"] for r in records) / len(records)
        payload = {
            "task": args.task,
            "run_id": spec.run_id,
            "optimizer": spec.optimizer,
            "library": f"dspy {DSPY_VERSION}",
            "optimized_on": args.model,
            "eval_model": eval_model,
            "fingerprint": "",
            "n": len(records),
            "n_available": len(ordered),
            "accuracy": accuracy,
            "truncated": 0,
            "records": records,
            "cost": meter.breakdown().get(f"test:{eval_model}", {}),
            "wall_seconds": round(time.perf_counter() - test_start, 1),
        }
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
        log(f"test {spec.run_id} on {eval_model}: acc={accuracy:.3f} n={len(records)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
