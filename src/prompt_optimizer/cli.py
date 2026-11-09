"""Command line interface for the prompt optimizer."""

from __future__ import annotations

import argparse
import json

from prompt_optimizer.data import (
    CANDIDATE_INSTRUCTIONS,
    NAIVE_INSTRUCTION,
    TRAIN,
    VALIDATION,
)
from prompt_optimizer.evaluation import format_table, run_ablation
from prompt_optimizer.optimizers import CompoundOptimizer
from prompt_optimizer.predictor import Predictor
from prompt_optimizer.types import Prompt


def _build_optimized() -> Prompt:
    base = Prompt(NAIVE_INSTRUCTION)
    return CompoundOptimizer(CANDIDATE_INSTRUCTIONS).compile(base, TRAIN, VALIDATION).prompt


def _cmd_evaluate(_: argparse.Namespace) -> None:
    print(format_table(run_ablation()))


def _cmd_compile(_: argparse.Namespace) -> None:
    prompt = _build_optimized()
    print("Optimized instruction:")
    print(f"  {prompt.instruction}")
    print(f"Selected demonstrations: {len(prompt.demonstrations)}")
    for demo in prompt.demonstrations:
        print(f"  - ({demo.label}) {demo.text}")


def _cmd_predict(args: argparse.Namespace) -> None:
    predictor = Predictor(_build_optimized())
    print(predictor.predict(args.text))


def _cmd_optimizers(_: argparse.Namespace) -> None:
    from prompt_optimizer.optim import OPTIMIZERS

    for name, cls in OPTIMIZERS.items():
        doc = (cls.__module__ and __import__(cls.__module__, fromlist=["x"]).__doc__) or ""
        print(f"{name:18s} {doc.strip().splitlines()[0]}")


def _cmd_optimize(args: argparse.Namespace) -> None:
    from prompt_optimizer.backends import CachedLM, CostMeter, FakeLM, LlamaCppLM, ResponseCache
    from prompt_optimizer.backends.base import LanguageModel
    from prompt_optimizer.optim import Budget, build
    from prompt_optimizer.tasks.base import Task
    from prompt_optimizer.tasks.registry import load_task
    from prompt_optimizer.tasks.synthetic import ArithmeticTask

    meter = CostMeter()
    task: Task
    if args.task == "arithmetic":
        task = ArithmeticTask()
        train = ArithmeticTask.make(60, seed=1, prefix="tr")
        val = ArithmeticTask.make(30, seed=2, prefix="va")
    else:
        task, splits = load_task(args.task)
        train, val = list(splits.train), list(splits.val)
    inner: LanguageModel = (
        FakeLM() if args.fake else LlamaCppLM(args.url, args.model, concurrency=args.concurrency)
    )
    cache = ResponseCache(args.cache) if args.cache else ResponseCache()
    model_id = inner.model_id if args.fake else f"{args.model}-q4km"
    lm = CachedLM(inner, cache, meter, model_id=model_id)
    overrides = {k: json.loads(v) for k, v in (item.split("=", 1) for item in args.set)}
    optimizer = build(args.optimizer, **overrides)
    result = optimizer.compile(task, lm, train, val, budget=Budget(args.budget), seed=args.seed)
    print(
        f"{args.optimizer} on {task.name} with {model_id}: "
        f"{len(train)} train, {len(val)} validation, budget {args.budget}"
    )
    for trial in result.trials:
        if trial.split == "val":
            print(
                f"  step {trial.step:>2}  val {100 * trial.score:5.1f}  "
                f"rollouts {trial.rollouts_used:>4}  {trial.note:<18} "
                f"{trial.instruction[:60]!r}"
            )
    usage = meter.total()
    print(f"selected: val {100 * result.val_score:.1f}, {len(result.program.demos)} demos")
    print(f"instruction: {result.program.instruction}")
    print(
        f"cost: {result.rollouts_used} task rollouts, {usage.calls} calls, "
        f"{usage.total_tokens:,} tokens, stopped: {result.info['stopped']}"
    )


def _cmd_report(args: argparse.Namespace) -> None:
    from pathlib import Path

    from prompt_optimizer.report import render

    print(render(Path(args.root)) if args.root else render())


def _cmd_serve(args: argparse.Namespace) -> None:
    from prompt_optimizer.service import serve

    serve(host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="prompt-optimizer", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("evaluate", help="run the four-way ablation on held-out data")
    sub.add_parser("compile", help="print the compiled prompt")

    predict = sub.add_parser("predict", help="label a single message")
    predict.add_argument("text", help="the customer message to classify")

    sub.add_parser("optimizers", help="list the prompt optimizers")

    opt = sub.add_parser("optimize", help="run one optimizer on a task")
    opt.add_argument("--task", default="arithmetic", help="arithmetic, gsm8k, bbh, ...")
    opt.add_argument("--optimizer", default="gepa")
    opt.add_argument("--budget", type=int, default=400, help="task rollout budget")
    opt.add_argument("--seed", type=int, default=0)
    opt.add_argument("--fake", action="store_true", help="use the deterministic fake model")
    opt.add_argument("--url", default="http://127.0.0.1:8091", help="llama-server URL")
    opt.add_argument("--model", default="qwen3.5-2b", help="model id for the cache key")
    opt.add_argument("--concurrency", type=int, default=32)
    opt.add_argument("--cache", default="", help="SQLite response cache path")
    opt.add_argument("--set", nargs="*", default=[], help="optimizer overrides key=json")

    rep = sub.add_parser("report", help="render the results tables")
    rep.add_argument("--root", default="", help="results directory")

    serve = sub.add_parser("serve", help="start the HTTP prediction service")
    serve.add_argument("--host", default="127.0.0.1", help="bind host")
    serve.add_argument("--port", type=int, default=8000, help="bind port")

    handlers = {
        "evaluate": _cmd_evaluate,
        "compile": _cmd_compile,
        "predict": _cmd_predict,
        "serve": _cmd_serve,
        "optimizers": _cmd_optimizers,
        "optimize": _cmd_optimize,
        "report": _cmd_report,
    }
    args = parser.parse_args(argv)
    handlers[args.command](args)


if __name__ == "__main__":
    main()
