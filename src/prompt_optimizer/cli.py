"""Command line interface for the prompt optimizer."""

from __future__ import annotations

import argparse

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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="prompt-optimizer", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("evaluate", help="run the four-way ablation on held-out data")
    sub.add_parser("compile", help="print the compiled prompt")

    predict = sub.add_parser("predict", help="label a single message")
    predict.add_argument("text", help="the customer message to classify")

    handlers = {
        "evaluate": _cmd_evaluate,
        "compile": _cmd_compile,
        "predict": _cmd_predict,
    }
    args = parser.parse_args(argv)
    handlers[args.command](args)


if __name__ == "__main__":
    main()
