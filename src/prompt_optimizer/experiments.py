"""Experiment runner: optimize on train/validation, score on test, write results.

One *run* is (task, optimizer, optimization model, seed, budget). It writes

* ``runs/<task>/<model>/<run_id>.json``: the selected program, its validation
  score, every trial, the stop reason and the cost of the optimization phase
  (calls and tokens by purpose: task rollouts versus proposer calls);
* ``test/<task>/<eval model>/<run_id>.json``: per-example test scores of that
  program on a model (the optimization model or a transfer model), with the
  cost of the evaluation phase.

Test examples are taken in a fixed seeded order and ``limit`` keeps a prefix,
so a 400-example result is the first 400 of the full-split result and a later
extension to the full split only adds examples. All model calls go through the
content-hash cache: rerunning a finished or interrupted run replays every
answered call from disk and continues where it stopped, which makes the result
files plus the cache a resumable optimization store.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prompt_optimizer.backends.accounting import CostMeter
from prompt_optimizer.backends.base import LanguageModel
from prompt_optimizer.optim import Budget, Evaluator, build
from prompt_optimizer.program import PromptProgram
from prompt_optimizer.tasks.base import Task, TaskExample
from prompt_optimizer.tasks.registry import SPLIT_SEED, Splits

RESULTS_ROOT = Path(__file__).resolve().parents[3] / "benchmarks" / "results"
SEED_RUN = "seed"


@dataclass(frozen=True)
class RunSpec:
    """Identity and settings of one optimization run."""

    task: str
    optimizer: str
    model: str = "qwen3.5-2b"
    seed: int = 0
    budget: int = 500
    overrides: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    @property
    def run_id(self) -> str:
        if self.optimizer == SEED_RUN:
            return SEED_RUN
        return f"{self.optimizer}-s{self.seed}-b{self.budget}"

    def run_path(self, root: Path = RESULTS_ROOT) -> Path:
        return root / "runs" / self.task / self.model / f"{self.run_id}.json"

    def test_path(self, eval_model: str, root: Path = RESULTS_ROOT) -> Path:
        return root / "test" / self.task / eval_model / f"{self.run_id}.json"


def test_order(examples: Sequence[TaskExample], seed: int = SPLIT_SEED) -> list[TaskExample]:
    """The fixed seeded order in which test examples are scored (prefixes nest)."""

    order = sorted(examples, key=lambda ex: ex.id)
    random.Random(f"test-order:{seed}").shuffle(order)
    grouped = {ex.group for ex in order}
    if len(grouped) <= 1:
        return order
    # Interleave groups so every prefix covers every group about equally.
    by_group: dict[str, list[TaskExample]] = {}
    for ex in order:
        by_group.setdefault(ex.group, []).append(ex)
    out: list[TaskExample] = []
    queues = [by_group[g] for g in sorted(by_group)]
    depth = max(len(q) for q in queues)
    for i in range(depth):
        out.extend(q[i] for q in queues if i < len(q))
    return out


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def optimize(
    spec: RunSpec,
    task: Task,
    splits: Splits,
    lm: LanguageModel,
    meter: CostMeter,
    *,
    root: Path = RESULTS_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    """Run (or load) one optimization and write its run file."""

    path = spec.run_path(root)
    if path.exists() and not force:
        return load_json(path)
    seed_program = PromptProgram(task.seed_instruction)
    start = time.perf_counter()
    with meter.phase("optimize"):
        if spec.optimizer == SEED_RUN:
            evaluator = Evaluator(task, lm)
            val = evaluator.evaluate(seed_program, splits.val).mean
            program, val_score, trials, used, info = seed_program, val, [], len(splits.val), {}
        else:
            optimizer = build(spec.optimizer, **dict(spec.overrides))
            result = optimizer.compile(
                task,
                lm,
                splits.train,
                splits.val,
                seed_program=seed_program,
                budget=Budget(spec.budget),
                seed=spec.seed,
            )
            program, val_score = result.program, result.val_score
            trials = [t.__dict__ for t in result.trials]
            used, info = result.rollouts_used, result.info
    payload: dict[str, Any] = {
        "task": spec.task,
        "optimizer": spec.optimizer,
        "model": spec.model,
        "seed": spec.seed,
        "budget": spec.budget,
        "overrides": dict(spec.overrides),
        "splits": splits.sizes(),
        "val_score": val_score,
        "rollouts_used": used,
        "program": program.to_dict(),
        "fingerprint": program.fingerprint(),
        "trials": trials,
        "info": info,
        "cost": meter.breakdown().get("optimize", {}),
        "wall_seconds": round(time.perf_counter() - start, 1),
    }
    _write(path, payload)
    return payload


def evaluate_program(
    spec: RunSpec,
    task: Task,
    program: PromptProgram,
    examples: Sequence[TaskExample],
    lm: LanguageModel,
    meter: CostMeter,
    eval_model: str,
    *,
    limit: int | None = None,
    chunk: int = 128,
    root: Path = RESULTS_ROOT,
    log: Any = None,
) -> dict[str, Any]:
    """Score ``program`` on a prefix of the ordered test set and write the test file."""

    path = spec.test_path(eval_model, root)
    ordered = test_order(examples)
    wanted = ordered if limit is None else ordered[:limit]
    if path.exists():
        existing = load_json(path)
        if existing.get("fingerprint") == program.fingerprint() and existing["n"] >= len(wanted):
            return existing
    evaluator = Evaluator(task, lm)
    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    with meter.phase(f"test:{eval_model}"):
        for begin in range(0, len(wanted), chunk):
            part = wanted[begin : begin + chunk]
            result = evaluator.evaluate(program, part)
            for rollout in result.rollouts:
                record: dict[str, Any] = {
                    "id": rollout.example.id,
                    "score": rollout.score,
                    "pred": task.extract(rollout.output)[:200],
                    "truncated": rollout.truncated,
                }
                extra = task.metrics(rollout.example, rollout.output)
                if extra:
                    record["metrics"] = extra
                if rollout.example.group:
                    record["group"] = rollout.example.group
                records.append(record)
            if log is not None:
                done = len(records)
                acc = sum(r["score"] for r in records) / done
                log(
                    f"{spec.task} {spec.run_id} on {eval_model}: {done}/{len(wanted)} acc={acc:.3f}"
                )
    scores = [r["score"] for r in records]
    payload: dict[str, Any] = {
        "task": spec.task,
        "run_id": spec.run_id,
        "optimizer": spec.optimizer,
        "optimized_on": spec.model,
        "eval_model": eval_model,
        "fingerprint": program.fingerprint(),
        "n": len(records),
        "n_available": len(ordered),
        "accuracy": sum(scores) / len(scores) if scores else 0.0,
        "truncated": sum(1 for r in records if r["truncated"]),
        "records": records,
        "cost": meter.breakdown().get(f"test:{eval_model}", {}),
        "wall_seconds": round(time.perf_counter() - start, 1),
    }
    _write(path, payload)
    return payload
