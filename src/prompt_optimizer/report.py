"""Aggregate result files into the tables of the README and evaluation report.

Reads ``runs/`` (optimization) and ``test/`` (evaluation) files written by
``prompt_optimizer.experiments`` and by ``benchmarks/dspy_compare.py`` and
renders Markdown tables:

* **Main table** per task: test accuracy with a 95% bootstrap interval, the
  paired difference from the seed prompt on the same examples (interval and
  p-value), the validation score (for the overfitting gap), and the
  optimization cost (task rollouts, proposer calls, tokens).
* **Transfer table**: programs optimized on one model scored on the others,
  restricted to the examples every listed model has scored.

Reproduce with ``python -m prompt_optimizer report``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt_optimizer.experiments import RESULTS_ROOT, SEED_RUN
from prompt_optimizer.stats import bootstrap_ci, paired_bootstrap

LABELS: dict[str, str] = {
    "seed": "Seed prompt (zero-shot)",
    "bootstrap_fewshot": "Bootstrapped few-shot (random search)",
    "ape": "APE",
    "opro": "OPRO",
    "miprov2": "MIPROv2-style",
    "textgrad": "TextGrad-style",
    "gepa": "GEPA-style",
    "promptbreeder": "PromptBreeder-style",
    "dspy-bootstrap_rs": "DSPy BootstrapFewShotWithRandomSearch",
    "dspy-miprov2": "DSPy MIPROv2",
    "dspy-gepa": "DSPy GEPA",
}


@dataclass(frozen=True)
class Row:
    run_id: str
    optimizer: str
    label: str
    val: float | None
    test: dict[str, Any]
    run: dict[str, Any] | None


def _load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _scores(test: dict[str, Any], ids: Sequence[str] | None = None) -> list[float]:
    by_id = {r["id"]: float(r["score"]) for r in test["records"]}
    return [by_id[i] for i in ids] if ids is not None else list(by_id.values())


def label_for(run_id: str, optimizer: str) -> str:
    base = LABELS.get(optimizer, optimizer)
    seed = run_id.split("-s")[-1].split("-")[0] if "-s" in run_id else ""
    return f"{base} (seed {seed})" if seed and seed != "0" else base


def load_rows(task: str, eval_model: str, opt_model: str, root: Path = RESULTS_ROOT) -> list[Row]:
    rows: list[Row] = []
    test_dir = root / "test" / task / eval_model
    if not test_dir.exists():
        return rows
    for path in sorted(test_dir.glob("*.json")):
        test = _load(path)
        if test.get("optimized_on", opt_model) != opt_model:
            continue
        run_path = root / "runs" / task / opt_model / path.name
        run = _load(run_path) if run_path.exists() else None
        optimizer = test.get("optimizer", path.stem)
        rows.append(
            Row(
                run_id=test["run_id"],
                optimizer=optimizer,
                label=label_for(test["run_id"], optimizer),
                val=run["val_score"] if run else None,
                test=test,
                run=run,
            )
        )
    order = list(LABELS)
    rows.sort(key=lambda r: (order.index(r.optimizer) if r.optimizer in order else 99, r.run_id))
    return rows


def _usage(run: dict[str, Any] | None) -> tuple[int, int, int]:
    """(task calls, proposer calls, total tokens) of the optimization phase."""

    if not run:
        return 0, 0, 0
    cost = run.get("cost", {})
    task_calls = int(cost.get("task", {}).get("calls", 0))
    other = sum(int(v.get("calls", 0)) for k, v in cost.items() if k != "task")
    tokens = sum(int(v.get("total_tokens", 0)) for v in cost.values())
    return task_calls, other, tokens


def common_ids(tests: Iterable[dict[str, Any]]) -> list[str]:
    """Example ids scored in every result, in the order of the first."""

    tests = list(tests)
    if not tests:
        return []
    shared = set.intersection(*({r["id"] for r in t["records"]} for t in tests))
    return [r["id"] for r in tests[0]["records"] if r["id"] in shared]


def main_table(task: str, eval_model: str, opt_model: str, root: Path = RESULTS_ROOT) -> str:
    rows = load_rows(task, eval_model, opt_model, root)
    if not rows:
        return f"No results for {task} on {eval_model}."
    seed = next((r for r in rows if r.run_id == SEED_RUN), None)
    ids = common_ids(r.test for r in rows)
    lines = [
        "| Method | Val acc | Test acc [95% CI] | Diff vs seed [95% CI] | p | "
        "Task rollouts | Proposer calls | Opt. tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        scores = _scores(row.test, ids)
        ci = bootstrap_ci(scores)
        if seed is not None and row is not seed:
            paired = paired_bootstrap(scores, _scores(seed.test, ids))
            diff = f"{paired.diff.fmt()}"
            p = f"{paired.p_value:.3f}"
        else:
            diff, p = "", ""
        task_calls, other_calls, tokens = _usage(row.run)
        val = f"{100 * row.val:.1f}" if row.val is not None else ""
        lines.append(
            f"| {row.label} | {val} | {ci.fmt()} | {diff} | {p} | {task_calls:,} | "
            f"{other_calls:,} | {tokens / 1e6:.2f}M |"
        )
    lines.append("")
    lines.append(
        f"Test examples: {len(ids)} shared by every row (of {rows[0].test['n_available']} "
        f"available). Accuracy in percent; intervals are 95% percentile bootstrap over "
        "examples (10,000 resamples); differences and p-values are paired bootstrap against "
        "the seed prompt on the same examples."
    )
    return "\n".join(lines)


def overfit_table(task: str, eval_model: str, opt_model: str, root: Path = RESULTS_ROOT) -> str:
    """Validation score of each selected program next to its test score.

    The validation score is what selected the program (the maximum over the
    candidates a method scored), so it is biased upward; the gap to the test score
    measures how much of the apparent gain was selection noise.
    """

    rows = [r for r in load_rows(task, eval_model, opt_model, root) if r.val is not None]
    if not rows:
        return f"No overfitting data for {task}."
    ids = common_ids(r.test for r in rows)
    lines = [
        "| Method | Candidates scored on val | Val acc | Test acc | Val minus test |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        test = bootstrap_ci(_scores(row.test, ids)).mean
        assert row.val is not None
        trials = (row.run or {}).get("trials", [])
        candidates = len({t["fingerprint"] for t in trials if t.get("split") == "val"}) or 1
        lines.append(
            f"| {row.label} | {candidates} | {100 * row.val:.1f} | {100 * test:.1f} | "
            f"{100 * (row.val - test):+.1f} |"
        )
    return "\n".join(lines)


def transfer_table(
    task: str, eval_models: Sequence[str], opt_model: str, root: Path = RESULTS_ROOT
) -> str:
    per_model = {m: {r.run_id: r for r in load_rows(task, m, opt_model, root)} for m in eval_models}
    run_ids = [
        rid for rid in per_model.get(opt_model, {}) if all(rid in per_model[m] for m in eval_models)
    ]
    if not run_ids:
        return f"No transfer results for {task}."
    header = "| Method | " + " | ".join(eval_models) + " |"
    lines = [header, "|---|" + "---:|" * len(eval_models)]
    shared = {m: common_ids(per_model[m][rid].test for rid in run_ids) for m in eval_models}
    for rid in run_ids:
        cells = []
        for m in eval_models:
            row = per_model[m][rid]
            cells.append(bootstrap_ci(_scores(row.test, shared[m])).fmt())
        lines.append(f"| {per_model[opt_model][rid].label} | " + " | ".join(cells) + " |")
    sizes = ", ".join(f"{m}: {len(shared[m])}" for m in eval_models)
    lines.append("")
    lines.append(f"Programs optimized on {opt_model}. Shared test examples per model: {sizes}.")
    return "\n".join(lines)


def render(root: Path = RESULTS_ROOT, opt_model: str = "qwen3.5-2b") -> str:
    parts: list[str] = []
    test_root = root / "test"
    if not test_root.exists():
        return "No results yet."
    for task_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        task = task_dir.name
        parts.append(f"## {task}\n")
        parts.append(main_table(task, opt_model, opt_model, root))
        parts.append("\n### Validation versus test\n")
        parts.append(overfit_table(task, opt_model, opt_model, root))
        models = sorted(p.name for p in task_dir.iterdir() if p.is_dir())
        if len(models) > 1:
            parts.append("\n### Transfer\n")
            parts.append(transfer_table(task, models, opt_model, root))
        parts.append("")
    return "\n".join(parts)
