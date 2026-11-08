"""Tests for statistics, the experiment runner, the report and the new CLI commands."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from prompt_optimizer import cli
from prompt_optimizer.backends import CachedLM, CostMeter, FakeLM
from prompt_optimizer.experiments import RunSpec, evaluate_program, optimize, ordered_examples
from prompt_optimizer.program import PromptProgram
from prompt_optimizer.report import (
    common_ids,
    label_for,
    load_rows,
    main_table,
    overfit_table,
    render,
    transfer_table,
)
from prompt_optimizer.stats import bootstrap_ci, mcnemar, paired_bootstrap
from prompt_optimizer.tasks.base import TaskExample
from prompt_optimizer.tasks.registry import Splits
from prompt_optimizer.tasks.synthetic import ArithmeticTask

TASK = ArithmeticTask()


class TestStats:
    def test_ci_contains_mean_and_narrows_with_n(self) -> None:
        small = bootstrap_ci([1.0, 0.0] * 10)
        large = bootstrap_ci([1.0, 0.0] * 500)
        assert small.low <= small.mean <= small.high
        assert (large.high - large.low) < (small.high - small.low)
        assert small.n == 20 and "50.0" in small.fmt()

    def test_ci_matches_normal_approximation(self) -> None:
        ci = bootstrap_ci([1.0] * 700 + [0.0] * 300, n_boot=20_000)
        half = 1.96 * math.sqrt(0.7 * 0.3 / 1000)
        assert abs((ci.high - ci.low) / 2 - half) < 0.004

    def test_ci_is_seeded(self) -> None:
        scores = list(np.random.default_rng(0).integers(0, 2, 100).astype(float))
        assert bootstrap_ci(scores, seed=4) == bootstrap_ci(scores, seed=4)
        with pytest.raises(ValueError):
            bootstrap_ci([])

    def test_paired_bootstrap(self) -> None:
        a = [1.0] * 60 + [0.0] * 40
        b = [1.0] * 40 + [0.0] * 60
        result = paired_bootstrap(a, b)
        assert result.diff.mean == pytest.approx(0.2)
        assert result.diff.low > 0 and result.p_value < 0.01
        assert result.wins == 20 and result.losses == 0
        reverse = paired_bootstrap(b, a)
        assert reverse.diff.high < 0 and reverse.p_value < 0.01
        assert paired_bootstrap(a, a).p_value == 1.0
        with pytest.raises(ValueError):
            paired_bootstrap([1.0], [1.0, 0.0])

    def test_mcnemar(self) -> None:
        assert mcnemar([1, 1], [1, 1]) == 1.0
        # 10 discordant pairs all one way: p = 2 * 0.5**10
        assert mcnemar([1.0] * 10, [0.0] * 10) == pytest.approx(2 * 0.5**10)
        with pytest.raises(ValueError):
            mcnemar([1.0], [])

    @settings(max_examples=40, deadline=None)
    @given(st.lists(st.tuples(st.booleans(), st.booleans()), min_size=1, max_size=60))
    def test_mcnemar_is_symmetric_and_bounded(self, pairs: list[tuple[bool, bool]]) -> None:
        a = [float(x) for x, _ in pairs]
        b = [float(y) for _, y in pairs]
        p = mcnemar(a, b)
        assert 0.0 <= p <= 1.0 and p == pytest.approx(mcnemar(b, a))


def _splits() -> Splits:
    return Splits(
        tuple(ArithmeticTask.make(40, seed=1, prefix="tr")),
        tuple(ArithmeticTask.make(20, seed=2, prefix="va")),
        tuple(ArithmeticTask.make(50, seed=3, prefix="te")),
    )


class TestExperiments:
    def test_test_order_prefixes_nest_and_cover(self) -> None:
        examples = ArithmeticTask.make(30, seed=5, prefix="x")
        order = ordered_examples(examples)
        assert sorted(e.id for e in order) == sorted(e.id for e in examples)
        assert ordered_examples(list(reversed(examples))) == order

    @given(st.integers(2, 6), st.integers(1, 8))
    def test_test_order_interleaves_groups(self, groups: int, per: int) -> None:
        examples = [
            TaskExample(f"g{g}-{i}", "q", "a", group=f"g{g}")
            for g in range(groups)
            for i in range(per)
        ]
        prefix = ordered_examples(examples)[:groups]
        assert {e.group for e in prefix} == {f"g{g}" for g in range(groups)}

    def test_optimize_evaluate_and_resume(self, tmp_path: Path) -> None:
        splits = _splits()
        spec = RunSpec("arith", "gepa", "fake", seed=0, budget=200)
        meter = CostMeter()
        lm = CachedLM(FakeLM(), meter=meter)
        run = optimize(spec, TASK, splits, lm, meter, root=tmp_path)
        assert spec.run_path(tmp_path).exists() and run["rollouts_used"] <= 200
        assert run["cost"]["task"]["calls"] > 0 and run["splits"]["val"] == 20
        again = optimize(spec, TASK, splits, CachedLM(FakeLM()), CostMeter(), root=tmp_path)
        assert again == run

        program = PromptProgram.from_dict(run["program"])
        lines: list[str] = []
        test = evaluate_program(
            spec,
            TASK,
            program,
            splits.test,
            lm,
            CostMeter(),
            "fake",
            limit=20,
            chunk=8,
            root=tmp_path,
            log=lines.append,
        )
        assert test["n"] == 20 and test["n_available"] == 50 and len(lines) == 3
        assert test["records"][0]["id"] == ordered_examples(splits.test)[0].id
        bigger = evaluate_program(
            spec, TASK, program, splits.test, lm, CostMeter(), "fake", root=tmp_path
        )
        assert bigger["n"] == 50
        assert [r["id"] for r in bigger["records"][:20]] == [r["id"] for r in test["records"]]
        same = evaluate_program(
            spec, TASK, program, splits.test, lm, CostMeter(), "fake", limit=10, root=tmp_path
        )
        assert same["n"] == 50

    def test_seed_run(self, tmp_path: Path) -> None:
        spec = RunSpec("arith", "seed", "fake")
        assert spec.run_id == "seed"
        run = optimize(spec, TASK, _splits(), CachedLM(FakeLM()), CostMeter(), root=tmp_path)
        assert run["trials"] == [] and run["rollouts_used"] == 20
        assert run["program"]["instruction"] == TASK.seed_instruction


def _fake_results(root: Path) -> None:
    splits = _splits()
    for optimizer in ("seed", "gepa", "opro"):
        spec = RunSpec("arith", optimizer, "fake", budget=150)
        meter = CostMeter()
        lm = CachedLM(FakeLM(), meter=meter)
        run = optimize(spec, TASK, splits, lm, meter, root=root)
        program = PromptProgram.from_dict(run["program"])
        for eval_model in ("fake", "fake-b"):
            evaluate_program(
                spec, TASK, program, splits.test, lm, CostMeter(), eval_model, root=root
            )


class TestReport:
    def test_tables(self, tmp_path: Path) -> None:
        _fake_results(tmp_path)
        rows = load_rows("arith", "fake", "fake", tmp_path)
        assert [r.optimizer for r in rows] == ["seed", "opro", "gepa"]
        table = main_table("arith", "fake", "fake", tmp_path)
        assert "| Seed prompt (zero-shot) |" in table and "GEPA-style" in table
        assert "Test examples: 50 shared" in table
        transfer = transfer_table("arith", ["fake", "fake-b"], "fake", tmp_path)
        assert "fake-b" in transfer and "OPRO" in transfer
        overfit = overfit_table("arith", "fake", "fake", tmp_path)
        assert "Val minus test" in overfit and "| Seed prompt (zero-shot) | 1 |" in overfit
        assert "No overfitting" in overfit_table("x", "m", "m", tmp_path)
        text = render(tmp_path, opt_model="fake")
        assert "## arith" in text and "### Transfer" in text

    def test_empty(self, tmp_path: Path) -> None:
        assert render(tmp_path) == "No results yet."
        assert "No results" in main_table("x", "m", "m", tmp_path)
        assert "No transfer" in transfer_table("x", ["m"], "m", tmp_path)
        assert common_ids([]) == []
        assert load_rows("x", "m", "m", tmp_path) == []

    def test_labels(self) -> None:
        assert label_for("gepa-s0-b400", "gepa") == "GEPA-style"
        assert label_for("gepa-s2-b400", "gepa") == "GEPA-style (seed 2)"
        assert label_for("custom", "custom") == "custom"


class TestCli:
    def test_optimizers_lists_all(self, capsys: pytest.CaptureFixture[str]) -> None:
        cli.main(["optimizers"])
        out = capsys.readouterr().out
        for name in ("gepa", "miprov2", "opro", "ape", "textgrad", "promptbreeder"):
            assert name in out

    def test_optimize_fake(self, capsys: pytest.CaptureFixture[str]) -> None:
        cli.main(
            ["optimize", "--fake", "--optimizer", "opro", "--budget", "200", "--set", "steps=2"]
        )
        out = capsys.readouterr().out
        assert "opro on arithmetic" in out and "selected: val" in out and "cost:" in out

    def test_report(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        cli.main(["report", "--root", str(tmp_path)])
        assert "No results yet." in capsys.readouterr().out

    def test_optimize_benchmark_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from prompt_optimizer.tasks import registry

        def fake_load(name: str) -> tuple[ArithmeticTask, Splits]:
            assert name == "gsm8k"
            return TASK, _splits()

        monkeypatch.setattr(registry, "load_task", fake_load)
        cache = tmp_path / "c.sqlite"
        cli.main(
            [
                "optimize",
                "--task",
                "gsm8k",
                "--fake",
                "--optimizer",
                "bootstrap_fewshot",
                "--budget",
                "100",
                "--cache",
                str(cache),
            ]
        )
        assert "bootstrap_fewshot on arithmetic: 40 train, 20 validation" in capsys.readouterr().out
        assert cache.exists()
        json.dumps({})
