"""Tests for the ablation and the command line interface."""

from __future__ import annotations

from prompt_optimizer.cli import main
from prompt_optimizer.evaluation import format_table, run_ablation


def test_ablation_compound_is_best() -> None:
    rows = run_ablation()
    by_method = {row.method: row for row in rows}
    baseline = by_method["baseline (naive, zero-shot)"].score
    compound = by_method["compound (instruction + few-shot)"].score
    assert compound >= baseline
    assert compound == 1.0
    assert by_method["baseline (naive, zero-shot)"].demonstrations == 0
    assert by_method["compound (instruction + few-shot)"].demonstrations >= 1


def test_each_optimizer_at_least_matches_baseline() -> None:
    rows = run_ablation()
    by_method = {row.method: row for row in rows}
    baseline = by_method["baseline (naive, zero-shot)"].score
    for name in ("instruction search", "few-shot bootstrap"):
        assert by_method[name].score >= baseline


def test_format_table_contains_methods() -> None:
    table = format_table(run_ablation())
    assert "compound" in table
    assert "test accuracy" in table


def test_cli_evaluate(capsys) -> None:
    main(["evaluate"])
    out = capsys.readouterr().out
    assert "compound" in out


def test_cli_compile(capsys) -> None:
    main(["compile"])
    out = capsys.readouterr().out
    assert "Optimized instruction" in out
    assert "Selected demonstrations" in out


def test_cli_predict(capsys) -> None:
    main(["predict", "I need a refund for my money"])
    out = capsys.readouterr().out.strip()
    assert out == "refund"
