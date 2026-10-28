"""Tests for the bootstrap, instruction-search, and compound optimizers."""

from __future__ import annotations

import pytest

from prompt_optimizer import (
    BootstrapFewShot,
    CompoundOptimizer,
    Example,
    InstructionSearch,
    Predictor,
    Prompt,
    evaluate,
)

_VALID = "Classify the message. Valid labels: refund, shipping, cancel"
_NAIVE = "Answer the message."

_TRAIN = [
    Example("I want my money back give me a refund", "refund"),
    Example("where is my package tracking shipping", "shipping"),
    Example("please cancel my order subscription", "cancel"),
    Example("refund my payment immediately", "refund"),
    Example("my parcel has not arrived shipping delay", "shipping"),
    Example("stop my plan cancel it", "cancel"),
]

_VALIDATION = [
    Example("give me a refund for my money", "refund"),
    Example("where is my parcel package", "shipping"),
    Example("cancel my subscription please", "cancel"),
]


def test_bootstrap_improves_over_zero_shot() -> None:
    base = Prompt(_VALID)
    before = evaluate(Predictor(base).predict, _VALIDATION)
    result = BootstrapFewShot(max_demonstrations=4).compile(base, _TRAIN, _VALIDATION)
    assert result.score >= before
    assert result.score == 1.0
    assert len(result.prompt.demonstrations) >= 1
    assert len(result.prompt.demonstrations) <= 4


def test_bootstrap_respects_budget() -> None:
    base = Prompt(_VALID)
    result = BootstrapFewShot(max_demonstrations=2).compile(base, _TRAIN, _VALIDATION)
    assert len(result.prompt.demonstrations) <= 2


def test_bootstrap_rejects_bad_budget() -> None:
    with pytest.raises(ValueError):
        BootstrapFewShot(max_demonstrations=0)


def test_instruction_search_prefers_vocabulary_instruction() -> None:
    base = Prompt(_NAIVE)
    search = InstructionSearch([_NAIVE, _VALID])
    result = search.compile(base, _VALIDATION)
    assert "Valid labels" in result.prompt.instruction


def test_instruction_search_requires_candidates() -> None:
    with pytest.raises(ValueError):
        InstructionSearch([])


def test_compound_beats_each_component() -> None:
    base = Prompt(_NAIVE)
    instruction_only = InstructionSearch([_NAIVE, _VALID]).compile(base, _VALIDATION)
    few_shot_only = BootstrapFewShot().compile(base, _TRAIN, _VALIDATION)
    compound = CompoundOptimizer([_NAIVE, _VALID]).compile(base, _TRAIN, _VALIDATION)
    assert compound.score >= instruction_only.score
    assert compound.score >= few_shot_only.score
    assert compound.score == 1.0
    assert len(compound.trace) >= 2


def test_compound_rejects_bad_rounds() -> None:
    with pytest.raises(ValueError):
        CompoundOptimizer([_VALID], rounds=0)
