"""Tests for the core types, text utilities, language model, and metrics."""

from __future__ import annotations

import pytest

from prompt_optimizer import (
    Example,
    Predictor,
    Prompt,
    SimulatedLanguageModel,
    evaluate,
    exact_match,
    token_f1,
)
from prompt_optimizer.lm import parse_label_vocabulary
from prompt_optimizer.text import jaccard, tokenize


def test_example_rejects_empty() -> None:
    with pytest.raises(ValueError):
        Example(text="  ", label="x")
    with pytest.raises(ValueError):
        Example(text="x", label="")


def test_prompt_render_contains_demonstrations() -> None:
    prompt = Prompt("Classify.", (Example("I want a refund", "refund"),))
    rendered = prompt.render()
    assert "I want a refund" in rendered
    assert "Label: refund" in rendered
    assert "{query}" in rendered


def test_prompt_immutable_updates() -> None:
    base = Prompt("a")
    updated = base.with_instruction("b").with_demonstrations((Example("x", "y"),))
    assert base.instruction == "a"
    assert updated.instruction == "b"
    assert updated.demonstrations[0].label == "y"


def test_tokenize_drops_stopwords() -> None:
    assert tokenize("I want a Refund please") == ["want", "refund"]


def test_jaccard_bounds() -> None:
    assert jaccard("refund money", "refund money") == 1.0
    assert jaccard("refund", "shipping") == 0.0
    assert jaccard("", "x") == 0.0


def test_exact_match_is_case_insensitive() -> None:
    assert exact_match("Refund", "refund") == 1.0
    assert exact_match("refund", "shipping") == 0.0


def test_token_f1_partial_overlap() -> None:
    assert token_f1("fast refund", "refund") == pytest.approx(2 / 3)
    assert token_f1("", "") == 1.0
    assert token_f1("refund", "") == 0.0


def test_parse_label_vocabulary() -> None:
    vocab = parse_label_vocabulary("Classify. Valid labels: refund, shipping, cancel")
    assert vocab == ("refund", "shipping", "cancel")
    assert parse_label_vocabulary("No vocabulary here") == ()


def test_model_copies_nearest_demonstration() -> None:
    prompt = Prompt(
        "Classify. Valid labels: refund, shipping",
        (
            Example("where is my package shipping", "shipping"),
            Example("I want my money back refund", "refund"),
        ),
    )
    model = SimulatedLanguageModel()
    assert model.generate(prompt, "my money back") == "refund"
    assert model.generate(prompt, "where is my package") == "shipping"


def test_model_keyword_guess_without_demonstrations() -> None:
    prompt = Prompt("Classify. Valid labels: refund, shipping, cancel")
    model = SimulatedLanguageModel()
    assert model.generate(prompt, "please cancel this") == "cancel"


def test_model_snaps_to_vocabulary() -> None:
    prompt = Prompt("Classify. Valid labels: refund, shipping")
    model = SimulatedLanguageModel()
    # An unrelated query with a declared vocabulary falls back to unknown.
    assert model.generate(prompt, "zzz") == "unknown"


def test_predictor_and_evaluate() -> None:
    prompt = Prompt(
        "Classify. Valid labels: refund, shipping",
        (
            Example("where is my package shipping", "shipping"),
            Example("I want my money back refund", "refund"),
        ),
    )
    predictor = Predictor(prompt)
    dataset = [Example("my money back", "refund"), Example("where is my package", "shipping")]
    assert evaluate(predictor.predict, dataset, exact_match) == 1.0
    assert evaluate(predictor.predict, []) == 0.0
