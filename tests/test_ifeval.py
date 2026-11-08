"""Tests for the 25 IFEval instruction checkers and the IFEval task metrics."""

from __future__ import annotations

from typing import Any

import pytest

from prompt_optimizer.tasks import ifeval
from prompt_optimizer.tasks.base import TaskExample

CASES: list[tuple[str, dict[str, Any], str, bool]] = [
    ("keywords:existence", {"keywords": ["apple", "pie"]}, "Apple and PIE.", True),
    ("keywords:existence", {"keywords": ["apple", "pie"]}, "Apple only.", False),
    (
        "keywords:frequency",
        {"keyword": "cat", "frequency": 2, "relation": "at least"},
        "cat Cat",
        True,
    ),
    (
        "keywords:frequency",
        {"keyword": "cat", "frequency": 2, "relation": "less than"},
        "cat Cat",
        False,
    ),
    ("keywords:forbidden_words", {"forbidden_words": ["bad"]}, "badge is fine", True),
    ("keywords:forbidden_words", {"forbidden_words": ["bad"]}, "this is Bad.", False),
    (
        "keywords:letter_frequency",
        {"letter": "z", "let_frequency": 2, "let_relation": "at least"},
        "Zz",
        True,
    ),
    (
        "keywords:letter_frequency",
        {"letter": "z", "let_frequency": 2, "let_relation": "less than"},
        "zzz",
        False,
    ),
    (
        "language:response_language",
        {"language": "en"},
        "This is clearly an English sentence about the weather today.",
        True,
    ),
    (
        "language:response_language",
        {"language": "de"},
        "This is clearly an English sentence about the weather today.",
        False,
    ),
    (
        "length_constraints:number_sentences",
        {"num_sentences": 3, "relation": "less than"},
        "One. Two.",
        True,
    ),
    (
        "length_constraints:number_sentences",
        {"num_sentences": 3, "relation": "at least"},
        "Dr. Smith came. He left.",
        False,
    ),
    ("length_constraints:number_paragraphs", {"num_paragraphs": 2}, "First\n***\nSecond", True),
    (
        "length_constraints:number_paragraphs",
        {"num_paragraphs": 2},
        "***\nFirst\n***\nSecond\n***",
        True,
    ),
    (
        "length_constraints:number_paragraphs",
        {"num_paragraphs": 2},
        "First\n***\n***\nSecond",
        False,
    ),
    (
        "length_constraints:number_words",
        {"num_words": 3, "relation": "at least"},
        "one two three",
        True,
    ),
    (
        "length_constraints:number_words",
        {"num_words": 3, "relation": "less than"},
        "one two three",
        False,
    ),
    (
        "length_constraints:nth_paragraph_first_word",
        {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "then"},
        'Intro.\n\n"Then, more.',
        True,
    ),
    (
        "length_constraints:nth_paragraph_first_word",
        {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "then"},
        "Intro.\n\nNow more.",
        False,
    ),
    (
        "length_constraints:nth_paragraph_first_word",
        {"num_paragraphs": 3, "nth_paragraph": 3, "first_word": "x"},
        "Only one.",
        False,
    ),
    (
        "length_constraints:nth_paragraph_first_word",
        {"num_paragraphs": 1, "nth_paragraph": 1, "first_word": "x"},
        "\n\nX marks",
        False,
    ),
    (
        "detectable_content:number_placeholders",
        {"num_placeholders": 2},
        "Dear [name] at [address]",
        True,
    ),
    ("detectable_content:number_placeholders", {"num_placeholders": 2}, "Dear [name]", False),
    ("detectable_content:postscript", {"postscript_marker": "P.S."}, "Body\nP.S. hi", True),
    ("detectable_content:postscript", {"postscript_marker": "P.P.S"}, "Body\nP.P.S hi", True),
    ("detectable_content:postscript", {"postscript_marker": "Note:"}, "Body\nnote: hi", True),
    ("detectable_content:postscript", {"postscript_marker": "P.S."}, "Body only", False),
    ("detectable_format:number_bullet_lists", {"num_bullets": 2}, "* a\n- b", True),
    ("detectable_format:number_bullet_lists", {"num_bullets": 2}, "* a\n** b", False),
    ("detectable_format:constrained_response", {}, "Well. My answer is maybe.", True),
    ("detectable_format:constrained_response", {}, "Probably yes.", False),
    (
        "detectable_format:number_highlighted_sections",
        {"num_highlights": 2},
        "*one* and **two**",
        True,
    ),
    (
        "detectable_format:number_highlighted_sections",
        {"num_highlights": 2},
        "*one* and ** **",
        False,
    ),
    (
        "detectable_format:multiple_sections",
        {"section_spliter": "Section", "num_sections": 2},
        "Section 1\na\nSection 2\nb",
        True,
    ),
    (
        "detectable_format:multiple_sections",
        {"section_spliter": "Section", "num_sections": 2},
        "Section 1\na",
        False,
    ),
    ("detectable_format:json_format", {}, '```json\n{"a": 1}\n```', True),
    ("detectable_format:json_format", {}, "{a: 1}", False),
    ("detectable_format:title", {}, "<<My Title>>\ntext", True),
    ("detectable_format:title", {}, "<< >>\ntext", False),
    ("combination:two_responses", {}, "first\n******\nsecond", True),
    ("combination:two_responses", {}, "same\n******\nsame", False),
    ("combination:two_responses", {}, "a\n******\n\n******\nb", False),
    ("combination:repeat_prompt", {"prompt_to_repeat": "Say hi"}, "say hi. Hi!", True),
    ("combination:repeat_prompt", {"prompt_to_repeat": "Say hi"}, "Hi!", False),
    ("startend:end_checker", {"end_phrase": "Any questions?"}, 'Thanks. any questions?"', True),
    ("startend:end_checker", {"end_phrase": "Any questions?"}, "Thanks.", False),
    (
        "change_case:capital_word_frequency",
        {"capital_frequency": 2, "capital_relation": "at least"},
        "NASA and ESA fly",
        True,
    ),
    (
        "change_case:capital_word_frequency",
        {"capital_frequency": 2, "capital_relation": "less than"},
        "NASA and ESA fly",
        False,
    ),
    (
        "change_case:english_capital",
        {},
        "THIS IS AN ENGLISH SENTENCE ABOUT THE WEATHER TODAY.",
        True,
    ),
    ("change_case:english_capital", {}, "This Is Not All Caps.", False),
    (
        "change_case:english_lowercase",
        {},
        "this is an english sentence about the weather today.",
        True,
    ),
    ("change_case:english_lowercase", {}, "This is not lowercase.", False),
    ("punctuation:no_comma", {}, "no commas here", True),
    ("punctuation:no_comma", {}, "a, b", False),
    ("startend:quotation", {}, '"quoted"', True),
    ("startend:quotation", {}, "not quoted", False),
]


@pytest.mark.parametrize(("instruction", "kwargs", "response", "expected"), CASES)
def test_checker(instruction: str, kwargs: dict[str, Any], response: str, expected: bool) -> None:
    assert ifeval.follows(instruction, kwargs, response) is expected


def test_every_instruction_type_is_covered() -> None:
    assert {case[0] for case in CASES} == set(ifeval.INSTRUCTION_IDS)
    assert len(ifeval.INSTRUCTION_IDS) == 25


def test_empty_response_and_unknowns() -> None:
    assert not ifeval.follows("punctuation:no_comma", {}, "   ")
    with pytest.raises(KeyError):
        ifeval.follows("made:up", {}, "x")
    with pytest.raises(ValueError):
        ifeval.follows(
            "length_constraints:number_words", {"num_words": 1, "relation": "about"}, "x"
        )


def test_none_kwargs_are_dropped_and_language_fallback() -> None:
    assert ifeval.follows("punctuation:no_comma", {"num_words": None}, "fine")
    assert ifeval.follows("language:response_language", {"language": "en"}, "12345 !!!")


def test_sentence_splitter_protects_abbreviations() -> None:
    sentences = ifeval.split_into_sentences(
        "Dr. Smith paid 3.5 dollars at example.com. Then he left!"
    )
    assert len(sentences) == 2
    assert ifeval.split_into_sentences('He said "hi." Then left... ok') == [
        'He said "hi".',
        "Then left...",
        "ok",
    ]


def test_loose_accepts_wrapped_answers() -> None:
    response = "Sure, here it is:\nhello world"
    assert not ifeval.follows("punctuation:no_comma", {}, response)
    assert ifeval.follows_loose("punctuation:no_comma", {}, response)
    assert len(ifeval.loose_variants("a\nb\nc")) == 8


def test_task_metrics_and_feedback() -> None:
    task = ifeval.IFEval()
    ex = TaskExample(
        "i",
        "q",
        "",
        meta={
            "instruction_id_list": ["punctuation:no_comma", "startend:quotation"],
            "kwargs": [{}, {}],
        },
    )
    assert task.score(ex, '"no commas"') == 1.0
    assert task.score(ex, "no commas") == 0.5
    metrics = task.metrics(ex, "no commas")
    assert metrics["prompt_strict"] == 0.0 and metrics["inst_strict"] == 0.5
    assert metrics["n_instructions"] == 2.0
    assert "startend:quotation" in task.feedback(ex, "no commas")
    assert task.feedback(ex, '"ok"').startswith("Every")
    assert task.extract("x") == "x"
    with pytest.raises(ValueError):
        task.gold_output(ex)
