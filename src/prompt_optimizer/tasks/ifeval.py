"""IFEval: verifiable instruction following (Zhou et al., 2023).

541 prompts, each carrying one to three of 25 verifiable instruction types
(word counts, forbidden words, JSON output, bullet counts, casing, and so on).
The checkers below reimplement the semantics of the official
``instruction_following_eval`` package: the same relations, regular
expressions and edge cases, with two documented differences. Sentence splitting
uses a regex splitter modelled on the official one, and capital-word counting
tokenizes with a regex instead of NLTK's ``word_tokenize``; both affect only
``length_constraints:number_sentences`` and ``change_case:capital_word_frequency``.

Metrics, as in the paper: prompt-level strict accuracy (every instruction in a
prompt followed), instruction-level strict accuracy, and the loose variants
that also accept the response with its first line, last line, or asterisks
removed. The optimization score is the instruction-level strict fraction for
one prompt, a smoother signal than the all-or-nothing prompt-level score.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from prompt_optimizer.tasks.base import Task, TaskExample

Checker = Callable[[str, Mapping[str, Any]], bool]


def _relation(count: int, target: int, relation: str) -> bool:
    if relation == "less than":
        return count < target
    if relation == "at least":
        return count >= target
    raise ValueError(f"unknown relation {relation!r}")


def _detect_language(text: str) -> str | None:
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0
    try:
        return str(detect(text))
    except LangDetectException:
        return None


_PREFIXES = r"(Mr|St|Mrs|Ms|Dr)[.]"
_SUFFIXES = r"(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = (
    r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|"
    r"However\s|That\s|This\s|Wherever)"
)
_ACRONYMS = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = r"[.](com|net|org|io|gov|edu|me)"
_DIGITS = r"([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"


def split_into_sentences(text: str) -> list[str]:
    """Regex sentence splitter that protects abbreviations, decimals and URLs."""

    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(_PREFIXES, "\\1<prd>", text)
    text = re.sub(_WEBSITES, "<prd>\\1", text)
    text = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", text)
    text = re.sub(_MULTIPLE_DOTS, lambda m: "<prd>" * len(m.group(0)) + "<stop>", text)
    text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s([A-Za-z])[.] ", " \\1<prd> ", text)
    text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
    text = re.sub(r"([A-Za-z])[.]([A-Za-z])[.]([A-Za-z])[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(r"([A-Za-z])[.]([A-Za-z])[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", text)
    text = re.sub(r" ([A-Za-z])[.]", " \\1<prd>", text)
    text = (
        text.replace(".\u201d", "\u201d.")
        .replace('."', '".')
        .replace('!"', '"!')
        .replace('?"', '"?')
    )
    text = text.replace(".", ".<stop>").replace("?", "?<stop>").replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = [s.strip() for s in text.split("<stop>")]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _first_word(paragraph: str) -> str:
    word = paragraph.split()[0].strip().lstrip("'").lstrip('"')
    out = ""
    for letter in word:
        if letter in {".", ",", "?", "!", "'", '"'}:
            break
        out += letter.lower()
    return out


def _check(instruction_id: str) -> Checker:
    if instruction_id == "keywords:existence":
        return lambda r, k: all(re.search(w, r, re.IGNORECASE) for w in k["keywords"])
    if instruction_id == "keywords:frequency":
        return lambda r, k: _relation(
            len(re.findall(k["keyword"], r, re.IGNORECASE)), k["frequency"], k["relation"]
        )
    if instruction_id == "keywords:forbidden_words":
        return lambda r, k: (
            not any(re.search(r"\b" + w + r"\b", r, re.IGNORECASE) for w in k["forbidden_words"])
        )
    if instruction_id == "keywords:letter_frequency":
        return lambda r, k: _relation(
            Counter(r.lower())[k["letter"].lower()], k["let_frequency"], k["let_relation"]
        )
    if instruction_id == "language:response_language":
        return lambda r, k: (lambda lang: lang is None or lang == k["language"])(
            _detect_language(r)
        )
    if instruction_id == "length_constraints:number_sentences":
        return lambda r, k: _relation(
            len(split_into_sentences(r)), k["num_sentences"], k["relation"]
        )
    if instruction_id == "length_constraints:number_paragraphs":
        return _number_paragraphs
    if instruction_id == "length_constraints:number_words":
        return lambda r, k: _relation(count_words(r), k["num_words"], k["relation"])
    if instruction_id == "length_constraints:nth_paragraph_first_word":
        return _nth_paragraph_first_word
    if instruction_id == "detectable_content:number_placeholders":
        return lambda r, k: len(re.findall(r"\[.*?\]", r)) >= k["num_placeholders"]
    if instruction_id == "detectable_content:postscript":
        return _postscript
    if instruction_id == "detectable_format:number_bullet_lists":
        return lambda r, k: (
            (
                len(re.findall(r"^\s*\*[^\*].*$", r, re.MULTILINE))
                + len(re.findall(r"^\s*-.*$", r, re.MULTILINE))
            )
            == k["num_bullets"]
        )
    if instruction_id == "detectable_format:constrained_response":
        options = ("My answer is yes.", "My answer is no.", "My answer is maybe.")
        return lambda r, k: any(option in r.strip() for option in options)
    if instruction_id == "detectable_format:number_highlighted_sections":
        return _highlights
    if instruction_id == "detectable_format:multiple_sections":
        return lambda r, k: (
            (len(re.split(r"\s?" + k["section_spliter"] + r"\s?\d+\s?", r)) - 1)
            >= k["num_sections"]
        )
    if instruction_id == "detectable_format:json_format":
        return _json_format
    if instruction_id == "detectable_format:title":
        return lambda r, k: any(
            t.lstrip("<").rstrip(">").strip() for t in re.findall(r"<<[^\n]+>>", r)
        )
    if instruction_id == "combination:two_responses":
        return _two_responses
    if instruction_id == "combination:repeat_prompt":
        return lambda r, k: r.strip().lower().startswith(k["prompt_to_repeat"].strip().lower())
    if instruction_id == "startend:end_checker":
        return lambda r, k: r.strip().strip('"').lower().endswith(k["end_phrase"].strip().lower())
    if instruction_id == "change_case:capital_word_frequency":
        return lambda r, k: _relation(
            sum(1 for w in re.findall(r"\w+|[^\w\s]", r) if w.isupper()),
            k["capital_frequency"],
            k["capital_relation"],
        )
    if instruction_id == "change_case:english_capital":
        return lambda r, k: (
            r.isupper() and (lambda lang: lang is None or lang == "en")(_detect_language(r))
        )
    if instruction_id == "change_case:english_lowercase":
        return lambda r, k: (
            r.islower() and (lambda lang: lang is None or lang == "en")(_detect_language(r))
        )
    if instruction_id == "punctuation:no_comma":
        return lambda r, k: "," not in r
    if instruction_id == "startend:quotation":
        return lambda r, k: len(r.strip()) > 1 and r.strip()[0] == '"' and r.strip()[-1] == '"'
    raise KeyError(f"unknown IFEval instruction {instruction_id!r}")


def _number_paragraphs(response: str, kwargs: Mapping[str, Any]) -> bool:
    paragraphs = re.split(r"\s?\*\*\*\s?", response)
    count = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index in (0, len(paragraphs) - 1):
                count -= 1
            else:
                return False
    return bool(count == kwargs["num_paragraphs"])


def _nth_paragraph_first_word(response: str, kwargs: Mapping[str, Any]) -> bool:
    paragraphs = re.split(r"\n\n", response)
    count = sum(1 for p in paragraphs if p.strip())
    nth = kwargs["nth_paragraph"]
    if nth > count or nth > len(paragraphs):
        return False
    paragraph = paragraphs[nth - 1].strip()
    if not paragraph:
        return False
    return bool(
        count == kwargs["num_paragraphs"] and _first_word(paragraph) == kwargs["first_word"]
    )


def _postscript(response: str, kwargs: Mapping[str, Any]) -> bool:
    marker = kwargs["postscript_marker"]
    value = response.lower()
    if marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + marker.lower() + r".*$"
    return bool(re.findall(pattern, value, flags=re.MULTILINE))


def _highlights(response: str, kwargs: Mapping[str, Any]) -> bool:
    single = [h for h in re.findall(r"\*[^\n\*]*\*", response) if h.strip("*").strip()]
    double = [
        h
        for h in re.findall(r"\*\*[^\n\*]*\*\*", response)
        if h.removeprefix("**").removesuffix("**").strip()
    ]
    return len(single) + len(double) >= int(kwargs["num_highlights"])


def _json_format(response: str, kwargs: Mapping[str, Any]) -> bool:
    value = response.strip()
    for prefix in ("```json", "```Json", "```JSON", "```"):
        value = value.removeprefix(prefix)
    value = value.removesuffix("```").strip()
    try:
        json.loads(value)
    except ValueError:
        return False
    return True


def _two_responses(response: str, kwargs: Mapping[str, Any]) -> bool:
    parts = response.split("******")
    valid = []
    for index, part in enumerate(parts):
        if not part.strip():
            if index not in (0, len(parts) - 1):
                return False
        else:
            valid.append(part)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


INSTRUCTION_IDS: tuple[str, ...] = (
    "keywords:existence",
    "keywords:frequency",
    "keywords:forbidden_words",
    "keywords:letter_frequency",
    "language:response_language",
    "length_constraints:number_sentences",
    "length_constraints:number_paragraphs",
    "length_constraints:number_words",
    "length_constraints:nth_paragraph_first_word",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "detectable_format:number_bullet_lists",
    "detectable_format:constrained_response",
    "detectable_format:number_highlighted_sections",
    "detectable_format:multiple_sections",
    "detectable_format:json_format",
    "detectable_format:title",
    "combination:two_responses",
    "combination:repeat_prompt",
    "startend:end_checker",
    "change_case:capital_word_frequency",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "punctuation:no_comma",
    "startend:quotation",
)


def follows(instruction_id: str, kwargs: Mapping[str, Any], response: str) -> bool:
    """Strict check of one instruction on one response."""

    clean = {key: value for key, value in kwargs.items() if value is not None}
    return bool(response.strip()) and _check(instruction_id)(response, clean)


def loose_variants(response: str) -> list[str]:
    lines = response.split("\n")
    first = "\n".join(lines[1:]).strip()
    last = "\n".join(lines[:-1]).strip()
    both = "\n".join(lines[1:-1]).strip()
    variants = [response, first, last, both]
    return variants + [v.replace("*", "") for v in variants]


def follows_loose(instruction_id: str, kwargs: Mapping[str, Any], response: str) -> bool:
    return any(follows(instruction_id, kwargs, variant) for variant in loose_variants(response))


def judge(example: TaskExample, response: str) -> dict[str, list[bool]]:
    ids: Sequence[str] = example.meta["instruction_id_list"]
    kwargs: Sequence[Mapping[str, Any]] = example.meta["kwargs"]
    strict = [follows(i, k, response) for i, k in zip(ids, kwargs, strict=True)]
    loose = [follows_loose(i, k, response) for i, k in zip(ids, kwargs, strict=True)]
    return {"strict": strict, "loose": loose}


class IFEval(Task):
    """Instruction following with programmatic verification."""

    name = "ifeval"
    contract = ""
    seed_instruction = "You are a helpful assistant."
    description = (
        "User requests that each carry one to three verifiable formatting or content "
        "constraints (length, keywords, casing, JSON, bullet lists, sections, and so on)."
    )
    max_tokens = 1024

    def extract(self, output: str) -> str:
        return output

    def score(self, example: TaskExample, output: str) -> float:
        verdicts = judge(example, output)["strict"]
        return sum(verdicts) / len(verdicts)

    def metrics(self, example: TaskExample, output: str) -> dict[str, float]:
        verdicts = judge(example, output)
        return {
            "prompt_strict": float(all(verdicts["strict"])),
            "prompt_loose": float(all(verdicts["loose"])),
            "inst_strict": sum(verdicts["strict"]) / len(verdicts["strict"]),
            "inst_loose": sum(verdicts["loose"]) / len(verdicts["loose"]),
            "n_instructions": float(len(verdicts["strict"])),
        }

    def feedback(self, example: TaskExample, output: str) -> str:
        verdicts = judge(example, output)["strict"]
        ids = example.meta["instruction_id_list"]
        failed = [i for i, ok in zip(ids, verdicts, strict=True) if not ok]
        if not failed:
            return "Every verifiable instruction in the request was followed."
        return "The response broke these instructions from the request: " + ", ".join(failed)

    def gold_output(self, example: TaskExample) -> str:
        raise ValueError("IFEval has no reference responses; use bootstrapped demonstrations")


def load(root: Path) -> list[TaskExample]:
    out: list[TaskExample] = []
    with open(root / "ifeval" / "input_data.jsonl", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            out.append(
                TaskExample(
                    id=f"ifeval-{record['key']}",
                    question=record["prompt"],
                    answer="",
                    meta={
                        "instruction_id_list": list(record["instruction_id_list"]),
                        "kwargs": [dict(k) for k in record["kwargs"]],
                    },
                )
            )
    return out
