"""Tests for the benchmark loaders, answer extraction and scoring rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from prompt_optimizer.tasks import bbh, gsm8k, hotpotqa, math500
from prompt_optimizer.tasks.base import TaskExample
from prompt_optimizer.tasks.registry import (
    SplitConfig,
    Splits,
    draw,
    draw_grouped,
    load_task,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    gsm = [
        {
            "question": f"Tom has {i} apples and buys 2. How many?",
            "answer": f"{i}+2=<<{i}+2={i + 2}>>{i + 2}\n#### {i + 2:,}",
        }
        for i in range(1, 13)
    ]
    _write_jsonl(root / "gsm8k" / "train.jsonl", gsm[:8])
    _write_jsonl(root / "gsm8k" / "test.jsonl", gsm[8:])
    for name in bbh.BBH_TASKS:
        examples = [
            {"input": f"{name} question {i}", "target": "(A)" if i % 2 else "True"}
            for i in range(6)
        ]
        (root / "bbh").mkdir(parents=True, exist_ok=True)
        (root / "bbh" / f"{name}.json").write_text(json.dumps({"examples": examples}))
        prompt = (
            "BENCHMARK DATA SHOULD NEVER APPEAR. canary GUID x\n-----\nDescribe "
            + name
            + ".\n\nQ: ..."
        )
        (root / "bbh" / "cot-prompts").mkdir(parents=True, exist_ok=True)
        (root / "bbh" / "cot-prompts" / f"{name}.txt").write_text(prompt)
    hotpot = [
        {
            "id": f"h{i}",
            "question": f"Who is {i}?",
            "answer": f"Person {i}",
            "type": "bridge",
            "level": "easy",
            "context": {"title": ["A", "B"], "sentences": [["One.", " Two."], ["Three."]]},
        }
        for i in range(10)
    ]
    _write_jsonl(root / "hotpotqa" / "distractor_validation.jsonl", hotpot)
    math_rows = [
        {
            "problem": f"What is {i}/2?",
            "solution": f"It is $\\boxed{{\\frac{{{i}}}{{2}}}}$.",
            "answer": f"\\frac{{{i}}}{{2}}",
            "subject": "Algebra",
            "level": 1,
            "unique_id": f"u{i}",
        }
        for i in range(1, 11)
    ]
    _write_jsonl(root / "math" / "train.jsonl", math_rows[:7])
    _write_jsonl(root / "math" / "test.jsonl", math_rows[7:])
    ifeval_rows = [
        {
            "key": i,
            "prompt": f"Write {i} words.",
            "instruction_id_list": ["punctuation:no_comma"],
            "kwargs": [{}],
        }
        for i in range(9)
    ]
    _write_jsonl(root / "ifeval" / "input_data.jsonl", ifeval_rows)
    return root


class TestRegistry:
    def test_all_tasks_load_with_disjoint_splits(self, data_root: Path) -> None:
        configs = {
            "gsm8k": SplitConfig(3, 2),
            "bbh": SplitConfig(2, 1, 2),
            "hotpotqa": SplitConfig(3, 2, 4),
            "math500": SplitConfig(3, 2),
            "ifeval": SplitConfig(3, 2),
        }
        sizes = {}
        for name, cfg in configs.items():
            task, splits = load_task(name, data_root, cfg)
            assert task.name == name
            sizes[name] = splits.sizes()
        assert sizes["gsm8k"] == {"train": 3, "val": 2, "test": 4}
        assert sizes["bbh"] == {"train": 54, "val": 27, "test": 54}
        assert sizes["hotpotqa"] == {"train": 3, "val": 2, "test": 4}
        assert sizes["math500"] == {"train": 3, "val": 2, "test": 3}
        assert sizes["ifeval"] == {"train": 3, "val": 2, "test": 4}

    def test_test_subsets_and_unknown(self, data_root: Path) -> None:
        _, splits = load_task("gsm8k", data_root, SplitConfig(2, 2, 3))
        assert len(splits.test) == 3
        _, splits = load_task("math500", data_root, SplitConfig(2, 2, 1))
        assert len(splits.test) == 1
        with pytest.raises(KeyError):
            load_task("nope", data_root)

    def test_split_overlap_is_rejected(self) -> None:
        ex = TaskExample("1", "q", "a")
        with pytest.raises(ValueError):
            Splits((ex,), (ex,), ())

    def test_draw_is_seeded_and_bounded(self) -> None:
        pool = [TaskExample(str(i), "q", "a") for i in range(20)]
        assert draw(pool, [5, 5], seed=1) == draw(pool, [5, 5], seed=1)
        assert draw(pool, [5], seed=1) != draw(pool, [5], seed=2)
        with pytest.raises(ValueError):
            draw(pool, [15, 10])

    @given(st.integers(min_value=0, max_value=6), st.integers(min_value=0, max_value=6))
    def test_draw_grouped_is_disjoint(self, a: int, b: int) -> None:
        pool = [TaskExample(f"{g}{i}", "q", "a", group=g) for g in "xyz" for i in range(8)]
        first, second, rest = draw_grouped(pool, [a, b, None])
        ids = [e.id for e in (*first, *second, *rest)]
        assert len(ids) == len(set(ids)) == 24


class TestGSM8K:
    task = gsm8k.GSM8K()

    @pytest.mark.parametrize(
        ("output", "expected"),
        [
            ("so 3+4=7\nAnswer: 7", "7"),
            ("Answer: $1,250.00", "1250"),
            ("Answer: 0.50", "0.5"),
            ("The total is 18 dollars", "18"),
            ("Answer: 18 dollars per 2 hours", "18"),
            ("no digits", ""),
            ("Answer: -3", "-3"),
        ],
    )
    def test_extract(self, output: str, expected: str) -> None:
        assert self.task.extract(output) == expected

    def test_score_and_parse(self) -> None:
        record = {"question": " Q? ", "answer": "a <<2*3=6>>6 b\n#### 1,000"}
        ex = gsm8k.parse_record(record, 0, "train")
        assert ex.answer == "1000" and "<<" not in ex.rationale
        assert self.task.score(ex, "Answer: 1000.0") == 1.0
        assert self.task.score(ex, "Answer: 999") == 0.0
        assert self.task.score(ex, "nothing") == 0.0
        assert gsm8k.normalize_number("abc") == "abc"
        weird = TaskExample("w", "q", "n/a")
        assert self.task.score(weird, "Answer: 3") == 0.0


class TestBBH:
    task = bbh.BBH()

    @pytest.mark.parametrize(
        ("output", "target", "score"),
        [
            ("reasoning\nAnswer: (B)", "(B)", 1.0),
            ("Answer: B", "(B)", 1.0),
            ("Answer: (C) the third one", "(C)", 1.0),
            ("So the answer is (D).", "(D)", 1.0),
            ("Answer: true", "True", 1.0),
            ("Answer: **False**", "True", 0.0),
            ("Answer: It is valid.", "valid", 1.0),
            ("Answer: invalid", "valid", 0.0),
            ("Answer: Yes", "Yes", 1.0),
            ("Answer: 1,234", "1234", 1.0),
            ("Answer: -12", "-12", 1.0),
            ("Answer: apple, banana, cherry", "apple banana cherry", 1.0),
            ("Answer: ) ] >", ") ] >", 1.0),
            ("final line only", "(A)", 0.0),
            ("", "(A)", 0.0),
            ("Answer: maybe", "Yes", 0.0),
            ("Answer: none", "7", 0.0),
        ],
    )
    def test_scoring(self, output: str, target: str, score: float) -> None:
        ex = TaskExample("b", "q", target, group="g")
        assert self.task.score(ex, output) == score

    def test_description(self) -> None:
        text = "BENCHMARK DATA SHOULD NEVER APPEAR\n-----\nSort the words.\n\nQ: x"
        assert bbh.task_description(text) == "Sort the words."
        assert bbh.task_description("") == ""

    def test_load_without_header(self, data_root: Path) -> None:
        (data_root / "bbh" / "cot-prompts" / "navigate.txt").write_text("")
        examples = bbh.load(data_root, ("navigate",))
        assert examples[0].question == "navigate question 0"


class TestHotpot:
    task = hotpotqa.HotpotQA()

    def test_normalization_and_metrics(self) -> None:
        assert hotpotqa.normalize("The  Eiffel-Tower!") == "eiffeltower"
        assert hotpotqa.exact_match("the Beatles", "Beatles") == 1.0
        assert hotpotqa.f1("yes", "no") == 0.0
        assert hotpotqa.f1("Barack Obama", "Obama") == pytest.approx(2 / 3)
        assert hotpotqa.f1("x", "y") == 0.0
        ex = TaskExample("h", "q", "Paris")
        assert self.task.score(ex, "thinking\nAnswer: Paris.") == 1.0
        assert self.task.metrics(ex, "Paris") == {"em": 1.0, "f1": 1.0}
        assert self.task.extract("") == ""

    def test_render(self, data_root: Path) -> None:
        examples = hotpotqa.load(data_root)
        assert examples[0].question.startswith("Paragraphs:\n[1] A: One. Two.")
        assert examples[0].meta["type"] == "bridge"


class TestMath:
    task = math500.MATH500()

    def test_boxed(self) -> None:
        assert math500.last_boxed("a \\boxed{1} b \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
        assert math500.last_boxed("\\fbox{x}") == "x"
        assert math500.last_boxed("\\boxed 5$") == "5"
        assert math500.last_boxed("\\boxed{unbalanced") is None
        assert math500.last_boxed("none") is None

    @pytest.mark.parametrize(
        ("predicted", "gold", "equal"),
        [
            ("\\dfrac{1}{2}", "\\frac{1}{2}", True),
            ("0.5", "\\frac12", True),
            ("1/2", "\\frac{1}{2}", True),
            ("\\left( 3, \\frac{\\pi}{2} \\right)", "(3,\\frac{\\pi}{2})", True),
            ("90^\\circ", "90", True),
            ("50\\%", "50", True),
            ("x = 5", "5", True),
            ("\\sqrt2", "\\sqrt{2}", True),
            ("10\\text{ cm}", "10", True),
            ("1,000", "1000", True),
            (".5", "0.5", True),
            ("\\$4", "4", True),
            ("3", "4", False),
            ("\\frac{a}{b}", "\\frac{b}{a}", False),
            ("\\frac1{2}", "\\frac{1}{2}", True),
            ("2.0000001", "2", True),
        ],
    )
    def test_equivalence(self, predicted: str, gold: str, equal: bool) -> None:
        assert math500.is_equiv(predicted, gold) is equal

    def test_task(self, data_root: Path) -> None:
        examples = math500.load(data_root, "train")
        ex = examples[0]
        assert self.task.score(ex, "so \\boxed{\\frac{1}{2}}") == 1.0
        assert self.task.score(ex, "Answer: $0.5$") == 1.0
        assert self.task.score(ex, "nothing") == 0.0
        assert self.task.gold_output(ex) == ex.rationale
        plain = TaskExample("m", "q", "3", rationale="Add.")
        assert self.task.gold_output(plain).endswith("\\boxed{3}")
        assert math500.strip_string("") == ""
