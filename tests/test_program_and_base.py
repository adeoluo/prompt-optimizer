"""Tests for programs, the task base class and the shared optimizer machinery."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from prompt_optimizer.backends import CachedLM, CostMeter, FakeLM
from prompt_optimizer.optim.base import (
    Budget,
    BudgetExhaustedError,
    Evaluator,
    Optimizer,
    OptimizerContext,
    extract_instruction,
    format_examples,
)
from prompt_optimizer.program import Demo, PromptProgram
from prompt_optimizer.tasks.base import TaskExample, last_answer_line, mean
from prompt_optimizer.tasks.synthetic import ArithmeticTask

TASK = ArithmeticTask()


class TestProgram:
    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            PromptProgram("  ")
        with pytest.raises(ValueError):
            Demo("", "x")

    def test_messages_layout_and_groups(self) -> None:
        program = PromptProgram(
            "Solve.",
            (Demo("q1", "a1"), Demo("q2", "a2", group="g2"), Demo("q3", "a3", group="g3")),
        )
        msgs = program.messages("Contract.", "query", group="g2")
        assert msgs[0].role == "system" and msgs[0].content == "Solve.\n\nContract."
        assert [m.content for m in msgs[1:]] == ["q1", "a1", "q2", "a2", "query"]
        assert program.system_text("") == "Solve."
        req = program.request("C", "q", max_tokens=9, group="g3", purpose="task")
        assert req.max_tokens == 9 and len(req.messages) == 6

    def test_fingerprint_and_round_trip(self) -> None:
        program = PromptProgram("Solve.", (Demo("q", "a", "id1", "g"),))
        clone = PromptProgram.from_dict(program.to_dict())
        assert clone == program
        assert clone.fingerprint() == program.fingerprint()
        assert program.with_instruction(" Other ").instruction == "Other"
        assert program.with_demos([]).demos == ()
        assert program.fingerprint() != program.with_instruction("Else.").fingerprint()

    @given(st.text(min_size=1).filter(str.strip), st.text(min_size=1).filter(str.strip))
    def test_fingerprint_distinguishes_instructions(self, a: str, b: str) -> None:
        same = PromptProgram(a).fingerprint() == PromptProgram(b).fingerprint()
        assert same == (a == b)


class TestTaskBase:
    def test_last_answer_line(self) -> None:
        assert last_answer_line("x\nAnswer: 4\nAnswer: 5") == "5"
        assert last_answer_line("The answer is: 7") == "7"
        assert last_answer_line("nothing") is None

    def test_arithmetic_task_scoring_and_feedback(self) -> None:
        ex = TaskExample("1", "What is 2 + 3?", "5", rationale="2 + 3 = 5.")
        assert TASK.score(ex, "blah\nAnswer: 5") == 1.0
        assert TASK.score(ex, "Answer: 6") == 0.0
        assert TASK.extract("no numbers") == ""
        assert TASK.extract("I think 3 then 5") == "5"
        assert TASK.feedback(ex, "Answer: 5").startswith("Correct")
        assert "reference answer is '5'" in TASK.feedback(ex, "Answer: 9")
        demo = TASK.gold_demo(ex)
        assert demo.output.endswith("Answer: 5") and demo.source_id == "1"
        assert TASK.gold_output(TaskExample("2", "q", "1")) == "Answer: 1"

    def test_make_is_seeded(self) -> None:
        assert ArithmeticTask.make(5, seed=1) == ArithmeticTask.make(5, seed=1)
        assert ArithmeticTask.make(5, seed=1) != ArithmeticTask.make(5, seed=2)

    def test_mean(self) -> None:
        assert mean([]) == 0.0
        assert mean([1.0, 0.0]) == 0.5


class TestBudgetAndEvaluator:
    def test_budget(self) -> None:
        budget = Budget(3)
        budget.charge(2)
        assert budget.remaining == 1 and not budget.can_afford(2)
        with pytest.raises(BudgetExhaustedError):
            budget.charge(2)
        assert Budget().remaining == float("inf")

    def test_evaluator_memoizes_and_charges_new_pairs_only(self) -> None:
        examples = ArithmeticTask.make(6)
        evaluator = Evaluator(TASK, FakeLM(), Budget(10))
        program = PromptProgram("Solve step by step.")
        first = evaluator.evaluate(program, examples[:4])
        second = evaluator.evaluate(program, examples)
        assert evaluator.budget.used == 6
        assert second.scores[:4] == first.scores
        assert set(second.by_id()) == {ex.id for ex in examples}
        assert evaluator.known(program, examples[0])
        with pytest.raises(BudgetExhaustedError):
            evaluator.evaluate(PromptProgram("Other."), examples)
        free = evaluator.evaluate(PromptProgram("Other."), examples, charge=False)
        assert len(free.rollouts) == 6


class TestExtractInstruction:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("<instruction>Do X.</instruction>", "Do X."),
            ("<INSTRUCTION>Do Y.", "Do Y."),
            ("Some text\n```\nDo Z.\n```", "Do Z."),
            ("New instruction: Do W.", "Do W."),
            ('"Quoted."', "Quoted."),
            ("   ", None),
            ("<instruction></instruction>", None),
        ],
    )
    def test_cases(self, text: str, expected: str | None) -> None:
        assert extract_instruction(text) == expected

    def test_truncates_long_text(self) -> None:
        text = "Do this. " * 400
        out = extract_instruction(text)
        assert out is not None and len(out) <= 2000 and out.endswith(".")


class _Noop(Optimizer):
    name = "noop"

    def _search(self, ctx: OptimizerContext, seed_program: PromptProgram) -> None:
        ctx.score_val(seed_program.with_instruction("Solve step by step and verify."))
        ctx.score_batch(seed_program, ctx.trainset[:2])
        ctx.score_val(seed_program)  # memoized


class TestContext:
    def test_compile_tracks_best_and_trials(self) -> None:
        train, val = ArithmeticTask.make(10, seed=1), ArithmeticTask.make(20, seed=2)
        result = _Noop().compile(TASK, FakeLM(), train, val)
        assert result.program.instruction == "Solve step by step and verify."
        assert result.rollouts_used == 42
        assert [t.split for t in result.trials] == ["val", "val", "train"]
        assert result.info["stopped"] == "completed"

    def test_budget_stop_and_too_small(self) -> None:
        train, val = ArithmeticTask.make(10, seed=1), ArithmeticTask.make(20, seed=2)
        result = _Noop().compile(TASK, FakeLM(), train, val, budget=Budget(30))
        assert result.info["stopped"] == "budget"
        with pytest.raises(BudgetExhaustedError):
            _Noop().compile(TASK, FakeLM(), train, val, budget=Budget(5))

    def test_context_requires_data_and_samples(self) -> None:
        with pytest.raises(ValueError):
            OptimizerContext(TASK, FakeLM(), [], ArithmeticTask.make(2))
        ctx = OptimizerContext(TASK, FakeLM(), ArithmeticTask.make(5), ArithmeticTask.make(5))
        assert len(ctx.sample(ctx.trainset, 99)) == 5
        assert ctx.known_val_score(PromptProgram("x")) is None

    def test_proposals_are_parsed_and_deduplicated(self) -> None:
        meter = CostMeter()
        lm = CachedLM(FakeLM(), meter=meter)
        ctx = OptimizerContext(TASK, lm, ArithmeticTask.make(5), ArithmeticTask.make(5))
        out = ctx.propose_instructions("sys", "make it better", n=6)
        assert out and len(out) == len(set(out))
        assert meter.total(purpose="propose").calls == 6
        assert ctx.ask("sys", "x", purpose="critique", temperature=0.0)[0]

    def test_format_examples(self) -> None:
        text = format_examples(ArithmeticTask.make(5), TASK, limit=2)
        assert text.count("Reference output") == 2
