"""Tests for the literature optimizers, TPE, Pareto selection and demo bootstrapping.

Every optimizer runs against the deterministic ``FakeLM`` on the synthetic
arithmetic task, where better instructions and correct demonstrations raise
accuracy (see ``backends.fake``). The tests check the shared contract (budget,
determinism, selection by validation, test split never touched) and each
method's characteristic behaviour.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from prompt_optimizer.backends import CachedLM, FakeLM, LMRequest, LMResponse
from prompt_optimizer.optim import (
    GEPA,
    OPRO,
    OPTIMIZERS,
    BootstrapFewShotRS,
    Budget,
    MIPROv2,
    OptimizerContext,
    PromptBreeder,
    TextGrad,
    build,
)
from prompt_optimizer.optim.base import clip, format_rollouts
from prompt_optimizer.optim.demos import (
    bootstrap_traces,
    first_demo_set,
    group_demos,
    labeled_demos,
    sample_demo_set,
    teacher_pool,
)
from prompt_optimizer.optim.pareto import instance_fronts, remove_dominated, select_candidate
from prompt_optimizer.optim.tpe import TPESampler
from prompt_optimizer.program import Demo, PromptProgram
from prompt_optimizer.tasks.base import TaskExample
from prompt_optimizer.tasks.synthetic import ArithmeticTask

TASK = ArithmeticTask()
TRAIN = ArithmeticTask.make(60, seed=1, prefix="tr")
VAL = ArithmeticTask.make(30, seed=2, prefix="va")


def _run(name: str, budget: int = 400, seed: int = 0, **kw: object):  # type: ignore[no-untyped-def]
    lm = CachedLM(FakeLM())
    return build(name, **kw).compile(TASK, lm, TRAIN, VAL, budget=Budget(budget), seed=seed), lm


class TestSharedContract:
    @pytest.mark.parametrize("name", sorted(OPTIMIZERS))
    def test_improves_on_seed_within_budget(self, name: str) -> None:
        result, _ = _run(name)
        seed_score = result.trials[0].score
        assert result.trials[0].note == "seed" and result.trials[0].split == "val"
        assert result.val_score >= seed_score
        assert result.val_score > 0.4  # the fake seed program scores about 0.25
        assert result.rollouts_used <= 400
        assert result.info["optimizer"] == name

    @pytest.mark.parametrize("name", sorted(OPTIMIZERS))
    def test_deterministic_for_a_seed(self, name: str) -> None:
        first, _ = _run(name, seed=3)
        second, _ = _run(name, seed=3)
        assert first.program == second.program
        assert first.val_score == second.val_score
        assert [t.score for t in first.trials] == [t.score for t in second.trials]

    @pytest.mark.parametrize("name", sorted(OPTIMIZERS))
    def test_selected_program_has_the_best_validation_score(self, name: str) -> None:
        result, _ = _run(name)
        val_trials = [t for t in result.trials if t.split == "val"]
        assert result.val_score == max(t.score for t in val_trials)
        assert result.program.fingerprint() in {t.fingerprint for t in val_trials}

    @pytest.mark.parametrize("name", sorted(OPTIMIZERS))
    def test_tiny_budget_returns_the_seed(self, name: str) -> None:
        result, _ = _run(name, budget=len(VAL))
        assert result.program.instruction == TASK.seed_instruction
        assert result.rollouts_used <= len(VAL)

    @pytest.mark.parametrize("name", sorted(OPTIMIZERS))
    def test_never_sees_test_examples(self, name: str) -> None:
        lm = CachedLM(FakeLM())
        build(name).compile(TASK, lm, TRAIN, VAL, budget=Budget(300))
        inner = lm.inner
        assert isinstance(inner, FakeLM)
        test = ArithmeticTask.make(20, seed=99, prefix="te")
        test_questions = {ex.question for ex in test} - {ex.question for ex in TRAIN + VAL}
        seen = " ".join(m.content for r in inner.requests for m in r.messages)
        assert not any(q in seen for q in test_questions)

    def test_build_unknown(self) -> None:
        with pytest.raises(KeyError, match="unknown optimizer"):
            build("nope")

    def test_cost_is_metered_by_purpose(self) -> None:
        _, lm = _run("gepa")
        breakdown = lm.meter.breakdown()["default"]
        assert breakdown["task"]["calls"] > 0
        assert breakdown["reflect"]["calls"] > 0


class TestBootstrap:
    def test_candidates_include_first_labeled_and_random(self) -> None:
        result, _ = _run("bootstrap_fewshot")
        notes = {t.note for t in result.trials}
        assert {"bootstrap", "bootstrap-first", "labeled"} <= notes
        assert result.program.demos

    def test_demos_are_correct_traces(self) -> None:
        result, _ = _run("bootstrap_fewshot", budget=1000, n_candidates=3, use_labeled=False)
        for demo in result.program.demos:
            truth = TASK.score(next(ex for ex in TRAIN if ex.question == demo.input), demo.output)
            assert truth == 1.0

    def test_no_traces_keeps_seed(self) -> None:
        class Wrong(FakeLM):
            def answer(self, request: LMRequest) -> str:
                return "Answer: -99999" if request.purpose == "task" else super().answer(request)

        opt = BootstrapFewShotRS(use_labeled=False)
        result = opt.compile(TASK, Wrong(), TRAIN, VAL, budget=Budget(200))
        assert result.program.demos == ()


class TestAPEAndOPRO:
    def test_ape_scores_induced_and_variant_candidates(self) -> None:
        result, lm = _run("ape")
        notes = {t.note for t in result.trials}
        assert {"induced", "variant", "finalist"} <= notes
        meta = [r for r in lm.inner.requests if r.purpose == "propose"]  # type: ignore[attr-defined]
        assert "input/output pairs" in meta[0].messages[-1].content

    def test_opro_meta_prompt_orders_trajectory_ascending(self) -> None:
        opt = OPRO()
        ctx = OptimizerContext(TASK, FakeLM(), TRAIN, VAL)
        text = opt.meta_prompt(ctx, {"low": 0.1, "high": 0.9, "mid": 0.5}, TRAIN[:2])
        assert text.index("low") < text.index("mid") < text.index("high")
        assert "score:\n90" in text and "<INS>" in text

    def test_opro_history_is_capped(self) -> None:
        opt = OPRO(max_history=2)
        ctx = OptimizerContext(TASK, FakeLM(), TRAIN, VAL)
        text = opt.meta_prompt(ctx, {f"ins{i}": i / 10 for i in range(5)}, [])
        assert "ins4" in text and "ins3" in text and "ins0" not in text

    def test_opro_records_steps(self) -> None:
        result, _ = _run("opro")
        assert any(t.note == "opro" for t in result.trials)


class TestMIPRO:
    def test_runs_tpe_trials_and_full_evaluations(self) -> None:
        result, lm = _run("miprov2", budget=600)
        notes = [t.note for t in result.trials]
        assert notes.count("tpe") >= 6
        assert any(n.startswith("trial-") or n == "final" for n in notes)
        purposes = {r.purpose for r in lm.inner.requests}  # type: ignore[attr-defined]
        assert "summarize" in purposes

    def test_proposal_prompt_is_grounded(self) -> None:
        opt = MIPROv2()
        ctx = OptimizerContext(TASK, FakeLM(), TRAIN, VAL)
        demos = [Demo("What is 1 + 1?", "Answer: 2")]
        text = opt.proposal_prompt(ctx, "SUMMARY", demos, ["old one"], "Be brief.")
        for part in ("SUMMARY", TASK.description, "What is 1 + 1?", "old one", "Be brief."):
            assert part in text


class TestTextGrad:
    def test_gradient_and_step_calls(self) -> None:
        result, lm = _run("textgrad")
        purposes = [r.purpose for r in lm.inner.requests]  # type: ignore[attr-defined]
        assert "critique" in purposes and "propose" in purposes
        assert {"forward", "step"} <= {t.note for t in result.trials}

    def test_prompts_carry_variable_and_feedback(self) -> None:
        opt = TextGrad()
        grad = opt.gradient_prompt("Solve it.", "FAILURES", "Contract.")
        step = opt.step_prompt("Solve it.", "GRADIENT", [("older", 0.4)])
        assert "<VARIABLE>" in grad and "FAILURES" in grad
        assert "GRADIENT" in step and "older" in step and "0.40" in step

    def test_perfect_model_skips_backward(self) -> None:
        class Perfect(FakeLM):
            def skill(self, request: LMRequest) -> float:
                return 0.999

        result = TextGrad(iterations=2).compile(TASK, Perfect(), TRAIN, VAL, budget=Budget(200))
        assert result.program.instruction == TASK.seed_instruction


class TestGEPA:
    def test_pool_grows_with_reflection(self) -> None:
        result, _ = _run("gepa")
        accepted = [t for t in result.trials if t.split == "val"]
        assert len(accepted) >= 2
        assert any(t.note == "reflect" for t in result.trials)

    def test_crossover_branch(self) -> None:
        result, _ = _run("gepa", budget=600, crossover_every=2)
        assert any(t.note in {"crossover", "parent"} for t in result.trials)

    def test_reflection_prompt(self) -> None:
        text = GEPA().reflection_prompt("Do it.", "EXAMPLES", "Contract.")
        assert "Do it." in text and "EXAMPLES" in text and "Contract." in text
        assert "A" in GEPA().crossover_prompt("A", "B")


class TestPromptBreeder:
    def test_evolves_with_every_operator(self) -> None:
        opt = PromptBreeder(generations=12, tournaments_per_generation=3)
        lm = CachedLM(FakeLM())
        result = opt.compile(TASK, lm, TRAIN, VAL, budget=Budget(2000), seed=1)
        prompts = " ".join(
            r.messages[-1].content
            for r in lm.inner.requests
            if r.purpose == "mutate"  # type: ignore[attr-defined]
        )
        for marker in (
            "INSTRUCTION MUTANT",
            "Continue the list",
            "history of the best",
            "Improve the following prompt",
            "Working out",
            "Thinking style",
        ):
            assert marker in prompts
        assert result.val_score > 0.4

    def test_operator_prompts(self) -> None:
        opt = PromptBreeder()
        ctx = OptimizerContext(TASK, FakeLM(), TRAIN, VAL)
        assert "Thinking style: X" in opt.zero_order(ctx, "X")
        assert "INSTRUCTION: P" in opt.first_order("M", "P")
        assert "1. a" in opt.eda(["a", "b"]) and "2. b" in opt.lineage(["a", "b"])
        assert "PROMPT: M" in opt.hyper("M")
        assert "Q" in opt.lamarckian(ctx, "Q", "W")

    def test_unaffordable_population_returns_seed(self) -> None:
        result = PromptBreeder().compile(TASK, FakeLM(), TRAIN, VAL, budget=Budget(len(VAL) + 5))
        assert result.program.instruction == TASK.seed_instruction


class TestTPE:
    def test_startup_is_random_then_exploits(self) -> None:
        sampler = TPESampler({"a": 4, "b": 3}, seed=0, n_startup=4)
        rng = random.Random(0)
        for _ in range(30):
            params = sampler.suggest()
            value = (1.0 if params["a"] == 2 else 0.0) + 0.1 * rng.random()
            sampler.observe(params, value)
        late = [p["a"] for p, _ in sampler.observations[-10:]]
        assert late.count(2) >= 7

    def test_density_is_smoothed_and_normalized(self) -> None:
        sampler = TPESampler({"a": 3})
        density = sampler.density([{"a": 0}, {"a": 0}], "a")
        assert abs(sum(density) - 1.0) < 1e-12
        assert density[0] > density[1] == density[2] > 0

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            TPESampler({})
        with pytest.raises(ValueError):
            TPESampler({"a": 2}, gamma=1.0)
        with pytest.raises(ValueError):
            TPESampler({"a": 2}).observe({"a": 5}, 1.0)

    @settings(max_examples=40, deadline=None)
    @given(
        st.dictionaries(st.sampled_from("abcd"), st.integers(1, 6), min_size=1),
        st.integers(0, 20),
        st.integers(0, 1000),
    )
    def test_suggestions_stay_in_space(self, space: dict[str, int], n_obs: int, seed: int) -> None:
        sampler = TPESampler(space, seed=seed, n_startup=2)
        rng = random.Random(seed)
        for _ in range(n_obs):
            params = sampler.suggest()
            assert set(params) == set(space)
            assert all(0 <= params[k] < space[k] for k in space)
            sampler.observe(params, rng.random())

    def test_split_puts_best_in_good(self) -> None:
        sampler = TPESampler({"a": 2}, gamma=0.25)
        for v in range(8):
            sampler.observe({"a": v % 2}, float(v))
        good, bad = sampler.split()
        assert len(good) == 2 and len(bad) == 6


class TestPareto:
    def test_fronts(self) -> None:
        fronts = instance_fronts({"x": [1, 0, 1], "y": [1, 1, 0]})
        assert fronts == [{"x", "y"}, {"y"}, {"x"}]
        assert instance_fronts({}) == []
        with pytest.raises(ValueError):
            instance_fronts({"x": [1], "y": [1, 0]})

    def test_dominated_candidate_is_removed(self) -> None:
        weights = remove_dominated({"weak": [1, 0, 0], "strong": [1, 1, 0], "other": [0, 0, 1]})
        assert "weak" not in weights
        assert weights == {"other": 1, "strong": 2}

    def test_identical_candidates_keep_one(self) -> None:
        weights = remove_dominated({"a": [1, 1], "b": [1, 1]})
        assert len(weights) == 1

    def test_selection_is_proportional(self) -> None:
        scores = {"a": [1, 1, 1, 0], "b": [0, 0, 0, 1]}
        rng = random.Random(0)
        draws = [select_candidate(scores, rng) for _ in range(4000)]
        assert 0.70 < draws.count("a") / len(draws) < 0.80

    def test_all_zero_scores(self) -> None:
        assert select_candidate({"a": [0, 0]}, random.Random(0)) == "a"

    @settings(max_examples=60, deadline=None)
    @given(
        st.lists(
            st.lists(st.sampled_from([0.0, 0.5, 1.0]), min_size=4, max_size=4),
            min_size=1,
            max_size=6,
        )
    )
    def test_survivors_cover_every_front(self, rows: list[list[float]]) -> None:
        scores = {f"c{i}": row for i, row in enumerate(rows)}
        survivors = remove_dominated(scores)
        assert survivors
        for front in instance_fronts(scores):
            assert front & set(survivors)
        assert select_candidate(scores, random.Random(1)) in survivors


class TestDemos:
    def _ctx(self, train: Sequence[TaskExample] = TRAIN) -> OptimizerContext:
        return OptimizerContext(TASK, FakeLM(), train, VAL, budget=Budget(500))

    def test_bootstrap_keeps_correct_untruncated_traces(self) -> None:
        class Trunc(FakeLM):
            def generate(self, requests: Sequence[LMRequest]) -> list[LMResponse]:
                out = super().generate(requests)
                return [
                    LMResponse(
                        r.text, r.prompt_tokens, r.completion_tokens, "length" if i % 2 else "stop"
                    )
                    for i, r in enumerate(out)
                ]

        ctx = OptimizerContext(TASK, Trunc(), TRAIN, VAL)
        demos = bootstrap_traces(ctx, PromptProgram(TASK.seed_instruction), TRAIN[:20])
        for demo in demos:
            index = next(i for i, ex in enumerate(TRAIN[:20]) if ex.id == demo.source_id)
            assert index % 2 == 0
        assert bootstrap_traces(ctx, PromptProgram("x"), []) == []

    def test_labeled_and_grouping(self) -> None:
        ctx = self._ctx()
        labeled = labeled_demos(ctx, TRAIN[:3])
        assert len(labeled) == 3 and labeled[0].output.endswith(f"Answer: {TRAIN[0].answer}")
        grouped = [Demo(f"q{i}", "a", group="g" + str(i % 2)) for i in range(6)]
        assert {k: len(v) for k, v in group_demos(grouped).items()} == {"g0": 3, "g1": 3}
        picked = sample_demo_set(random.Random(0), grouped, 2)
        assert len(picked) == 4
        assert sample_demo_set(random.Random(0), grouped, 0) == ()
        assert [d.input for d in first_demo_set(grouped, 1)] == ["q0", "q1"]

    def test_teacher_pool_balances_groups(self) -> None:
        train = [TaskExample(f"e{i}", "What is 1 + 1?", "2", group=f"g{i % 3}") for i in range(30)]
        pool = teacher_pool(self._ctx(train), 9)
        assert sorted(ex.group for ex in pool) == ["g0"] * 3 + ["g1"] * 3 + ["g2"] * 3
        assert len(teacher_pool(self._ctx(), 7)) == 7


class TestHelpers:
    def test_clip(self) -> None:
        assert clip("  short ") == "short"
        long = "x" * 50
        assert clip(long, 30).endswith("[truncated]") and len(clip(long, 30)) <= 30

    def test_format_rollouts(self) -> None:
        ctx = OptimizerContext(TASK, FakeLM(), TRAIN, VAL)
        result = ctx.evaluator.evaluate(PromptProgram("Solve."), TRAIN[:2])
        text = format_rollouts(result.rollouts, TASK, limit=1)
        assert "Example 1" in text and "Example 2" not in text and "Feedback:" in text

    def test_context_budget_helpers(self) -> None:
        ctx = OptimizerContext(TASK, FakeLM(), TRAIN, VAL, budget=Budget(100))
        assert ctx.val_cost == len(VAL)
        assert ctx.cap(500, reserve=30) == 70 and ctx.cap(5) == 5
        assert ctx.affordable(70, reserve=30) and not ctx.affordable(71, reserve=30)
        program = PromptProgram("Solve.")
        assert ctx.unseen_val_cost(program) == len(VAL)
        ctx.score_val(program)
        assert ctx.unseen_val_cost(program) == 0
        assert ctx.scored_programs()[0][1] == program
        assert ctx.val_rollouts(program).mean == ctx.known_val_score(program)

    def test_score_finalists_respects_budget(self) -> None:
        ctx = OptimizerContext(TASK, FakeLM(), TRAIN, VAL, budget=Budget(len(VAL) + 10))
        programs = [PromptProgram("A."), PromptProgram("A."), PromptProgram("B.")]
        ctx.score_finalists(programs)
        assert len(ctx.scored_programs()) == 1

    def test_ask_many_and_propose_many(self) -> None:
        ctx = OptimizerContext(TASK, FakeLM(), TRAIN, VAL)
        replies = ctx.ask_many("sys", ["one", "two"])
        assert len(replies) == 2
        parsed = ctx.propose_many("sys", ["one"])
        assert parsed[0] and parsed[0].startswith("Solve the arithmetic problem")

    def test_evaluate_many_batches_once(self) -> None:
        lm = FakeLM()
        ctx = OptimizerContext(TASK, lm, TRAIN, VAL)
        ctx.score_batch_many([PromptProgram("A."), PromptProgram("B.")], TRAIN[:5])
        assert len(lm.requests) == 10
        assert ctx.evaluator.pending(PromptProgram("A."), TRAIN[:6]) == 1
