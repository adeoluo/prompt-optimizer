"""Offline checks of the DSPy adapter (skipped unless the ``compare`` group is installed)."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

dspy = pytest.importorskip("dspy")
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import dspy_compare as dc  # noqa: E402

from prompt_optimizer.backends import CachedLM, CostMeter, LMRequest, LMResponse  # noqa: E402
from prompt_optimizer.tasks.gsm8k import GSM8K  # noqa: E402


class Scripted:
    """Answers every request in DSPy's chat-adapter format with a fixed answer."""

    model_id = "scripted"

    def __init__(self) -> None:
        self.requests: list[LMRequest] = []

    def generate(self, requests: Sequence[LMRequest]) -> list[LMResponse]:
        self.requests.extend(requests)
        text = (
            "[[ ## reasoning ## ]]\n6 times 7 is 42.\n\n[[ ## answer ## ]]\n42\n\n"
            "[[ ## completed ## ]]"
        )
        return [LMResponse(text, 10, 20) for _ in requests]


def test_routed_lm_answers_through_the_shared_client() -> None:
    meter = CostMeter()
    inner = Scripted()
    lm = dc.RoutedLM(CachedLM(inner, meter=meter), "task", 256)
    task = GSM8K()
    with dspy.context(lm=lm):
        program = dc.make_program(task)
        prediction = program(question="What is 6 times 7?")
    assert prediction.answer == "42"
    assert "6 times 7 is 42." in prediction.reasoning
    assert dc.as_output(prediction).endswith("Answer: 42")
    usage = meter.total(purpose="task")
    assert usage.calls == 1 and usage.prompt_tokens == 10 and usage.completion_tokens == 20
    request = inner.requests[0]
    assert request.purpose == "task" and request.max_tokens == 256
    assert task.seed_instruction in request.messages[0].content


def test_program_dict_and_examples() -> None:
    task = GSM8K()
    program = dc.make_program(task)
    data = dc.program_dict(program)
    assert data["instruction"] == task.seed_instruction and data["demos"] == []
    from prompt_optimizer.tasks.base import TaskExample

    examples = dc.to_examples([TaskExample("e1", "Q?", "3")])
    assert examples[0].ex_id == "e1" and examples[0].inputs().question == "Q?"
    assert dc._text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
    assert dc._text(5) == "5"
