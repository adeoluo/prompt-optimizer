"""Integration tests against a running llama-server (opt-in).

Run with ``pytest -m integration`` while ``llama-server`` serves a Qwen3.5 model
on ``PROMPT_OPTIMIZER_URL`` (default http://127.0.0.1:8091) and the benchmark
data is downloaded. They are excluded from the default run and from CI.
"""

from __future__ import annotations

import os

import httpx
import pytest

from prompt_optimizer.backends import CachedLM, CostMeter, LlamaCppLM
from prompt_optimizer.backends.base import system_user
from prompt_optimizer.optim import Budget, Evaluator, build
from prompt_optimizer.program import PromptProgram
from prompt_optimizer.tasks.registry import DATA_ROOT, load_task

URL = os.environ.get("PROMPT_OPTIMIZER_URL", "http://127.0.0.1:8091")

pytestmark = pytest.mark.integration


def _server_up() -> bool:
    try:
        return httpx.get(f"{URL}/health", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


needs_server = pytest.mark.skipif(not _server_up(), reason="no llama-server running")
needs_data = pytest.mark.skipif(not (DATA_ROOT / "gsm8k").exists(), reason="no benchmark data")


@needs_server
def test_batch_round_trip_and_usage() -> None:
    meter = CostMeter()
    lm = CachedLM(LlamaCppLM(URL, "integration", concurrency=2), meter=meter)
    replies = lm.generate(
        [system_user("Reply with one word.", f"Say {w}.") for w in ("red", "blue")]
    )
    assert len(replies) == 2 and all(r.completion_tokens > 0 for r in replies)
    assert meter.total().fresh_calls == 2
    again = lm.generate([system_user("Reply with one word.", "Say red.")])
    assert again[0].cached and meter.total().fresh_calls == 2


@needs_server
@needs_data
def test_gsm8k_seed_program_scores_a_few_examples() -> None:
    task, splits = load_task("gsm8k")
    lm = CachedLM(LlamaCppLM(URL, "integration", concurrency=4))
    result = Evaluator(task, lm).evaluate(PromptProgram(task.seed_instruction), splits.val[:4])
    assert len(result.rollouts) == 4
    assert all(r.output for r in result.rollouts)
    assert 0.0 <= result.mean <= 1.0


@needs_server
@needs_data
def test_one_optimizer_runs_end_to_end_on_a_tiny_budget() -> None:
    task, splits = load_task("gsm8k")
    lm = CachedLM(LlamaCppLM(URL, "integration", concurrency=4))
    optimizer = build("gepa", iterations=1)
    result = optimizer.compile(task, lm, splits.train[:6], splits.val[:4], budget=Budget(12))
    assert result.rollouts_used <= 12
    assert result.program.instruction
