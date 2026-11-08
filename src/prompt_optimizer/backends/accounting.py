"""Cost accounting: LM calls and tokens, by purpose and by phase.

Prompt optimizers differ far more in what they spend than in what they
report, so every call is recorded. A call is counted once per request whether
it was answered by the model or replayed from the cache; ``fresh_calls``
separately counts the calls that actually reached the model in this process.
Counting replayed calls at their original usage keeps the cost columns of a
results table identical between the first run and any rerun.

Phases (``optimize``, ``evaluate``) are entered with ``meter.phase(name)``; the
purpose comes from the request (``task`` rollouts versus optimizer-side calls
such as proposals and critiques).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass

from prompt_optimizer.backends.base import LMResponse


@dataclass
class Usage:
    """Accumulated usage for one (phase, purpose) cell."""

    calls: int = 0
    fresh_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    truncated: int = 0
    model_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: Usage) -> None:
        self.calls += other.calls
        self.fresh_calls += other.fresh_calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.truncated += other.truncated
        self.model_seconds += other.model_seconds

    def as_dict(self) -> dict[str, float | int]:
        data: dict[str, float | int] = dict(asdict(self))
        data["total_tokens"] = self.total_tokens
        return data


class CostMeter:
    """Thread-safe usage ledger keyed by (phase, purpose)."""

    def __init__(self) -> None:
        self._cells: dict[tuple[str, str], Usage] = {}
        self._phase = "default"
        self._lock = threading.Lock()

    @property
    def current_phase(self) -> str:
        return self._phase

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        previous = self._phase
        self._phase = name
        try:
            yield
        finally:
            self._phase = previous

    def record(self, purpose: str, response: LMResponse) -> None:
        with self._lock:
            cell = self._cells.setdefault((self._phase, purpose), Usage())
            cell.calls += 1
            cell.fresh_calls += 0 if response.cached else 1
            cell.prompt_tokens += response.prompt_tokens
            cell.completion_tokens += response.completion_tokens
            cell.truncated += 1 if response.truncated else 0
            cell.model_seconds += response.latency_s

    def total(self, phase: str | None = None, purpose: str | None = None) -> Usage:
        out = Usage()
        with self._lock:
            for (cell_phase, cell_purpose), usage in self._cells.items():
                if phase is not None and cell_phase != phase:
                    continue
                if purpose is not None and cell_purpose != purpose:
                    continue
                out.add(usage)
        return out

    def breakdown(self) -> dict[str, dict[str, dict[str, float | int]]]:
        """Nested ``{phase: {purpose: usage}}`` for JSON results files."""

        out: dict[str, dict[str, dict[str, float | int]]] = {}
        with self._lock:
            for (phase, purpose), usage in sorted(self._cells.items()):
                out.setdefault(phase, {})[purpose] = usage.as_dict()
        return out

    def reset(self) -> None:
        with self._lock:
            self._cells.clear()
