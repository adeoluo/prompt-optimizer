"""Language model request and response types and the backend protocol.

Every optimizer and evaluator talks to a model through one narrow interface:
``LanguageModel.generate`` takes a batch of ``LMRequest`` objects and returns one
``LMResponse`` per request, in order. Batching is part of the contract because
prompt optimization is dominated by many independent rollouts (one per
validation example per candidate prompt), and a local inference server serves
them far faster concurrently than one at a time.

A request is fully described by its chat messages and decoding parameters, so
its content hash is a stable cache key (see ``backends.cache``). The
``purpose`` label is accounting metadata only: it does not change the text the
model sees and is excluded from the cache key.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]

#: Accounting labels. ``task`` rollouts are the program being optimized; the
#: others are optimizer-side calls (instruction proposals, critiques, reflections).
Purpose = Literal["task", "propose", "critique", "reflect", "mutate", "summarize"]


@dataclass(frozen=True)
class Message:
    """One chat message."""

    role: Role
    content: str


@dataclass(frozen=True)
class LMRequest:
    """A single chat completion request.

    ``temperature`` 0 requests greedy decoding. ``seed`` makes sampled decoding
    reproducible on backends that honour it (llama.cpp does, per request).
    """

    messages: tuple[Message, ...]
    max_tokens: int = 512
    temperature: float = 0.0
    seed: int = 0
    stop: tuple[str, ...] = ()
    purpose: Purpose = "task"

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("a request needs at least one message")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")

    def cache_key(self, model_id: str) -> str:
        """SHA-256 over the canonical JSON of everything that affects the output."""

        payload = {
            "model": model_id,
            "messages": [[m.role, m.content] for m in self.messages],
            "max_tokens": self.max_tokens,
            "temperature": round(self.temperature, 6),
            "seed": self.seed,
            "stop": list(self.stop),
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LMResponse:
    """The model output plus the usage needed for cost accounting."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"
    cached: bool = False
    latency_s: float = 0.0
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


@runtime_checkable
class LanguageModel(Protocol):
    """Anything that can answer a batch of chat requests."""

    @property
    def model_id(self) -> str: ...

    def generate(self, requests: Sequence[LMRequest]) -> list[LMResponse]: ...


def system_user(system: str, user: str, **kwargs: object) -> LMRequest:
    """Build the common two-message request."""

    messages = (Message("system", system), Message("user", user))
    return LMRequest(messages=messages, **kwargs)  # type: ignore[arg-type]
