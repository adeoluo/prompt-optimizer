"""The object every optimizer searches over: an instruction plus demonstrations.

A ``PromptProgram`` is rendered to chat messages as

    system:    <instruction>\\n\\n<task output contract>
    user:      <demo 1 input>          (one user/assistant pair per demonstration)
    assistant: <demo 1 output>
    ...
    user:      <query>

The output contract (for example "End with a line 'Answer: <number>'") belongs
to the task, not to the program, and no optimizer may edit it. Fixing it keeps
answer extraction identical across optimizers, so a score difference is a
difference in reasoning quality rather than in parse rate (ADR-0005).

Demonstrations may carry a ``group`` (for example a BIG-Bench Hard subtask). A
query of group ``g`` sees only ungrouped demonstrations and those of group ``g``,
so one program can hold one shared instruction and per-subtask demonstrations.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from prompt_optimizer.backends.base import LMRequest, Message, Purpose


@dataclass(frozen=True)
class Demo:
    """One in-context demonstration: a rendered input and the full model output."""

    input: str
    output: str
    source_id: str = ""
    group: str = ""

    def __post_init__(self) -> None:
        if not self.input.strip() or not self.output.strip():
            raise ValueError("demonstrations need a non-empty input and output")


@dataclass(frozen=True)
class PromptProgram:
    """An instruction and an ordered tuple of demonstrations."""

    instruction: str
    demos: tuple[Demo, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.instruction.strip():
            raise ValueError("instruction must be non-empty")

    def with_instruction(self, instruction: str) -> PromptProgram:
        return PromptProgram(instruction.strip(), self.demos)

    def with_demos(self, demos: Sequence[Demo]) -> PromptProgram:
        return PromptProgram(self.instruction, tuple(demos))

    def system_text(self, contract: str) -> str:
        return f"{self.instruction.strip()}\n\n{contract.strip()}" if contract else self.instruction

    def demos_for(self, group: str = "") -> tuple[Demo, ...]:
        """Demonstrations shown for a query of ``group`` (ungrouped demos always apply)."""

        return tuple(d for d in self.demos if not d.group or d.group == group)

    def messages(self, contract: str, query: str, group: str = "") -> tuple[Message, ...]:
        out = [Message("system", self.system_text(contract))]
        for demo in self.demos_for(group):
            out.append(Message("user", demo.input))
            out.append(Message("assistant", demo.output))
        out.append(Message("user", query))
        return tuple(out)

    def request(
        self,
        contract: str,
        query: str,
        *,
        group: str = "",
        max_tokens: int,
        temperature: float = 0.0,
        seed: int = 0,
        purpose: Purpose = "task",
    ) -> LMRequest:
        return LMRequest(
            messages=self.messages(contract, query, group),
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            purpose=purpose,
        )

    def fingerprint(self) -> str:
        """Short content hash used to memoize evaluations and name candidates."""

        payload = json.dumps(
            [self.instruction, [[d.input, d.output, d.group] for d in self.demos]],
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, object]:
        return {
            "instruction": self.instruction,
            "demos": [
                {"input": d.input, "output": d.output, "source_id": d.source_id, "group": d.group}
                for d in self.demos
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> PromptProgram:
        demos_raw = data.get("demos", [])
        assert isinstance(demos_raw, list)
        demos = tuple(
            Demo(
                str(d["input"]),
                str(d["output"]),
                str(d.get("source_id", "")),
                str(d.get("group", "")),
            )
            for d in demos_raw
        )
        return cls(str(data["instruction"]), demos)
