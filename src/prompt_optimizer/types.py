"""Core data types shared across the package."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Example:
    """A single labeled example: a text input and its expected label."""

    text: str
    label: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("example text must be non-empty")
        if not self.label.strip():
            raise ValueError("example label must be non-empty")


@dataclass(frozen=True)
class Prompt:
    """A prompt is an instruction plus an ordered list of demonstrations.

    This is the object the optimizers search over: they change the instruction
    and the set and order of demonstrations to maximize a metric.
    """

    instruction: str
    demonstrations: tuple[Example, ...] = field(default_factory=tuple)

    def with_instruction(self, instruction: str) -> Prompt:
        return Prompt(instruction=instruction, demonstrations=self.demonstrations)

    def with_demonstrations(self, demonstrations: tuple[Example, ...]) -> Prompt:
        return Prompt(instruction=self.instruction, demonstrations=demonstrations)

    def render(self) -> str:
        """Render the prompt to the text a language model would receive."""

        lines = [self.instruction.strip(), ""]
        for demo in self.demonstrations:
            lines.append(f"Input: {demo.text}")
            lines.append(f"Label: {demo.label}")
        lines.append("Input: {query}")
        lines.append("Label:")
        return "\n".join(lines)


# A metric scores a predicted label against the expected label in [0, 1].
Metric = Callable[[str, str], float]
