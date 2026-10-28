"""A predictor binds a prompt to the simulated language model."""

from __future__ import annotations

from prompt_optimizer.lm import SimulatedLanguageModel
from prompt_optimizer.types import Prompt


class Predictor:
    """Answer inputs using a fixed prompt and a language model."""

    def __init__(self, prompt: Prompt, model: SimulatedLanguageModel | None = None) -> None:
        self.prompt = prompt
        self.model = model or SimulatedLanguageModel()

    def predict(self, text: str) -> str:
        return self.model.generate(self.prompt, text)

    def with_prompt(self, prompt: Prompt) -> Predictor:
        return Predictor(prompt, self.model)
