"""Prompt optimizers from the literature, implemented against one shared contract.

Every optimizer receives a seed program, training and validation examples and a
rollout budget, and returns the program with the best validation score it
measured (``optim.base``). ``build`` creates one by name with default settings.
"""

from __future__ import annotations

from typing import Any

from prompt_optimizer.optim.ape import APE
from prompt_optimizer.optim.base import (
    Budget,
    BudgetExhaustedError,
    Evaluator,
    Optimizer,
    OptimizerContext,
    OptimizerResult,
    Trial,
)
from prompt_optimizer.optim.bootstrap import BootstrapFewShotRS
from prompt_optimizer.optim.gepa import GEPA
from prompt_optimizer.optim.mipro import MIPROv2
from prompt_optimizer.optim.opro import OPRO
from prompt_optimizer.optim.promptbreeder import PromptBreeder
from prompt_optimizer.optim.textgrad import TextGrad

OPTIMIZERS: dict[str, type[Optimizer]] = {
    "bootstrap_fewshot": BootstrapFewShotRS,
    "ape": APE,
    "opro": OPRO,
    "miprov2": MIPROv2,
    "textgrad": TextGrad,
    "gepa": GEPA,
    "promptbreeder": PromptBreeder,
}


def build(name: str, **overrides: Any) -> Optimizer:
    """Instantiate optimizer ``name`` with its defaults, overriding any field."""

    try:
        cls = OPTIMIZERS[name]
    except KeyError:
        raise KeyError(f"unknown optimizer {name!r}; known: {sorted(OPTIMIZERS)}") from None
    return cls(**overrides)


__all__ = [
    "APE",
    "GEPA",
    "OPRO",
    "OPTIMIZERS",
    "BootstrapFewShotRS",
    "Budget",
    "BudgetExhaustedError",
    "Evaluator",
    "MIPROv2",
    "Optimizer",
    "OptimizerContext",
    "OptimizerResult",
    "PromptBreeder",
    "TextGrad",
    "Trial",
    "build",
]
