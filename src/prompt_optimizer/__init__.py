"""Prompt Optimizer - automated prompt and few-shot optimization from scratch."""

from __future__ import annotations

from prompt_optimizer.lm import SimulatedLanguageModel, parse_label_vocabulary
from prompt_optimizer.metrics import evaluate, exact_match, token_f1
from prompt_optimizer.optimizers import (
    BootstrapFewShot,
    CompoundOptimizer,
    InstructionSearch,
    OptimizationResult,
)
from prompt_optimizer.predictor import Predictor
from prompt_optimizer.types import Example, Metric, Prompt

__version__ = "0.1.0"

__all__ = [
    "BootstrapFewShot",
    "CompoundOptimizer",
    "Example",
    "InstructionSearch",
    "Metric",
    "OptimizationResult",
    "Predictor",
    "Prompt",
    "SimulatedLanguageModel",
    "evaluate",
    "exact_match",
    "parse_label_vocabulary",
    "token_f1",
]
