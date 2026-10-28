"""Prompt Optimizer - automated prompt and few-shot optimization from scratch."""

from __future__ import annotations

from prompt_optimizer.evaluation import Row, format_table, run_ablation
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
    "Row",
    "SimulatedLanguageModel",
    "evaluate",
    "exact_match",
    "format_table",
    "parse_label_vocabulary",
    "run_ablation",
    "token_f1",
]
