"""The local models used in the experiments and where their weights live.

All three are Qwen3.5 instruction models (February 2026) quantized to
Q4_K_M GGUF by Unsloth. The 2B model runs every optimization loop; the final
prompts are then evaluated on 0.8B, 2B and 4B to measure transfer across scale.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODELS_DIR = Path(
    os.environ.get(
        "PROMPT_OPTIMIZER_MODELS",
        Path(__file__).resolve().parents[4] / ".bench-data" / "prompt-optimizer" / "models",
    )
)
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


@dataclass(frozen=True)
class ModelSpec:
    """A GGUF model file and its Hugging Face origin."""

    name: str
    repo: str
    filename: str
    parallel: int
    ctx_per_slot: int

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.filename}"

    def resolve(self, models_dir: Path = DEFAULT_MODELS_DIR) -> Path:
        """Find the weights in the models directory or the shared HF cache."""

        local = models_dir / self.filename
        if local.exists():
            return local.resolve()
        snapshots = HF_CACHE / f"models--{self.repo.replace('/', '--')}" / "snapshots"
        if snapshots.exists():
            for snapshot in sorted(snapshots.iterdir()):
                candidate = snapshot / self.filename
                if candidate.exists():
                    return candidate.resolve()
        raise FileNotFoundError(
            f"{self.filename} not found in {models_dir} or the HF cache; download {self.url}"
        )


MODELS: dict[str, ModelSpec] = {
    "qwen3.5-0.8b": ModelSpec(
        "qwen3.5-0.8b", "unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf", 32, 4096
    ),
    "qwen3.5-2b": ModelSpec(
        "qwen3.5-2b", "unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf", 12, 3072
    ),
    "qwen3.5-4b": ModelSpec(
        "qwen3.5-4b", "unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf", 16, 4096
    ),
}


def get_model(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS)}") from None
