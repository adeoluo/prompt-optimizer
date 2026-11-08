"""Task registry, data location and seeded train/validation/test splits.

Split discipline (the same for every optimizer and for DSPy):

* **test** is the official evaluation split where one exists (GSM8K test
  1,319; MATH-500), never seen during optimization.
* **train** and **validation** come from a disjoint pool (the official training
  split where there is one) with a fixed seed.
* Benchmarks without a training split (BIG-Bench Hard, HotpotQA distractor dev,
  IFEval) are partitioned once with a fixed seed; BIG-Bench Hard is partitioned
  per task so every task is represented in every split.

Split sizes are part of the experiment configuration and are reported with
every result.
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from prompt_optimizer.tasks.base import Task, TaskExample

DATA_ROOT = Path(
    os.environ.get(
        "PROMPT_OPTIMIZER_DATA",
        Path(__file__).resolve().parents[4] / ".bench-data" / "prompt-optimizer" / "data",
    )
)
SPLIT_SEED = 20260211


@dataclass(frozen=True)
class Splits:
    """Disjoint train, validation and test examples for one benchmark."""

    train: tuple[TaskExample, ...]
    val: tuple[TaskExample, ...]
    test: tuple[TaskExample, ...]

    def __post_init__(self) -> None:
        ids = [ex.id for ex in (*self.train, *self.val, *self.test)]
        if len(ids) != len(set(ids)):
            raise ValueError("splits overlap")

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def draw(
    pool: Sequence[TaskExample], sizes: Sequence[int], seed: int = SPLIT_SEED
) -> list[tuple[TaskExample, ...]]:
    """Shuffle ``pool`` with ``seed`` and cut consecutive disjoint slices."""

    if sum(sizes) > len(pool):
        raise ValueError(f"asked for {sum(sizes)} examples from a pool of {len(pool)}")
    order = list(pool)
    random.Random(seed).shuffle(order)
    out: list[tuple[TaskExample, ...]] = []
    start = 0
    for size in sizes:
        out.append(tuple(order[start : start + size]))
        start += size
    return out


def draw_grouped(
    pool: Sequence[TaskExample],
    per_group: Sequence[int | None],
    seed: int = SPLIT_SEED,
) -> list[tuple[TaskExample, ...]]:
    """Per-group version of ``draw``; a size of None takes the rest of the group."""

    groups: dict[str, list[TaskExample]] = defaultdict(list)
    for ex in pool:
        groups[ex.group].append(ex)
    out: list[list[TaskExample]] = [[] for _ in per_group]
    for name in sorted(groups):
        members = list(groups[name])
        random.Random(f"{seed}:{name}").shuffle(members)
        start = 0
        for index, size in enumerate(per_group):
            stop = len(members) if size is None else min(len(members), start + size)
            out[index].extend(members[start:stop])
            start = stop
    return [tuple(part) for part in out]


@dataclass(frozen=True)
class SplitConfig:
    """Split sizes for one benchmark; ``test`` None means the full test split."""

    train: int
    val: int
    test: int | None = None


DEFAULT_SPLITS: dict[str, SplitConfig] = {
    "gsm8k": SplitConfig(train=200, val=60, test=None),
    "bbh": SplitConfig(train=8, val=2, test=15),
    "hotpotqa": SplitConfig(train=150, val=100, test=500),
    "math500": SplitConfig(train=200, val=100, test=None),
    "ifeval": SplitConfig(train=100, val=100, test=None),
}


def load_task(
    name: str,
    root: Path = DATA_ROOT,
    config: SplitConfig | None = None,
    seed: int = SPLIT_SEED,
) -> tuple[Task, Splits]:
    """Instantiate a task and build its splits from the downloaded data."""

    cfg = config or DEFAULT_SPLITS[name]
    if name == "gsm8k":
        from prompt_optimizer.tasks import gsm8k

        train, val = draw(gsm8k.load(root, "train"), [cfg.train, cfg.val], seed)
        test = tuple(gsm8k.load(root, "test"))
        if cfg.test is not None:
            (test,) = draw(test, [cfg.test], seed)
        return gsm8k.GSM8K(), Splits(train, val, test)
    if name == "bbh":
        from prompt_optimizer.tasks import bbh

        train, val, test = draw_grouped(bbh.load(root), [cfg.train, cfg.val, cfg.test], seed)
        return bbh.BBH(), Splits(train, val, test)
    if name == "hotpotqa":
        from prompt_optimizer.tasks import hotpotqa

        train, val, test = draw(hotpotqa.load(root), [cfg.train, cfg.val, cfg.test or 0], seed)
        return hotpotqa.HotpotQA(), Splits(train, val, test)
    if name == "math500":
        from prompt_optimizer.tasks import math500

        train, val = draw(math500.load(root, "train"), [cfg.train, cfg.val], seed)
        test = tuple(math500.load(root, "test"))
        if cfg.test is not None:
            (test,) = draw(test, [cfg.test], seed)
        return math500.MATH500(), Splits(train, val, test)
    if name == "ifeval":
        from prompt_optimizer.tasks import ifeval

        pool = ifeval.load(root)
        rest = len(pool) - cfg.train - cfg.val
        train, val, test = draw(pool, [cfg.train, cfg.val, cfg.test or rest], seed)
        return ifeval.IFEval(), Splits(train, val, test)
    raise KeyError(f"unknown task {name!r}; known: {sorted(DEFAULT_SPLITS)}")
