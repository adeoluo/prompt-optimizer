"""Download the public benchmark data into the shared bench-data directory.

    uv run --with pyarrow python scripts/download_data.py [--root DIR] [--only gsm8k bbh ...]

Every file comes from the dataset authors' official location. HotpotQA's
original host no longer resolves, so the distractor dev set is read from the
authors' Hugging Face dataset (hotpotqa/hotpot_qa, 7,405 questions) and
converted once to JSON lines with pyarrow. Nothing is committed to the
repository; a manifest with SHA-256 digests and sizes is written next to the
data so a rerun can confirm it read the same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prompt_optimizer.tasks.bbh import BBH_TASKS
from prompt_optimizer.tasks.registry import DATA_ROOT

GSM8K = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data"
BBH = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main"
HOTPOT = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "distractor/validation-00000-of-00001.parquet"
)
MATH = "https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/math_splits"
IFEVAL = (
    "https://raw.githubusercontent.com/google-research/google-research/master/"
    "instruction_following_eval/data/input_data.jsonl"
)


def sources() -> dict[str, list[tuple[str, str]]]:
    return {
        "gsm8k": [(f"{GSM8K}/{s}.jsonl", f"{s}.jsonl") for s in ("train", "test")],
        "bbh": [(f"{BBH}/bbh/{t}.json", f"{t}.json") for t in BBH_TASKS]
        + [(f"{BBH}/cot-prompts/{t}.txt", f"cot-prompts/{t}.txt") for t in BBH_TASKS],
        "hotpotqa": [(HOTPOT, "distractor_validation.parquet")],
        "math": [(f"{MATH}/{s}.jsonl", f"{s}.jsonl") for s in ("train", "test")],
        "ifeval": [(IFEVAL, "input_data.jsonl")],
    }


def fetch(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, open(partial, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    partial.rename(target)


def parquet_to_jsonl(source: Path, target: Path) -> None:
    """Convert the HotpotQA parquet file to JSON lines (needs pyarrow)."""

    if target.exists():
        return
    import pyarrow.parquet as pq

    rows = pq.read_table(source).to_pylist()
    with open(target, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    manifest: dict[str, dict[str, object]] = {}
    manifest_path = args.root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    for name, files in sources().items():
        if args.only and name not in args.only:
            continue
        for url, relative in files:
            target = args.root / name / relative
            fetch(url, target)
            manifest[f"{name}/{relative}"] = {
                "url": url,
                "bytes": target.stat().st_size,
                "sha256": digest(target),
            }
        if name == "hotpotqa":
            folder = args.root / name
            parquet_to_jsonl(
                folder / "distractor_validation.parquet", folder / "distractor_validation.jsonl"
            )
        print(f"{name}: {len(files)} files")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
