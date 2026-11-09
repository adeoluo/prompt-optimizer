"""Run optimization and test jobs against local llama.cpp models, resumably.

Examples (each job is skipped when its result file is complete, and every
model call is replayed from the cache on a rerun):

    # optimize every method on GSM8K with Qwen3.5-2B, then score on test
    python benchmarks/run_experiments.py --task gsm8k --optimizers seed bootstrap_fewshot \\
        ape opro miprov2 textgrad gepa promptbreeder --budget 500 --test-limit 400

    # transfer: score the 2B-optimized programs with Qwen3.5-4B
    python benchmarks/run_experiments.py --task gsm8k --optimizers seed gepa \\
        --eval-model qwen3.5-4b --test-limit 300 --transfer-only

The script starts the right ``llama-server`` for each model (one model in
memory at a time) and leaves a server running only if it found one running.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

from prompt_optimizer.backends import CachedLM, CostMeter, LlamaCppLM, ResponseCache
from prompt_optimizer.backends.llamacpp import LlamaServer, terminate_pid
from prompt_optimizer.backends.models import get_model
from prompt_optimizer.experiments import RESULTS_ROOT, RunSpec, evaluate_program, optimize
from prompt_optimizer.program import PromptProgram
from prompt_optimizer.tasks.registry import DATA_ROOT, load_task

BENCH = DATA_ROOT.parent
CACHE = BENCH / "cache" / "lm.sqlite"
PID_FILE = BENCH / "logs" / "server.pid"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def served_model(url: str) -> str | None:
    try:
        data = httpx.get(f"{url}/v1/models", timeout=5.0).json()
    except (httpx.HTTPError, ValueError):
        return None
    entries = data.get("models") or data.get("data") or []
    return str(entries[0].get("model") or entries[0].get("id")) if entries else None


def ensure_server(model: str, port: int) -> None:
    """Make sure ``llama-server`` on ``port`` serves ``model``; swap it if not."""

    spec = get_model(model)
    url = f"http://127.0.0.1:{port}"
    current = served_model(url)
    if current is not None and Path(current).resolve() == spec.resolve().resolve():
        return
    if current is not None and PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        log(f"stopping server pid {pid} ({Path(current).name})")
        if terminate_pid(pid):
            log(f"server pid {pid} ignored SIGTERM; killed")
    elif current is not None:
        raise RuntimeError(f"port {port} serves {current} and no pid file says who owns it")
    server = LlamaServer(
        spec.resolve(),
        port=port,
        parallel=spec.parallel,
        ctx_per_slot=spec.ctx_per_slot,
        log_path=BENCH / "logs" / f"server_{model}.log",
        unified_kv=True,
        # --cache-ram 0: the default 8 GiB host-RAM prompt cache made the shared 8 GB
        # machine swap heavily; the response cache already makes repeats free.
        extra_args=("-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--cache-ram", "0", "-ub", "128"),
    )
    log(f"starting {model}: {' '.join(server.command())}")
    server.start()
    assert server._process is not None
    PID_FILE.write_text(str(server._process.pid))
    server._process = None  # leave it running after this script exits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--optimizers", nargs="+", required=True)
    parser.add_argument("--model", default="qwen3.5-2b", help="optimization model")
    parser.add_argument("--eval-model", default=None, help="test model (default: --model)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--transfer-only", action="store_true", help="skip optimization")
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], help="optimizer overrides key=json")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--root", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args(argv)

    overrides = tuple((k, json.loads(v)) for k, v in (item.split("=", 1) for item in args.set))
    eval_model = args.eval_model or args.model
    task, splits = load_task(args.task)
    log(f"{args.task}: splits {splits.sizes()}")
    cache = ResponseCache(CACHE)
    url = f"http://127.0.0.1:{args.port}"

    def client(model: str, meter: CostMeter) -> CachedLM:
        ensure_server(model, args.port)
        inner = LlamaCppLM(url, model, concurrency=args.concurrency)
        return CachedLM(inner, cache, meter, model_id=f"{model}-q4km")

    for optimizer in args.optimizers:
        for seed in args.seeds if optimizer != "seed" else [0]:
            spec = RunSpec(args.task, optimizer, args.model, seed, args.budget, overrides)
            path = spec.run_path(args.root)
            if args.transfer_only:
                if not path.exists():
                    log(f"skip {spec.run_id}: no run file")
                    continue
                run = json.loads(path.read_text())
            else:
                meter = CostMeter()
                start = time.perf_counter()
                log(f"optimize {args.task} {spec.run_id} on {args.model}")
                run = optimize(spec, task, splits, client(args.model, meter), meter, root=args.root)
                log(
                    f"done {spec.run_id}: val={run['val_score']:.3f} "
                    f"rollouts={run['rollouts_used']} ({time.perf_counter() - start:.0f}s)"
                )
            if args.no_test:
                continue
            program = PromptProgram.from_dict(run["program"])
            meter = CostMeter()
            result = evaluate_program(
                spec,
                task,
                program,
                splits.test,
                client(eval_model, meter),
                meter,
                eval_model,
                limit=args.test_limit,
                root=args.root,
                log=log,
            )
            log(f"test {spec.run_id} on {eval_model}: acc={result['accuracy']:.3f} n={result['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
