"""A production-style stress benchmark for the prediction service.

Compiles the prompt once, then drives the full predict handler under sustained,
concurrent load and reports throughput with p50, p95, and p99 latency.
Deterministic and offline; run with:

    uv run python benchmarks/stress.py
"""

from __future__ import annotations

import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from prompt_optimizer.service import PromptService, handle

_MESSAGES = [
    "I need a refund for my money",
    "where is my package shipping",
    "please cancel my subscription",
    "refund the overcharge on my order",
    "my parcel is late what is the delivery status",
    "stop my recurring plan",
]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    service = PromptService.compiled()
    rng = random.Random(0)
    requests = [{"text": rng.choice(_MESSAGES)} for _ in range(3000)]

    def one(body: dict[str, str]) -> float:
        start = time.perf_counter()
        handle("POST", "/predict", body, service)
        return (time.perf_counter() - start) * 1000

    print(f"Requests: {len(requests)}")
    for workers in (1, 4, 8):
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            latencies = list(pool.map(one, requests))
        wall = time.perf_counter() - start
        print(
            f"workers={workers:<2}  "
            f"throughput={len(requests) / wall:>8.0f} req/s  "
            f"p50={statistics.median(latencies):>6.3f} ms  "
            f"p95={_percentile(latencies, 0.95):>6.3f} ms  "
            f"p99={_percentile(latencies, 0.99):>6.3f} ms"
        )


if __name__ == "__main__":
    main()
