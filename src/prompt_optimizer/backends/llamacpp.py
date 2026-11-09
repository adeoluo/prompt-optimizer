"""llama.cpp backend: an async client for ``llama-server`` and a process manager.

``LlamaCppLM`` speaks the OpenAI-compatible ``/v1/chat/completions`` endpoint of
``llama-server``. A batch of requests is sent concurrently (bounded by
``concurrency``) over one ``httpx.AsyncClient``; the server's continuous
batching packs the in-flight sequences into shared forward passes, which is
where the throughput comes from on a single Apple M1 GPU.

``LlamaServer`` starts and stops one server process for one GGUF model. The
experiments keep exactly one model in memory at a time (the 8 GB machine is
shared), so switching models means stopping one server and starting the next.
Qwen3.5 thinking is disabled at the server with ``--reasoning off``.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

import httpx

from prompt_optimizer.backends.base import LMRequest, LMResponse

_THINK_CLOSE = "</think>"

#: HTTP statuses that will not change on retry (bad request, payload too large).
_PERMANENT = frozenset({400, 413, 422})

#: Called with (position in the batch, response) as soon as one request finishes.
ResultCallback = Callable[[int, LMResponse], None]


def _strip_thinking(text: str) -> str:
    """Drop an empty or leaked ``<think>...</think>`` block some templates emit."""

    if _THINK_CLOSE in text:
        text = text.split(_THINK_CLOSE, 1)[1]
    return text.strip()


def _payload(request: LMRequest) -> dict[str, Any]:
    body: dict[str, Any] = {
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "seed": request.seed,
        "cache_prompt": True,
    }
    if request.stop:
        body["stop"] = list(request.stop)
    return body


def parse_completion(data: dict[str, Any], latency_s: float) -> LMResponse:
    """Turn an OpenAI-style chat completion body into an ``LMResponse``."""

    choice = data["choices"][0]
    usage = data.get("usage") or {}
    return LMResponse(
        text=_strip_thinking(choice["message"].get("content") or ""),
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        finish_reason=str(choice.get("finish_reason") or "stop"),
        latency_s=latency_s,
    )


class LlamaCppLM:
    """Batched async client for a running ``llama-server``."""

    #: ``CachedLM`` passes a per-response callback so every answer is cached on arrival.
    streams_results = True

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8091",
        model_id: str = "llama.cpp",
        *,
        concurrency: int = 16,
        timeout_s: float = 900.0,
        retries: int = 8,
        retry_delay_s: float = 1.0,
        max_retry_delay_s: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self.base_url = base_url.rstrip("/")
        self._model_id = model_id
        self.concurrency = concurrency
        self.timeout_s = timeout_s
        self.retries = retries
        self.retry_delay_s = retry_delay_s
        self.max_retry_delay_s = max_retry_delay_s
        self.transport = transport

    @property
    def model_id(self) -> str:
        return self._model_id

    async def _one(
        self,
        client: httpx.AsyncClient,
        gate: asyncio.Semaphore,
        request: LMRequest,
        index: int = 0,
        on_result: ResultCallback | None = None,
    ) -> LMResponse:
        response = await self._attempt(client, gate, request)
        if on_result is not None:
            on_result(index, response)
        return response

    async def _attempt(
        self, client: httpx.AsyncClient, gate: asyncio.Semaphore, request: LMRequest
    ) -> LMResponse:
        async with gate:
            delay = self.retry_delay_s
            for attempt in range(self.retries + 1):
                start = time.perf_counter()
                try:
                    reply = await client.post(
                        f"{self.base_url}/v1/chat/completions", json=_payload(request)
                    )
                    if reply.status_code in _PERMANENT:
                        # A request the server can never answer (for example a prompt
                        # longer than the slot context) scores as an empty answer instead
                        # of aborting a long run; the reason is kept for accounting.
                        return LMResponse(
                            text="",
                            prompt_tokens=0,
                            completion_tokens=0,
                            finish_reason="error",
                            latency_s=time.perf_counter() - start,
                            extra={"status": str(reply.status_code), "error": reply.text[:200]},
                        )
                    reply.raise_for_status()
                    return parse_completion(reply.json(), time.perf_counter() - start)
                except (httpx.TransportError, httpx.HTTPStatusError):
                    if attempt == self.retries:
                        raise
                    await asyncio.sleep(delay)
                    delay = min(2.0 * delay, self.max_retry_delay_s)
        raise RuntimeError("unreachable")  # pragma: no cover

    async def agenerate(
        self, requests: Sequence[LMRequest], on_result: ResultCallback | None = None
    ) -> list[LMResponse]:
        gate = asyncio.Semaphore(self.concurrency)
        limits = httpx.Limits(max_connections=self.concurrency)
        async with httpx.AsyncClient(
            timeout=self.timeout_s, limits=limits, transport=self.transport
        ) as client:
            return list(
                await asyncio.gather(
                    *(self._one(client, gate, r, i, on_result) for i, r in enumerate(requests))
                )
            )

    def generate(
        self, requests: Sequence[LMRequest], on_result: ResultCallback | None = None
    ) -> list[LMResponse]:
        """Answer a batch concurrently; ``on_result(i, response)`` fires as each finishes."""

        if not requests:
            return []
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.agenerate(requests, on_result))
        # Called from inside a running event loop: run the batch on a helper thread.
        box: dict[str, list[LMResponse]] = {}

        def runner() -> None:
            box["out"] = asyncio.run(self.agenerate(requests, on_result))

        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()
        return box["out"]


def _alive(pid: int) -> bool:
    """True while ``pid`` runs (a finished child of this process is reaped here)."""

    try:
        finished, _ = os.waitpid(pid, os.WNOHANG)
        if finished == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def terminate_pid(pid: int, grace_s: float = 30.0, poll_s: float = 0.2) -> bool:
    """Stop a process: SIGTERM, wait up to ``grace_s`` for it to exit, then SIGKILL.

    ``llama-server`` closes its listening socket on SIGTERM but can keep running
    while clients hold open requests, so a closed port does not mean the memory is
    free. Returns True when the process had to be killed.
    """

    if not _alive(pid):
        return False
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not _alive(pid):
            return False
        time.sleep(poll_s)
    os.kill(pid, signal.SIGKILL)
    while _alive(pid):
        time.sleep(poll_s)
    return True


@dataclass
class LlamaServer:
    """Start, health-check and stop one ``llama-server`` process.

    The KV cache holds ``parallel * ctx_per_slot`` tokens. With ``unified_kv`` the
    slots share that pool (``-kvu``), so one long prompt can use more than its share
    while the total stays small.
    """

    model_path: Path
    port: int = 8091
    parallel: int = 16
    ctx_per_slot: int = 4096
    binary: str = "llama-server"
    log_path: Path | None = None
    extra_args: tuple[str, ...] = ()
    unified_kv: bool = False
    _process: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def command(self) -> list[str]:
        return [
            self.binary,
            "-m",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "-np",
            str(self.parallel),
            "-c",
            str(self.parallel * self.ctx_per_slot),
            "-ngl",
            "99",
            "--reasoning",
            "off",
            "--no-webui",
            *(("-kvu",) if self.unified_kv else ()),
            *self.extra_args,
        ]

    def healthy(self) -> bool:
        try:
            return httpx.get(f"{self.base_url}/health", timeout=2.0).status_code == 200
        except httpx.HTTPError:
            return False

    def start(self, wait_s: float = 300.0) -> None:
        if self._process is not None:
            return
        if shutil.which(self.binary) is None and not os.path.exists(self.binary):
            raise FileNotFoundError(f"{self.binary} not found on PATH")
        log = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL  # noqa: SIM115
        self._process = subprocess.Popen(self.command(), stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self._process.returncode}")
            if self.healthy():
                return
            time.sleep(1.0)
        self.stop()
        raise TimeoutError("llama-server did not become healthy in time")

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.send_signal(signal.SIGTERM)
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._process = None

    def __enter__(self) -> LlamaServer:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
