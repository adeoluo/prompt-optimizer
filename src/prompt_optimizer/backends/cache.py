"""A content-addressed response cache and the caching model wrapper.

Every LM call is keyed by the SHA-256 of its canonical request (model id,
messages, decoding parameters; see ``LMRequest.cache_key``). The cache lives in
one SQLite file in WAL mode, so concurrent readers never block and a crashed
run loses at most the in-flight batch.

Two properties follow and the rest of the system relies on them:

* **Free reruns.** Re-running an experiment replays every previously answered
  call from disk, so evaluation tables regenerate without a model.
* **Resumable optimization.** Optimizers are deterministic given their seed and
  the model's answers. After a crash, rerunning the same optimizer replays the
  finished prefix of its search from the cache in seconds and continues with
  the first unanswered call (ADR-0006).

Cached responses keep their original token usage so cost accounting is the
same whether a call was fresh or replayed.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

from prompt_optimizer.backends.accounting import CostMeter
from prompt_optimizer.backends.base import LanguageModel, LMRequest, LMResponse

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    text TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    finish_reason TEXT NOT NULL,
    latency_s REAL NOT NULL,
    request TEXT NOT NULL
)
"""


class ResponseCache:
    """SQLite-backed map from request hash to response."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(_SCHEMA)

    def get_many(self, keys: Sequence[str]) -> dict[str, LMResponse]:
        found: dict[str, LMResponse] = {}
        with self._lock:
            for start in range(0, len(keys), 500):
                chunk = list(keys[start : start + 500])
                marks = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    "SELECT key, text, prompt_tokens, completion_tokens, finish_reason, latency_s "
                    f"FROM responses WHERE key IN ({marks})",
                    chunk,
                ).fetchall()
                for key, text, pt, ct, finish, latency in rows:
                    found[key] = LMResponse(
                        text=text,
                        prompt_tokens=pt,
                        completion_tokens=ct,
                        finish_reason=finish,
                        cached=True,
                        latency_s=latency,
                    )
        return found

    def put_many(self, items: Sequence[tuple[str, str, LMRequest, LMResponse]]) -> None:
        rows = [
            (
                key,
                model,
                resp.text,
                resp.prompt_tokens,
                resp.completion_tokens,
                resp.finish_reason,
                resp.latency_s,
                json.dumps([[m.role, m.content] for m in req.messages], ensure_ascii=False),
            )
            for key, model, req, resp in items
        ]
        with self._lock:
            self._conn.execute("BEGIN")
            self._conn.executemany(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
            self._conn.execute("COMMIT")

    def __len__(self) -> int:
        with self._lock:
            (count,) = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()
        return int(count)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class CachedLM:
    """Wrap a model with the response cache and a cost meter.

    Only cache misses reach the inner model; they are sent as one batch so the
    inner backend can run them concurrently. Duplicate requests inside one
    batch are answered once.
    """

    def __init__(
        self,
        inner: LanguageModel,
        cache: ResponseCache | None = None,
        meter: CostMeter | None = None,
        *,
        model_id: str | None = None,
    ) -> None:
        self.inner = inner
        self.cache = cache if cache is not None else ResponseCache()
        self.meter = meter if meter is not None else CostMeter()
        self._model_id = model_id or inner.model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(self, requests: Sequence[LMRequest]) -> list[LMResponse]:
        keys = [req.cache_key(self._model_id) for req in requests]
        hits = self.cache.get_many(list(dict.fromkeys(keys)))
        pending: dict[str, LMRequest] = {}
        for key, req in zip(keys, requests, strict=True):
            if key not in hits and key not in pending:
                pending[key] = req
        fresh: dict[str, LMResponse] = {}
        if pending:
            answers = self.inner.generate(list(pending.values()))
            fresh = dict(zip(pending.keys(), answers, strict=True))
            self.cache.put_many(
                [(key, self._model_id, pending[key], fresh[key]) for key in pending]
            )
        out: list[LMResponse] = []
        counted: set[str] = set()
        for key, req in zip(keys, requests, strict=True):
            if key in fresh and key not in counted:
                resp = fresh[key]
                counted.add(key)
            else:
                base = fresh.get(key) or hits[key]
                resp = LMResponse(
                    text=base.text,
                    prompt_tokens=base.prompt_tokens,
                    completion_tokens=base.completion_tokens,
                    finish_reason=base.finish_reason,
                    cached=True,
                    latency_s=base.latency_s,
                )
            self.meter.record(req.purpose, resp)
            out.append(resp)
        return out
