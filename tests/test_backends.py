"""Tests for request types, the response cache, cost accounting and the llama.cpp client."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from prompt_optimizer.backends import (
    CachedLM,
    CostMeter,
    FakeLM,
    LlamaCppLM,
    LlamaServer,
    LMRequest,
    LMResponse,
    Message,
    ResponseCache,
    get_model,
    system_user,
)
from prompt_optimizer.backends.llamacpp import parse_completion
from prompt_optimizer.backends.models import ModelSpec


def _req(text: str = "hi", **kwargs: object) -> LMRequest:
    return system_user("sys", text, **kwargs)


class TestRequest:
    def test_rejects_empty_messages(self) -> None:
        with pytest.raises(ValueError):
            LMRequest(messages=())

    def test_rejects_bad_params(self) -> None:
        with pytest.raises(ValueError):
            _req(max_tokens=0)
        with pytest.raises(ValueError):
            _req(temperature=-0.1)

    def test_cache_key_ignores_purpose(self) -> None:
        a = _req(purpose="task")
        b = _req(purpose="propose")
        assert a.cache_key("m") == b.cache_key("m")

    def test_cache_key_depends_on_model_and_decoding(self) -> None:
        base = _req()
        assert base.cache_key("m1") != base.cache_key("m2")
        assert base.cache_key("m") != _req(seed=1).cache_key("m")
        assert base.cache_key("m") != _req(temperature=0.5).cache_key("m")
        assert base.cache_key("m") != _req(max_tokens=10).cache_key("m")
        assert base.cache_key("m") != _req(stop=("x",)).cache_key("m")

    @given(st.text(min_size=1), st.text(min_size=1))
    def test_cache_key_is_a_function_of_content(self, a: str, b: str) -> None:
        r1, r2 = _req(a), _req(b)
        assert (r1.cache_key("m") == r2.cache_key("m")) == (a == b)

    def test_response_properties(self) -> None:
        resp = LMResponse("x", 3, 4, finish_reason="length")
        assert resp.total_tokens == 7
        assert resp.truncated


class _Counting:
    """Inner model that counts how many requests reach it."""

    def __init__(self) -> None:
        self.seen = 0

    @property
    def model_id(self) -> str:
        return "counting"

    def generate(self, requests: list[LMRequest]) -> list[LMResponse]:
        self.seen += len(requests)
        return [LMResponse(r.messages[-1].content.upper(), 5, 2) for r in requests]


class TestCache:
    def test_hits_skip_the_model(self, tmp_path: Path) -> None:
        inner = _Counting()
        lm = CachedLM(inner, ResponseCache(tmp_path / "c.sqlite"))
        first = lm.generate([_req("a"), _req("b")])
        second = lm.generate([_req("a"), _req("b")])
        assert inner.seen == 2
        assert [r.text for r in first] == [r.text for r in second] == ["A", "B"]
        assert not first[0].cached and second[0].cached

    def test_duplicates_in_one_batch_are_sent_once(self) -> None:
        inner = _Counting()
        lm = CachedLM(inner)
        out = lm.generate([_req("a"), _req("a"), _req("a")])
        assert inner.seen == 1
        assert [r.cached for r in out] == [False, True, True]

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "c.sqlite"
        CachedLM(_Counting(), ResponseCache(path)).generate([_req("z")])
        inner = _Counting()
        cache = ResponseCache(path)
        CachedLM(inner, cache).generate([_req("z")])
        assert inner.seen == 0
        assert len(cache) == 1
        cache.close()

    def test_large_batches_chunk(self) -> None:
        lm = CachedLM(_Counting())
        reqs = [_req(str(i)) for i in range(1200)]
        lm.generate(reqs)
        again = lm.generate(reqs)
        assert all(r.cached for r in again)

    def test_meter_counts_replays_at_original_usage(self) -> None:
        meter = CostMeter()
        lm = CachedLM(_Counting(), meter=meter)
        with meter.phase("optimize"):
            lm.generate([_req("a", purpose="task")])
            lm.generate([_req("a", purpose="task")])
        usage = meter.total(phase="optimize", purpose="task")
        assert usage.calls == 2
        assert usage.fresh_calls == 1
        assert usage.prompt_tokens == 10
        assert usage.total_tokens == 14


class TestMeter:
    def test_phases_and_purposes(self) -> None:
        meter = CostMeter()
        assert meter.current_phase == "default"
        with meter.phase("evaluate"):
            meter.record("task", LMResponse("x", 1, 1, finish_reason="length"))
            meter.record("propose", LMResponse("x", 2, 2, cached=True))
        assert meter.current_phase == "default"
        assert meter.total(phase="evaluate").calls == 2
        assert meter.total(purpose="propose").fresh_calls == 0
        assert meter.total().truncated == 1
        tree = meter.breakdown()
        assert tree["evaluate"]["task"]["total_tokens"] == 2
        meter.reset()
        assert meter.total().calls == 0


def _completion(text: str, finish: str = "stop") -> dict[str, object]:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
    }


class TestLlamaCpp:
    def test_parse_strips_thinking(self) -> None:
        resp = parse_completion(_completion("<think>\n\n</think>\n\nAnswer: 4"), 0.5)
        assert resp.text == "Answer: 4"
        assert resp.prompt_tokens == 11 and resp.latency_s == 0.5

    def test_parse_handles_missing_usage(self) -> None:
        resp = parse_completion({"choices": [{"message": {"content": None}}]}, 0.0)
        assert resp.text == "" and resp.prompt_tokens == 0

    def test_batch_over_mock_transport(self) -> None:
        bodies: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            bodies.append(body)
            return httpx.Response(200, json=_completion(body["messages"][-1]["content"]))

        lm = LlamaCppLM(transport=httpx.MockTransport(handler), concurrency=2)
        out = lm.generate([_req("one", seed=7), _req("two", stop=("\n",))])
        assert [r.text for r in out] == ["one", "two"]
        assert bodies[0]["seed"] == 7 and bodies[0]["cache_prompt"] is True
        assert any(b.get("stop") == ["\n"] for b in bodies)
        assert lm.generate([]) == []

    def test_retries_then_succeeds(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(503)
            return httpx.Response(200, json=_completion("ok"))

        lm = LlamaCppLM(transport=httpx.MockTransport(handler), retry_delay_s=0.0)
        assert lm.generate([_req()])[0].text == "ok"

    def test_gives_up_after_retries(self) -> None:
        lm = LlamaCppLM(
            transport=httpx.MockTransport(lambda r: httpx.Response(500)),
            retries=1,
            retry_delay_s=0.0,
        )
        with pytest.raises(httpx.HTTPStatusError):
            lm.generate([_req()])

    def test_generate_inside_running_loop(self) -> None:
        lm = LlamaCppLM(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_completion("x")))
        )

        async def inner() -> list[LMResponse]:
            return lm.generate([_req()])

        assert asyncio.run(inner())[0].text == "x"

    def test_rejects_zero_concurrency(self) -> None:
        with pytest.raises(ValueError):
            LlamaCppLM(concurrency=0)

    def test_server_command_and_missing_binary(self, tmp_path: Path) -> None:
        server = LlamaServer(tmp_path / "m.gguf", port=9999, parallel=4, ctx_per_slot=1000)
        cmd = server.command()
        assert cmd[cmd.index("-c") + 1] == "4000"
        assert "--reasoning" in cmd and cmd[cmd.index("--reasoning") + 1] == "off"
        assert server.base_url.endswith(":9999")
        assert not server.healthy()
        server.stop()  # no process: a no-op
        missing = LlamaServer(tmp_path / "m.gguf", binary="definitely-not-a-binary-xyz")
        with pytest.raises(FileNotFoundError):
            missing.start()

    def test_server_exits_early(self, tmp_path: Path) -> None:
        server = LlamaServer(tmp_path / "m.gguf", binary="false", port=1, log_path=tmp_path / "log")
        with pytest.raises(RuntimeError):
            server.start(wait_s=5)


class TestModels:
    def test_registry(self) -> None:
        spec = get_model("qwen3.5-2b")
        assert spec.url.endswith("Qwen3.5-2B-Q4_K_M.gguf")
        with pytest.raises(KeyError):
            get_model("nope")

    def test_resolve_local_and_missing(self, tmp_path: Path) -> None:
        spec = ModelSpec("x", "org/x-GGUF", "x.gguf", 1, 1)
        with pytest.raises(FileNotFoundError):
            spec.resolve(tmp_path)
        (tmp_path / "x.gguf").write_bytes(b"0")
        assert spec.resolve(tmp_path).name == "x.gguf"


class TestFake:
    def test_skill_rewards_good_phrases_and_correct_demos(self) -> None:
        lm = FakeLM()
        plain = LMRequest((Message("system", "Answer."), Message("user", "What is 2 + 2?")))
        good = LMRequest(
            (
                Message("system", "Work step by step and verify."),
                Message("user", "What is 1 + 1?"),
                Message("assistant", "Answer: 2"),
                Message("user", "What is 2 + 2?"),
            )
        )
        wrong = LMRequest(
            (
                Message("system", "Answer."),
                Message("user", "What is 1 + 1?"),
                Message("assistant", "Answer: 3"),
                Message("user", "What is 2 + 2?"),
            )
        )
        harmful = LMRequest((Message("system", "Just guess."), Message("user", "What is 2 + 2?")))
        assert lm.skill(good) > lm.skill(plain) > lm.skill(wrong)
        assert lm.skill(harmful) < lm.skill(plain)

    def test_answers_and_proposals_are_deterministic(self) -> None:
        lm = FakeLM()
        reqs = [
            _req("What is 3 * 4?"),
            _req("unparsable"),
            _req("improve", purpose="propose", temperature=1.0, seed=3),
            _req("critique this", purpose="critique"),
            _req("summarize", purpose="summarize"),
        ]
        a = [r.text for r in lm.generate(reqs)]
        b = [r.text for r in FakeLM().generate(reqs)]
        assert a == b
        assert a[0].endswith(("Answer: 12", "Answer: 13"))
        assert "Answer: 0" in a[1]
        assert a[2].startswith("<instruction>")
        assert "instruction" in a[3]
        assert "arithmetic" in a[4]
        assert len(lm.requests) == 5

    def test_critique_when_nothing_missing(self) -> None:
        text = "step by step verify double-check carefully units"
        out = FakeLM().generate([_req(text, purpose="critique")])[0].text
        assert "already covers" in out

    def test_feedback_steers_proposals(self) -> None:
        lm = FakeLM()
        hits = 0
        for seed in range(40):
            reply = lm.generate([_req("please verify", purpose="reflect", seed=seed)])[0].text
            hits += "verify" in reply
        assert hits >= 18  # preferred phrases are chosen far more often than 0.22 * 40


class TestStreamingCache:
    def test_each_answer_is_cached_on_arrival(self) -> None:
        from prompt_optimizer.backends import CachedLM, FakeLM, ResponseCache
        from prompt_optimizer.backends.base import system_user

        seen: list[int] = []

        class Streaming(FakeLM):
            streams_results = True

            def generate(self, requests, on_result=None):  # type: ignore[no-untyped-def,override]
                out = super().generate(requests)
                for index, resp in enumerate(out):
                    seen.append(len(cache))
                    if on_result is not None:
                        on_result(index, resp)
                return out

        cache = ResponseCache()
        lm = CachedLM(Streaming(), cache)
        reqs = [system_user("s", f"What is {i} + 1?") for i in range(3)]
        lm.generate(reqs)
        assert seen == [0, 1, 2] and len(cache) == 3
        assert all(r.cached for r in lm.generate(reqs))


class TestPermanentErrors:
    def test_bad_request_becomes_an_empty_error_answer(self) -> None:
        import httpx

        from prompt_optimizer.backends import LlamaCppLM
        from prompt_optimizer.backends.base import system_user

        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(400, text="the request exceeds the available context size")

        lm = LlamaCppLM(transport=httpx.MockTransport(handler), retry_delay_s=0.0)
        (resp,) = lm.generate([system_user("s", "u")])
        assert resp.text == "" and resp.finish_reason == "error" and len(calls) == 1
        assert resp.extra["status"] == "400" and "context" in resp.extra["error"]


def test_error_answers_are_not_cached() -> None:
    from prompt_optimizer.backends import CachedLM, FakeLM, ResponseCache
    from prompt_optimizer.backends.base import LMResponse, system_user

    class Failing(FakeLM):
        def generate(self, requests):  # type: ignore[no-untyped-def,override]
            return [LMResponse("", 0, 0, finish_reason="error") for _ in requests]

    class FailingStream(Failing):
        streams_results = True

        def generate(self, requests, on_result=None):  # type: ignore[no-untyped-def,override]
            out = super().generate(requests)
            for i, r in enumerate(out):
                if on_result is not None:
                    on_result(i, r)
            return out

    for inner in (Failing(), FailingStream()):
        cache = ResponseCache()
        CachedLM(inner, cache).generate([system_user("s", "u")])
        assert len(cache) == 0
