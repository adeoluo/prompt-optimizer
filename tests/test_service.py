"""Tests for the dependency-free HTTP prediction service (pure handler layer)."""

from __future__ import annotations

from prompt_optimizer.service import PromptService, handle


def _service() -> PromptService:
    return PromptService.compiled()


def test_health() -> None:
    status, payload = handle("GET", "/health", {}, _service())
    assert status == 200
    assert payload["status"] == "ok"
    assert "instruction" in payload


def test_predict() -> None:
    status, payload = handle(
        "POST", "/predict", {"text": "I need a refund for my money"}, _service()
    )
    assert status == 200
    assert payload["label"] == "refund"


def test_predict_rejects_missing_text() -> None:
    status, payload = handle("POST", "/predict", {}, _service())
    assert status == 400
    assert "text" in payload["error"]


def test_predict_rejects_overlong_text() -> None:
    status, _ = handle("POST", "/predict", {"text": "a" * 5000}, _service())
    assert status == 400


def test_compile_returns_prompt() -> None:
    status, payload = handle("POST", "/compile", {}, _service())
    assert status == 200
    assert "instruction" in payload
    assert isinstance(payload["demonstrations"], list)


def test_unknown_route() -> None:
    status, payload = handle("GET", "/nope", {}, _service())
    assert status == 404
    assert "error" in payload
