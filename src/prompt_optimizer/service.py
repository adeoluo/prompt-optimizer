"""A dependency-free HTTP service exposing the compiled prompt.

The service compiles the optimized prompt once at startup and then answers
classification requests with it, which is how a compiled prompt would be served
in production. A pure handler is separated from the socket binding for testing.

Endpoints
---------
``GET  /health``   liveness, instruction, and demonstration count.
``POST /predict``  body ``{"text": str}``; returns the predicted label.
``POST /compile``  recompile and return the optimized prompt (instruction + demos).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from prompt_optimizer.data import CANDIDATE_INSTRUCTIONS, NAIVE_INSTRUCTION, TRAIN, VALIDATION
from prompt_optimizer.optimizers import CompoundOptimizer
from prompt_optimizer.predictor import Predictor
from prompt_optimizer.types import Prompt

JsonObject = dict[str, Any]
_MAX_TEXT_CHARS = 2000


@dataclass(frozen=True)
class PromptService:
    """Holds the compiled prompt and serves predictions."""

    predictor: Predictor

    @classmethod
    def compiled(cls) -> PromptService:
        base = Prompt(NAIVE_INSTRUCTION)
        prompt = CompoundOptimizer(CANDIDATE_INSTRUCTIONS).compile(base, TRAIN, VALIDATION).prompt
        return cls(predictor=Predictor(prompt))

    def predict(self, text: str) -> JsonObject:
        return {"text": text, "label": self.predictor.predict(text)}

    def prompt_payload(self) -> JsonObject:
        prompt = self.predictor.prompt
        return {
            "instruction": prompt.instruction,
            "demonstrations": [
                {"text": demo.text, "label": demo.label} for demo in prompt.demonstrations
            ],
        }


def _validate_text(body: Mapping[str, Any]) -> str:
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("'text' must be a non-empty string")
    if len(text) > _MAX_TEXT_CHARS:
        raise ValueError(f"'text' exceeds {_MAX_TEXT_CHARS} characters")
    return text


def handle(
    method: str, path: str, body: Mapping[str, Any], service: PromptService
) -> tuple[int, JsonObject]:
    """Pure request handler: route a request to a status code and payload."""

    if method == "GET" and path == "/health":
        prompt = service.predictor.prompt
        return 200, {
            "status": "ok",
            "instruction": prompt.instruction,
            "demonstrations": len(prompt.demonstrations),
        }
    if method == "POST" and path == "/predict":
        try:
            text = _validate_text(body)
        except ValueError as error:
            return 400, {"error": str(error)}
        return 200, service.predict(text)
    if method == "POST" and path == "/compile":
        return 200, service.prompt_payload()
    return 404, {"error": f"no route for {method} {path}"}


class _Handler(BaseHTTPRequestHandler):  # pragma: no cover - exercised over sockets
    service: PromptService

    def _respond(self, status: int, payload: JsonObject) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        status, payload = handle("GET", self.path, {}, self.service)
        self._respond(status, payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._respond(400, {"error": "request body is not valid JSON"})
            return
        status, payload = handle("POST", self.path, body, self.service)
        self._respond(status, payload)

    def log_message(self, *_: Any) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    """Start the HTTP service (blocking)."""

    handler = type("_BoundHandler", (_Handler,), {"service": PromptService.compiled()})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"prompt-optimizer service listening on http://{host}:{port}")
    server.serve_forever()
