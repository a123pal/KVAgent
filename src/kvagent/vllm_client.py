from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class GenerationResult:
    text: str
    ttft_seconds: float
    e2e_seconds: float
    output_chunks: int
    finish_reason: str | None
    usage: dict[str, Any] | None


class VLLMClient:
    """Minimal client for vLLM's OpenAI-compatible Completions endpoint."""

    def __init__(self, base_url: str, model: str, timeout_seconds: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def healthcheck(self) -> None:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{self.base_url}/health")
            response.raise_for_status()

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> GenerationResult:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        start = time.perf_counter()
        first_token_at: float | None = None
        pieces: list[str] = []
        output_chunks = 0
        finish_reason: str | None = None
        usage: dict[str, Any] | None = None

        with httpx.Client(timeout=self.timeout_seconds) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/v1/completions",
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    if event.get("usage"):
                        usage = event["usage"]
                    choices = event.get("choices") or []
                    for choice in choices:
                        text = choice.get("text") or ""
                        if text:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            pieces.append(text)
                            output_chunks += 1
                        if choice.get("finish_reason") is not None:
                            finish_reason = choice["finish_reason"]

        end = time.perf_counter()
        # Empty generations still get a defined TTFT equal to total latency.
        ttft = (first_token_at or end) - start
        return GenerationResult(
            text="".join(pieces),
            ttft_seconds=ttft,
            e2e_seconds=end - start,
            output_chunks=output_chunks,
            finish_reason=finish_reason,
            usage=usage,
        )
