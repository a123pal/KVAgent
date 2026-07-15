from __future__ import annotations

import re
from dataclasses import dataclass

import httpx


_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$"
)


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum samples by metric name, ignoring labels.

    This is sufficient for a single-model benchmark and remains robust when vLLM
    appends Prometheus's `_total` suffix to Counter samples.
    """
    values: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.match(line)
        if match is None:
            continue
        name = match.group("name")
        values[name] = values.get(name, 0.0) + float(match.group("value"))
    return values


def _first(metrics: dict[str, float], *names: str) -> float:
    for name in names:
        if name in metrics:
            return metrics[name]
    return 0.0


@dataclass(frozen=True)
class CacheSnapshot:
    prefix_queries: float
    prefix_hits: float
    prompt_tokens: float
    cached_prompt_tokens: float

    @property
    def hit_rate(self) -> float:
        return self.prefix_hits / self.prefix_queries if self.prefix_queries else 0.0

    def delta(self, earlier: "CacheSnapshot") -> "CacheSnapshot":
        return CacheSnapshot(
            prefix_queries=max(0.0, self.prefix_queries - earlier.prefix_queries),
            prefix_hits=max(0.0, self.prefix_hits - earlier.prefix_hits),
            prompt_tokens=max(0.0, self.prompt_tokens - earlier.prompt_tokens),
            cached_prompt_tokens=max(
                0.0, self.cached_prompt_tokens - earlier.cached_prompt_tokens
            ),
        )


class VLLMMetricsClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def snapshot(self) -> CacheSnapshot:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{self.base_url}/metrics")
            response.raise_for_status()
        metrics = parse_prometheus(response.text)
        return CacheSnapshot(
            prefix_queries=_first(
                metrics,
                "vllm:prefix_cache_queries_total",
                "vllm:prefix_cache_queries",
            ),
            prefix_hits=_first(
                metrics,
                "vllm:prefix_cache_hits_total",
                "vllm:prefix_cache_hits",
            ),
            prompt_tokens=_first(
                metrics,
                "vllm:prompt_tokens_total",
                "vllm:prompt_tokens",
            ),
            cached_prompt_tokens=_first(
                metrics,
                "vllm:prompt_tokens_cached_total",
                "vllm:prompt_tokens_cached",
            ),
        )
