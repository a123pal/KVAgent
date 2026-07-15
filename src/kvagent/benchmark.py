from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .metrics import CacheSnapshot, VLLMMetricsClient
from .prompt_compiler import AgentTask, PromptCompiler, PromptMode
from .vllm_client import VLLMClient


DEFAULT_TASKS = (
    AgentTask(
        name="rust-translator",
        role="senior Rust systems engineer",
        instruction=(
            "Translate the C program into safe, idiomatic Rust. Preserve observable behavior, "
            "include error handling, and return one complete Rust source file."
        ),
    ),
    AgentTask(
        name="security-reviewer",
        role="memory-safety and secure-code reviewer",
        instruction=(
            "Audit the program for memory safety, integer overflow, file handling, and undefined "
            "behavior. Return a prioritized review with concrete fixes."
        ),
    ),
    AgentTask(
        name="performance-reviewer",
        role="low-level performance engineer",
        instruction=(
            "Analyze asymptotic cost, allocation behavior, I/O behavior, and cache efficiency. "
            "Return specific optimizations without changing semantics."
        ),
    ),
    AgentTask(
        name="test-author",
        role="software verification engineer",
        instruction=(
            "Design edge-case and property-style tests for the program. Return a compact test plan "
            "and representative test inputs with expected behavior."
        ),
    ),
)


@dataclass(frozen=True)
class RequestRecord:
    mode: str
    repetition: int
    agent: str
    prompt_chars: int
    shared_prefix_chars: int
    shared_prefix_sha256: str
    ttft_seconds: float
    e2e_seconds: float
    output_chars: int
    finish_reason: str | None
    usage: dict | None


@dataclass(frozen=True)
class ModeSummary:
    mode: str
    request_count: int
    median_ttft_seconds: float
    p95_ttft_seconds: float
    median_e2e_seconds: float
    prefix_query_tokens: float
    prefix_hit_tokens: float
    prefix_hit_rate: float
    prompt_tokens: float
    cached_prompt_tokens: float


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def pad_artifact(artifact: str, target_bytes: int) -> str:
    current = len(artifact.encode("utf-8"))
    if target_bytes <= current:
        return artifact
    remaining = target_bytes - current
    line = "/* kvagent benchmark padding: stable shared context */\n"
    repetitions = (remaining // len(line.encode("utf-8"))) + 1
    return artifact + "\n" + (line * repetitions)


def run_mode(
    *,
    mode: PromptMode,
    artifact: str,
    tasks: tuple[AgentTask, ...],
    repetitions: int,
    compiler: PromptCompiler,
    client: VLLMClient,
    metrics: VLLMMetricsClient,
    max_tokens: int,
    experiment_id: str,
) -> tuple[list[RequestRecord], ModeSummary]:
    before = metrics.snapshot()
    records: list[RequestRecord] = []

    # Sequential execution intentionally exposes warm-prefix behavior: request 1
    # fills the cache and later branches should reuse it.
    for repetition in range(repetitions):
        for task in tasks:
            compiled = compiler.compile(
                artifact=artifact,
                task=task,
                mode=mode,
                experiment_id=experiment_id,
            )
            result = client.generate(
                compiled.text,
                max_tokens=max_tokens,
                temperature=0.0,
                seed=0,
            )
            records.append(
                RequestRecord(
                    mode=mode.value,
                    repetition=repetition,
                    agent=task.name,
                    prompt_chars=len(compiled.text),
                    shared_prefix_chars=len(compiled.shared_prefix),
                    shared_prefix_sha256=compiled.shared_prefix_sha256,
                    ttft_seconds=result.ttft_seconds,
                    e2e_seconds=result.e2e_seconds,
                    output_chars=len(result.text),
                    finish_reason=result.finish_reason,
                    usage=result.usage,
                )
            )

    after = metrics.snapshot()
    delta: CacheSnapshot = after.delta(before)
    ttfts = [record.ttft_seconds for record in records]
    e2es = [record.e2e_seconds for record in records]
    summary = ModeSummary(
        mode=mode.value,
        request_count=len(records),
        median_ttft_seconds=statistics.median(ttfts),
        p95_ttft_seconds=percentile(ttfts, 0.95),
        median_e2e_seconds=statistics.median(e2es),
        prefix_query_tokens=delta.prefix_queries,
        prefix_hit_tokens=delta.prefix_hits,
        prefix_hit_rate=delta.hit_rate,
        prompt_tokens=delta.prompt_tokens,
        cached_prompt_tokens=delta.cached_prompt_tokens,
    )
    return records, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--fanout", type=int, default=4, choices=range(2, 5))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--pad-bytes", type=int, default=65536)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument(
        "--mode",
        choices=("both", PromptMode.NAIVE.value, PromptMode.SHARED_PREFIX.value),
        default="both",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be >= 1")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be >= 1")

    artifact = pad_artifact(args.artifact.read_text(encoding="utf-8"), args.pad_bytes)
    tasks = DEFAULT_TASKS[: args.fanout]
    client = VLLMClient(args.base_url, args.model)
    metrics = VLLMMetricsClient(args.base_url)
    compiler = PromptCompiler()
    client.healthcheck()

    selected_modes = (
        (PromptMode.NAIVE, PromptMode.SHARED_PREFIX)
        if args.mode == "both"
        else (PromptMode(args.mode),)
    )

    all_records: list[RequestRecord] = []
    summaries: list[ModeSummary] = []
    run_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"

    for mode in selected_modes:
        records, summary = run_mode(
            mode=mode,
            artifact=artifact,
            tasks=tasks,
            repetitions=args.repetitions,
            compiler=compiler,
            client=client,
            metrics=metrics,
            max_tokens=args.max_tokens,
            experiment_id=f"{run_id}-{mode.value}",
        )
        all_records.extend(records)
        summaries.append(summary)

    output = {
        "metadata": {
            "run_id": run_id,
            "model": args.model,
            "base_url": args.base_url,
            "artifact": str(args.artifact),
            "artifact_bytes": len(artifact.encode("utf-8")),
            "fanout": args.fanout,
            "repetitions": args.repetitions,
            "max_tokens": args.max_tokens,
        },
        "summaries": [asdict(summary) for summary in summaries],
        "requests": [asdict(record) for record in all_records],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summaries"], indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
