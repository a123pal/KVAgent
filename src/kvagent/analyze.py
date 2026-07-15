from __future__ import annotations

import argparse
import json
from pathlib import Path


def reduction(baseline: float, optimized: float) -> float:
    return (baseline - optimized) / baseline if baseline else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    by_mode = {row["mode"]: row for row in payload["summaries"]}

    naive = by_mode.get("naive")
    shared = by_mode.get("shared-prefix")
    if not naive or not shared:
        raise SystemExit("Result must contain both naive and shared-prefix modes")

    print("KVAgent benchmark comparison")
    print(f"artifact bytes: {payload['metadata']['artifact_bytes']:,}")
    print(f"fan-out: {payload['metadata']['fanout']}")
    print(
        "median TTFT reduction: "
        f"{100 * reduction(naive['median_ttft_seconds'], shared['median_ttft_seconds']):.1f}%"
    )
    print(
        "median E2E reduction: "
        f"{100 * reduction(naive['median_e2e_seconds'], shared['median_e2e_seconds']):.1f}%"
    )
    print(f"naive cache hit rate: {100 * naive['prefix_hit_rate']:.1f}%")
    print(f"shared cache hit rate: {100 * shared['prefix_hit_rate']:.1f}%")
    print(f"shared cached prompt tokens: {shared['cached_prompt_tokens']:,.0f}")


if __name__ == "__main__":
    main()
