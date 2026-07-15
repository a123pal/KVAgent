# KVAgent

KVAgent is a cache-aware prompt compiler and benchmark for multi-agent LLM workflows. It tests a simple systems hypothesis:

> When several agents operate on the same large artifact, moving stable shared bytes into an identical prompt prefix lets vLLM reuse KV-cache blocks instead of prefilling the artifact once per agent.

The project compares two layouts:

```text
Naive
[agent A role][artifact][task A]
[agent B role][artifact][task B]

Shared-prefix
[stable policy][artifact][task A]
[stable policy][artifact][task B]
               ^ identical prefix
```

## What is implemented

- Exact-prefix-aware prompt compiler
- Naive and shared-prefix baselines
- Four-agent software workflow: translation, security, performance, testing
- Streaming TTFT and end-to-end latency measurement
- vLLM Prometheus metric snapshots for queried and cached prefix tokens
- Reproducible workload isolation using per-run cache namespaces
- Artifact-size and fan-out sweep script
- Unit tests for prompt identity and metric parsing

This MVP uses vLLM's existing Automatic Prefix Caching. The project contribution is the **agent-workflow prompt compiler and evaluation harness**, not a new KV-cache implementation.

## Setup

Requirements:

- Linux system with a CUDA-capable GPU
- Python 3.10+
- A recent vLLM installation

Install KVAgent:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Start vLLM with prefix caching:

```bash
./scripts/run_vllm.sh
```

The default model is `Qwen/Qwen3-4B-Instruct-2507`. Override it with:

```bash
MODEL=your/model ./scripts/run_vllm.sh
```

## Run the benchmark

```bash
kvagent-bench \
  --artifact examples/reverse_lines.c \
  --fanout 4 \
  --repetitions 3 \
  --pad-bytes 65536 \
  --max-tokens 32 \
  --output results/benchmark.json
```

Then summarize it:

```bash
kvagent-analyze results/benchmark.json
```

For a context-size × fan-out sweep:

```bash
python scripts/sweep.py --artifact examples/reverse_lines.c
```

## Metrics to report

The strongest evaluation reports:

1. Median and p95 TTFT
2. Median end-to-end latency with a small fixed decode budget
3. Prefix cache hit tokens / queried prefix tokens
4. Cached prompt tokens
5. Results across 16 KiB, 64 KiB, and 256 KiB artifacts
6. Results across fan-out 2, 3, and 4
7. Output-quality comparison between prompt layouts

Prefix caching only accelerates prefill, not decoding. Keep `--max-tokens` small when isolating the systems effect.


## Important limitation

Transformer KV states are context-dependent. KVAgent does **not** claim that arbitrary middle-of-prompt spans can be reused. It restructures requests so reusable content is an exact common prefix. Moving role instructions after the artifact can affect model behavior, so serious experiments must measure quality as well as latency.
