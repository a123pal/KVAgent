#!/usr/bin/env bash
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
PORT="${PORT:-8000}"

exec vllm serve "$MODEL" \
  --port "$PORT" \
  --enable-prefix-caching \
  --generation-config vllm \
  --gpu-memory-utilization 0.90
