#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-Qwen/Qwen3.5-2B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-shopping-agent}"
LLM_PORT="${LLM_PORT:-8000}"
# 22G 卡上 2B 推理够用；若 vLLM 报 KV cache 不足，降到 0.85 或 0.8 重试。
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"

if [[ ! -x "$ROOT/.venv/bin/vllm" ]]; then
  echo "vLLM is not installed. Run: bash scripts/setup.sh" >&2
  exit 1
fi

exec "$ROOT/.venv/bin/vllm" serve "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --port "$LLM_PORT" \
  --max-model-len 24576 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
