#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${1:-model}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$ROOT/outputs/evaluation/$LABEL}"
SHOPSIM_BASE_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-shopping-agent}"
# 诊断开关：买之前必须打开过多少个不同候选。默认 0=关闭，行为与历史评测完全一致。
MIN_OPENED_BEFORE_BUY="${MIN_OPENED_BEFORE_BUY:-0}"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"
"$ROOT/.venv/bin/python" scripts/evaluate_shop_benchmark.py \
  --benchmark data/evaluation/tasks.jsonl \
  --output "$OUTPUT_DIR/trajectories.jsonl" \
  --summary "$OUTPUT_DIR/summary.json" \
  --base-url "$SHOPSIM_BASE_URL" \
  --model "$SERVED_MODEL_NAME" \
  --llm-base-url "$LLM_BASE_URL" \
  --api-key "$LLM_API_KEY" \
  --min-opened-before-buy "$MIN_OPENED_BEFORE_BUY"

"$ROOT/.venv/bin/python" scripts/build_eval_report.py --run-dir "$OUTPUT_DIR"
