#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${1:-model}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$ROOT/outputs/evaluation/$LABEL}"
# 评测集默认是冻结的 Final-200。要跑 dev 集或训练集自测卷时显式传 BENCHMARK=，
# 产物目录也请另起名字（Final-200 的分数只能和 Final-200 比）。
BENCHMARK="${BENCHMARK:-$ROOT/data/evaluation/tasks.jsonl}"
SHOPSIM_BASE_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-shopping-agent}"
# 诊断开关：买之前必须打开过多少个不同候选。默认 0=关闭，行为与历史评测完全一致。
MIN_OPENED_BEFORE_BUY="${MIN_OPENED_BEFORE_BUY:-0}"
# 采样温度。默认 0.0 = 贪心解码，与全部历史评测口径一致（分数可比）。
# 只有做「同一模型多次采样再投票」时才需要调高；调高后单次分数会掉，别拿它比历史。
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
# 系统提示词变体。留空 = 用冻结的 SYSTEM_PROMPT（分数与历史可比）。
# 非空时 = 「同一权重、只改提示词」的 A/B，分数不可与 62.0%/63.5% 直接比，
# protocol 里会记文件名与指纹。例：SYSTEM_PROMPT_FILE=configs/prompts/query-attributes.txt
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-}"
PROMPT_ARGS=()
if [[ -n "$SYSTEM_PROMPT_FILE" ]]; then
  PROMPT_ARGS=(--system-prompt-file "$SYSTEM_PROMPT_FILE")
fi
# 每题步数上限固定 35，不给环境变量：环境侧 configs/environment.json 的
# termination.max_steps=35 是冻结契约，SHOP_MAX_STEPS 设成别的值会直接抛 ValueError。
# 调大这里的值没有意义（环境已经在 35 步终止），调小会让客户端先停而环境不终止，
# 产出的 max_steps 轨迹与历史不可比。

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"
"$ROOT/.venv/bin/python" scripts/evaluate_shop_benchmark.py \
  --benchmark "$BENCHMARK" \
  --output "$OUTPUT_DIR/trajectories.jsonl" \
  --summary "$OUTPUT_DIR/summary.json" \
  --base-url "$SHOPSIM_BASE_URL" \
  --model "$SERVED_MODEL_NAME" \
  --llm-base-url "$LLM_BASE_URL" \
  --api-key "$LLM_API_KEY" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --min-opened-before-buy "$MIN_OPENED_BEFORE_BUY" \
  ${PROMPT_ARGS[@]+"${PROMPT_ARGS[@]}"}

"$ROOT/.venv/bin/python" scripts/build_eval_report.py --run-dir "$OUTPUT_DIR"
