#!/usr/bin/env bash
# M10-R：跑题 → 筛「正确答案不在第 1 名」的轨迹。
#
# 为什么这么做（2026-10-20 实测）：SFT 池里 65.9% 的样本，打开的 ASIN 就是搜索第 1 名；
# 评测里「第 1 名对 → 86.8%，不对 → 34.0%」⇒ 模型学成了「开第一个」。
# 把这类样本占比从 34% 推到 50–70%，「开第一个」才会亏本。
#
# 不需要显卡：跑题用老师 API，筛样本是纯 CPU。
#
# 用法：
#   bash scripts/run_m10_rank2.sh collect 1000   # 跑前 1000 题（可反复跑，自动续跑）
#   bash scripts/run_m10_rank2.sh filter         # 筛样本 + 统计（不写文件）
#   bash scripts/run_m10_rank2.sh all 1000       # 先跑再筛
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODE="${1:-all}"
LIMIT="${2:-1000}"
PY="${PY:-${ROOT}/.venv/bin/python}"
TARGETS="${TARGETS:-data/collection/m10_rank2_targets.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/m10-rank2}"
RAW="${OUT_DIR}/raw.jsonl"
KEPT="${OUT_DIR}/kept.jsonl"
ENV_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"
MODEL="${OPENAI_MODEL:-deepseek-v4-flash}"

check_environment() {
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 --noproxy '*' "${ENV_URL}/" || true)"
  if [[ "${code}" != "404" ]]; then
    cat >&2 <<EOF
环境服务不可用：${ENV_URL} 返回 ${code}（期望 404）
先起：tmux new -s env && cd ${ROOT} && bash scripts/start_environment.sh
EOF
    exit 1
  fi
  echo "[ok] 环境服务在线：${ENV_URL}"
}

check_teacher() {
  if [[ -z "${OPENAI_BASE_URL:-}" || -z "${OPENAI_API_KEY:-}" ]]; then
    echo "老师 API 没配：先 export OPENAI_BASE_URL / OPENAI_API_KEY（或 source 环境文件）。" >&2
    exit 1
  fi
  echo "[ok] 老师模型：${MODEL} @ ${OPENAI_BASE_URL}"
}

collect() {
  local limit="${1}"
  check_environment
  check_teacher
  mkdir -p "${OUT_DIR}"
  echo "[run] 跑 ${limit} 题 × 1 次，temperature 0.0，max_steps 35（已有轨迹会自动跳过）"
  "${PY}" scripts/collect_sft_data.py \
    --tasks "${TARGETS}" \
    --output-dir "${OUT_DIR}" \
    --model "${MODEL}" \
    --base-url "${ENV_URL}" \
    --attempts-per-task 1 \
    --limit "${limit}" \
    --workers 8 \
    --target-accepted "${limit}" \
    --temperature 0.0 \
    --max-steps 35
}

filter() {
  if [[ ! -f "${RAW}" ]]; then
    echo "没有 ${RAW}，先跑 collect。" >&2
    exit 1
  fi
  "${PY}" scripts/filter_m10_rank2.py "${RAW}" --out "${KEPT}"
}

case "${MODE}" in
  collect) collect "${LIMIT}" ;;
  filter)  filter ;;
  all)     collect "${LIMIT}"; echo; filter ;;
  *) echo "用法: $0 [collect|filter|all] [题数]" >&2; exit 2 ;;
esac
