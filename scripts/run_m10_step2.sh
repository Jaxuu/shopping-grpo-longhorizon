#!/usr/bin/env bash
# M10 Step 2：验证老师会不会做出目标动作（翻页 / 回头买 / 排除超预算）。
#
# 不需要显卡。只需要两样东西：
#   1) ShopSimulator 环境服务（127.0.0.1:5700）
#   2) 老师模型 API（OPENAI_BASE_URL / OPENAI_API_KEY，deepseek-v4-flash）
#
# 用法：
#   bash scripts/run_m10_step2.sh            # 全流程：健康检查 → 采集 → 判定
#   bash scripts/run_m10_step2.sh analyze    # 只判定（已有 raw.jsonl）
#   bash scripts/run_m10_step2.sh collect    # 只采集
#   bash scripts/run_m10_step2.sh binding    # 复核选题是否真的逼出了动作（纯 CPU，不调 API）
#
# 断点续跑：raw.jsonl 是事实来源，重跑同一条命令会跳过已完成的 (task, attempt)。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODE="${1:-all}"
PY="${PY:-${ROOT}/.venv/bin/python}"
TARGETS="${TARGETS:-data/collection/m10_step2_targets.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/m10-step2}"
RAW="${OUT_DIR}/raw.jsonl"
ENV_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"
MODEL="${OPENAI_MODEL:-deepseek-v4-flash}"

check_environment() {
  local code
  # 404 = 服务在线（根路径没路由）；000 = 没进程。
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 --noproxy '*' "${ENV_URL}/" || true)"
  if [[ "${code}" != "404" ]]; then
    cat >&2 <<EOF
环境服务不可用：${ENV_URL} 返回 ${code}（期望 404）
先在另一个 tmux 会话里起：
  tmux new -s env
  cd ${ROOT} && bash scripts/start_environment.sh
EOF
    exit 1
  fi
  echo "[ok] 环境服务在线：${ENV_URL}"
}

check_teacher() {
  if [[ -z "${OPENAI_BASE_URL:-}" || -z "${OPENAI_API_KEY:-}" ]]; then
    cat >&2 <<'EOF'
老师 API 没配。先 source 环境文件（不要在命令行里贴 key）：
  source ~/.m2_collect.env
  echo $OPENAI_BASE_URL   # 应该是 deepseek 的地址，不是本地 vllm
EOF
    exit 1
  fi
  echo "[ok] 老师模型：${MODEL} @ ${OPENAI_BASE_URL}"
}

collect() {
  check_environment
  check_teacher
  mkdir -p "${OUT_DIR}"
  echo "[run] 采集 90 题 × 1 次，temperature 0.0，max_steps 35"
  "${PY}" scripts/collect_sft_data.py \
    --tasks "${TARGETS}" \
    --output-dir "${OUT_DIR}" \
    --model "${MODEL}" \
    --base-url "${ENV_URL}" \
    --attempts-per-task 1 \
    --workers 8 \
    --target-accepted 90 \
    --temperature 0.0 \
    --max-steps 35
}

analyze() {
  if [[ ! -f "${RAW}" ]]; then
    echo "没有 ${RAW}，先跑采集。" >&2
    exit 1
  fi
  # 判定必须读 raw.jsonl：accepted.jsonl 只剩成功轨迹，看不到老师做没做动作。
  "${PY}" scripts/analyze_m10_step2_verification.py \
    "${RAW}" \
    --targets "${TARGETS}" \
    --detail
}

case "${MODE}" in
  all)     collect; echo; analyze ;;
  collect) collect ;;
  analyze) analyze ;;
  binding)
    # 复核：用老师自己写的 query 重算 gold 名次，判断「必须做目标动作」的处境是否真的成立。
    # 纯 CPU，不调 API，不需要重启任何服务。
    if [[ ! -f "${RAW}" ]]; then
      echo "没有 ${RAW}，先跑采集。" >&2
      exit 1
    fi
    "${PY}" scripts/audit_m10_step2_binding.py \
      "${RAW}" \
      --targets "${TARGETS}" \
      --detail
    ;;
  *) echo "用法: $0 [all|collect|analyze|binding]" >&2; exit 2 ;;
esac
