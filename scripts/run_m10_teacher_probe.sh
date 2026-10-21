#!/usr/bin/env bash
# M10-R 探针：老师在高温度、多次采样下会不会自己翻页 / 换词重搜 / 对比候选。
#
# 为什么要有这一步
# ----------------
# M10 Step 2 那次采集是「每题 1 次、temperature 0.0」，量到的是老师的**默认行为**，
# 三条目标动作全部 <20% 就地关闭。但「默认不做」不等于「做不到」。
# 训练很贵，所以在决定「脚本合成」之前，先用最低成本把这个问题问清楚：
#
#   把温度提上去、同一道题跑 4 次，老师会不会自己做出目标动作？
#
#   会（轨迹级 ≥20%）⇒ 走真实采集，不需要合成。
#   不会            ⇒ 采集拿不到，才轮到合成。
#
# 成本：默认 120 题 × 4 次 = 480 条轨迹，8 并发，约 1–2 小时，不用显卡。
#
# 用法
# ----
#   bash scripts/run_m10_teacher_probe.sh            # 抽样 → 采集 → 判定
#   bash scripts/run_m10_teacher_probe.sh api        # 老师 API 参数体检（几十秒，不碰环境）
#   bash scripts/run_m10_teacher_probe.sh smoke      # 每场景 1 题试跑（验温度参数）
#   bash scripts/run_m10_teacher_probe.sh targets    # 只抽样（纯 CPU）
#   bash scripts/run_m10_teacher_probe.sh collect    # 只采集（可反复跑，自动续跑）
#   bash scripts/run_m10_teacher_probe.sh analyze    # 只判定（已有 raw.jsonl）
#
# 常用覆盖：
#   PROBE_PER_SCENARIO=30 PROBE_ATTEMPTS=6 PROBE_TEMPERATURE=1.0 \
#     bash scripts/run_m10_teacher_probe.sh
#
# 断点续跑：raw.jsonl 是事实来源，重跑同一条命令会跳过已完成的 (task, attempt)。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODE="${1:-all}"
PY="${PY:-${ROOT}/.venv/bin/python}"
TARGETS="${TARGETS:-data/collection/m10_teacher_probe_targets.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/m10-teacher-probe}"
RAW="${OUT_DIR}/raw.jsonl"
ENV_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"
MODEL="${OPENAI_MODEL:-deepseek-v4-flash}"

PROBE_PER_SCENARIO="${PROBE_PER_SCENARIO:-40}"
PROBE_ATTEMPTS="${PROBE_ATTEMPTS:-4}"
PROBE_TEMPERATURE="${PROBE_TEMPERATURE:-0.9}"
PROBE_WORKERS="${PROBE_WORKERS:-8}"

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
    cat >&2 <<'EOF'
老师 API 没配。先 source 环境文件（不要在命令行里贴 key）：
  source ~/.m2_collect.env
  echo $OPENAI_BASE_URL   # 应该是 deepseek 的地址，不是本地 vllm
EOF
    exit 1
  fi
  echo "[ok] 老师模型：${MODEL} @ ${OPENAI_BASE_URL}"
}

build_targets() {
  "${PY}" scripts/build_m10_teacher_probe.py \
    --per-scenario "${PROBE_PER_SCENARIO}" \
    --output "${TARGETS}"
}

count_targets() {
  if [[ ! -f "${TARGETS}" ]]; then
    echo "0"
    return
  fi
  grep -c . "${TARGETS}" || true
}

smoke() {
  check_environment
  check_teacher
  local tmp="/tmp/m10-teacher-probe-smoke"
  mkdir -p "${tmp}"
  rm -f "${tmp}/raw.jsonl"
  # 每场景 1 题：确认「温度 0.9 这个参数服务端收」+「探针清单能跑」。
  "${PY}" scripts/build_m10_teacher_probe.py \
    --per-scenario 1 \
    --output "${tmp}/targets.jsonl"
  local tasks
  tasks="$(grep -c . "${tmp}/targets.jsonl")"
  echo "[smoke] ${tasks} 题 × 1 次，temperature ${PROBE_TEMPERATURE}"
  "${PY}" scripts/collect_sft_data.py \
    --tasks "${tmp}/targets.jsonl" \
    --output-dir "${tmp}" \
    --model "${MODEL}" \
    --base-url "${ENV_URL}" \
    --attempts-per-task 1 \
    --workers 1 \
    --target-accepted "${tasks}" \
    --temperature "${PROBE_TEMPERATURE}" \
    --max-steps 35
  echo "[smoke] 通过：raw 里有 $(wc -l < "${tmp}/raw.jsonl") 条轨迹。"
  # 冒烟必须自带诊断：3 条轨迹全部 has_error 时，上面那行 summary 只说明「跑完了」，
  # 不说明「跑对了」。这里直接把异常类型和调用栈打出来。
  echo
  # 用退出码判「判定是否有效」（3 = 异常太多）。不要 grep JSON 里的 "error"
  # 子串——观测文本里出现这个词是正常的，会造成误报。
  if ! "${PY}" scripts/analyze_teacher_action_rate.py \
    "${tmp}/raw.jsonl" \
    --targets "${tmp}/targets.jsonl" \
    --detail; then
    echo
    echo "[smoke] ⚠️ 有轨迹带异常。下一步不是跑全量，是先看 API 到底返回什么："
    echo "  bash scripts/run_m10_teacher_probe.sh api"
    exit 1
  fi
  echo
  echo "[smoke] 无异常，可以跑全量：bash scripts/run_m10_teacher_probe.sh"
}

collect() {
  check_environment
  check_teacher
  local tasks total
  tasks="$(count_targets)"
  if [[ "${tasks}" -eq 0 ]]; then
    echo "没有探针清单 ${TARGETS}，先跑 targets。" >&2
    exit 1
  fi
  # 关键：--target-accepted 必须等于「题数 × 次数」，否则采到足够多的成功轨迹就早停，
  # 后面的题一次都不跑，探针会变成「只测了前几道」。
  total=$(( tasks * PROBE_ATTEMPTS ))
  mkdir -p "${OUT_DIR}"
  echo "[run] ${tasks} 题 × ${PROBE_ATTEMPTS} 次 = ${total} 条，temperature ${PROBE_TEMPERATURE}，${PROBE_WORKERS} 并发"
  "${PY}" scripts/collect_sft_data.py \
    --tasks "${TARGETS}" \
    --output-dir "${OUT_DIR}" \
    --model "${MODEL}" \
    --base-url "${ENV_URL}" \
    --attempts-per-task "${PROBE_ATTEMPTS}" \
    --workers "${PROBE_WORKERS}" \
    --target-accepted "${total}" \
    --temperature "${PROBE_TEMPERATURE}" \
    --max-steps 35
}

analyze() {
  if [[ ! -f "${RAW}" ]]; then
    echo "没有 ${RAW}，先跑 collect。" >&2
    exit 1
  fi
  # 判定必须读 raw.jsonl：accepted.jsonl 只剩成功轨迹，看不到「做没做这个动作」。
  "${PY}" scripts/analyze_teacher_action_rate.py \
    "${RAW}" \
    --targets "${TARGETS}" \
    --pool data/sft_pure_v4/all.jsonl \
    --detail
}

case "${MODE}" in
  all)     build_targets; echo; collect; echo; analyze ;;
  targets) build_targets ;;
  collect) collect ;;
  analyze) analyze ;;
  smoke)   smoke ;;
  api)
    # 只打 API：不带环境、不采集，几十秒。冒烟报 400 时先跑这个。
    # 多余参数透传，例如：... api --base-url https://... --without-session
    # 这里只校验 key：base_url 可以由 --base-url 指定，不要求环境变量。
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      echo "老师 API 没配：先 export OPENAI_API_KEY（或 source 环境文件）。" >&2
      exit 1
    fi
    echo "[ok] 老师模型：${MODEL} @ ${OPENAI_BASE_URL:-（由 --base-url 指定）}"
    shift || true
    "${PY}" scripts/probe_teacher_api.py --model "${MODEL}" "$@"
    ;;
  *)
    echo "未知模式：${MODE}（可选 all / targets / collect / analyze / smoke / api）" >&2
    exit 2
    ;;
esac
