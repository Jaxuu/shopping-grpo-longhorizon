#!/usr/bin/env bash
# M10-R 全流程：每一步一个 tmux 会话，断线不影响，随时能回去看。
#
# 为什么要有这个：训练 2 小时、评测 1 小时，全靠 SSH 会话保持不现实；而
# 手工在三个 tmux 里粘命令，任何一步粘错都会让人以为「卡住了」。这个脚本
# 只做三件事——**检查前置条件、把命令丢进 tmux、事后能看状态**。
#
# 用法
# ----
#   bash scripts/run_m10r_pipeline.sh env      # 0. 依赖（uv sync，幂等，首次 3–5 分钟）
#   bash scripts/run_m10r_pipeline.sh build    # 1. 建池（纯 CPU，约 2 分钟）
#   bash scripts/run_m10r_pipeline.sh train    # 2. 训练（要显卡，约 2–2.5 小时）
#   bash scripts/run_m10r_pipeline.sh eval     # 3. Final-200（自带商店环境 + vLLM，40–60 分钟）
#   bash scripts/run_m10r_pipeline.sh status   # 随时看：会话活着吗、跑到哪、显存多少
#
# 会话名：m10r-env / m10r-build / m10r-train / m10r-eval
#   回去看：tmux attach -t m10r-train    脱离：Ctrl+B 松手再按 D
#
# 可覆盖的环境变量
#   KEEP_TEMPLATE=700      模板保留条数（建池）
#   LABEL=m10r-final200    评测输出目录名
#   MODEL=...              评测权重（默认 outputs/models/sft-m10r/stage-b/merged）
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PY="${PY:-${ROOT}/.venv/bin/python}"
LOG_DIR="${ROOT}/outputs/logs"
CAPPED="${ROOT}/data/sft_m10r/capped"
MODEL_DIR="${ROOT}/outputs/models/sft-m10r"
LABEL="${LABEL:-m10r-final200}"
MODEL="${MODEL:-${MODEL_DIR}/stage-b/merged}"
KEEP_TEMPLATE="${KEEP_TEMPLATE:-700}"

mkdir -p "${LOG_DIR}"

die() { printf '失败：%s\n' "$*" >&2; exit 1; }
need_tmux() { command -v tmux >/dev/null 2>&1 || die "找不到 tmux，先 apt install tmux"; }

# 起一个会话；已存在就跳过（不重复跑、不覆盖日志）。
launch() {
  local name="$1"
  shift
  local body="cd '${ROOT}' && { $* ; } 2>&1 | tee -a '${LOG_DIR}/${name}.log'; \
     echo; echo '[结束] 会话保留中；Ctrl+B 松手按 D 脱离，或 exit 关闭'; exec bash"
  # PIPELINE_DRY=1 只打印不启动：改完脚本先看一眼要跑什么，别拿 2 小时的训练试错。
  if [[ "${PIPELINE_DRY:-0}" == "1" ]]; then
    echo "[dry] 会话 [${name}] 将执行："
    printf '  %s\n' "${body}"
    return 0
  fi
  need_tmux
  if tmux has-session -t "${name}" 2>/dev/null; then
    echo "[skip] 会话 [${name}] 已在跑：tmux attach -t ${name}"
    return 0
  fi
  tmux new-session -d -s "${name}" "${body}"
  echo "[run] 会话 [${name}] 已启动 · 日志 ${LOG_DIR}/${name}.log"
  echo "      回去看：tmux attach -t ${name}（脱离：Ctrl+B 松手再按 D）"
}

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

free_gb() {
  df -Pk "$1" 2>/dev/null | awk 'NR==2{printf "%d", $4/1024/1024}'
}

# ---------------------------------------------------------------- env
step_env() {
  [[ -f "${ROOT}/scripts/setup.sh" ]] || die "找不到 scripts/setup.sh"
  command -v uv >/dev/null 2>&1 || die "找不到 uv；这台机器上 bash scripts/setup.sh 需要 uv"
  launch m10r-env "bash scripts/setup.sh"
  echo "[提示] 首次 3–5 分钟；已同步过会很快。用 status 看结果。"
}

# ---------------------------------------------------------------- build
step_build() {
  [[ -x "${PY}" ]] || die "找不到 Python：${PY}（先跑 env 步骤，或设置 PY）"
  launch m10r-build "KEEP_TEMPLATE=${KEEP_TEMPLATE} bash scripts/run_m10r_build.sh"
}

# ---------------------------------------------------------------- train
preflight_train() {
  local problems=0
  [[ -f "${CAPPED}/all.jsonl" ]] || { echo "缺 ${CAPPED}/all.jsonl（先跑 build）" >&2; problems=1; }
  [[ -f "${CAPPED}/manifest.json" ]] || { echo "缺 ${CAPPED}/manifest.json（先跑 build）" >&2; problems=1; }
  [[ "${problems}" -eq 0 ]] || return 1

  # run_sft_curriculum 会拒绝覆盖已完成的阶段，提前说清楚，别跑到一半才报错。
  local stale=""
  for stage in a b; do
    if [[ -d "${MODEL_DIR}/stage-${stage}/merged" ]] \
       && [[ -n "$(ls -A "${MODEL_DIR}/stage-${stage}/merged" 2>/dev/null)" ]]; then
      stale="${stale} ${MODEL_DIR}/stage-${stage}/merged"
    fi
  done
  if [[ -n "${stale}" ]]; then
    cat >&2 <<EOF
已有训练产物，脚本会拒绝覆盖：${stale}
要重训就先挪走：
  mv ${MODEL_DIR} ${MODEL_DIR}.$(date +%m%d-%H%M)
EOF
    return 1
  fi

  local used
  used="$(gpu_used_mib)"
  if [[ -n "${used}" ]] && (( used > 2000 )); then
    cat >&2 <<EOF
显卡上有别的进程占着 ${used} MiB。先清掉，否则会和训练抢显存：
  nvidia-smi                 # 看是谁
  pkill -f vllm              # 或者按 PID kill
EOF
    return 1
  fi

  local free
  free="$(free_gb "${ROOT}/outputs")"
  if [[ -n "${free}" ]] && (( free < 30 )); then
    echo "outputs 所在盘只剩 ${free} GB；两段 merged 各约 4 GB，建议先清出 30 GB。" >&2
    return 1
  fi
  echo "[ok] 前置检查通过（显存 ${used:-?} MiB 已用，outputs 余量 ${free:-?} GB）"
}

step_train() {
  [[ -x "${PY}" ]] || die "找不到 Python：${PY}"
  if [[ "${PIPELINE_DRY:-0}" == "1" ]]; then
    echo "[dry] 跳过前置检查（PIPELINE_DRY=1）"
  else
    preflight_train || die "前置检查没过，见上面的提示"
  fi
  # expandable_segments 是历史踩坑的固定配置（碎片回收过 4.4 GiB），别省。
  launch m10r-train \
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     bash scripts/sft_curriculum.sh \
       --source '${CAPPED}/all.jsonl' \
       --manifest '${CAPPED}/manifest.json' \
       --output-root '${MODEL_DIR}' \
       --loss-scope tool-call-only"
  echo "[提示] 约 2–2.5 小时。进度：bash scripts/run_m10r_pipeline.sh status"
}

# ---------------------------------------------------------------- eval
step_eval() {
  if [[ "${PIPELINE_DRY:-0}" != "1" ]]; then
    [[ -f "${MODEL}/config.json" ]] || die "找不到权重：${MODEL}/config.json（训练没跑完？或换 MODEL=...）"
  fi
  # MIN_OPENED_BEFORE_BUY 默认是 3（强制多开的诊断开关），Final-200 必须关掉，
  # 否则测出来的不是真实水平。
  launch m10r-eval \
    "MIN_OPENED_BEFORE_BUY=0 MODEL='${MODEL}' LABEL='${LABEL}' \
     bash scripts/run_eval_m9.sh"
  echo "[提示] 自带商店环境（tmux env）+ vLLM（tmux vllm）；vLLM 首次加载 20–25 分钟。"
}

# ---------------------------------------------------------------- status
step_status() {
  echo "== 会话"
  local any=0
  for name in m10r-env m10r-build m10r-train m10r-eval env vllm; do
    if tmux has-session -t "${name}" 2>/dev/null; then
      echo "  [活] ${name}"
      any=1
    fi
  done
  (( any )) || echo "  （没有正在跑的会话）"

  echo
  echo "== 显卡"
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null || echo "  nvidia-smi 不可用"

  echo
  echo "== 产物"
  for path in \
    "data/sft_m10r/increment_merged.jsonl" \
    "data/sft_m10r/all.jsonl" \
    "data/sft_m10r/capped/all.jsonl"; do
    if [[ -f "${ROOT}/${path}" ]]; then
      echo "  ${path}: $(wc -l < "${ROOT}/${path}") 行"
    else
      echo "  ${path}: 缺"
    fi
  done
  if [[ -f "${CAPPED}/manifest.json" ]]; then
    "${PY}" scripts/show_sft_manifest.py "${CAPPED}/manifest.json" 2>/dev/null | sed -n '1,12p' | sed 's/^/  /'
  fi

  echo
  echo "== 训练/评测进度（日志末尾）"
  for name in m10r-train m10r-eval; do
    if [[ -f "${LOG_DIR}/${name}.log" ]]; then
      echo "  --- ${name} ---"
      grep -E "stage|loss|epoch|eval|error|Error|OOM" "${LOG_DIR}/${name}.log" 2>/dev/null | tail -5 | sed 's/^/    /'
      echo "    最后一行：$(tail -1 "${LOG_DIR}/${name}.log" | cut -c1-140)"
    fi
  done
  if [[ -d "${MODEL_DIR}" ]]; then
    echo
    echo "== 模型目录"
    ls -1 "${MODEL_DIR}" 2>/dev/null | sed 's/^/  /'
  fi
}

case "${1:-status}" in
  env)    step_env ;;
  build)  step_build ;;
  train)  step_train ;;
  eval)   step_eval ;;
  status) step_status ;;
  *)
    sed -n '2,25p' "$0" >&2
    exit 2
    ;;
esac
