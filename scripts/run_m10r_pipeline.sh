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
#   bash scripts/run_m10r_pipeline.sh train    # 2. 训练（要显卡，约 3.5–4 小时；被关机打断后再执行一次即续跑）
#   bash scripts/run_m10r_pipeline.sh eval     # 3. Final-200（自带商店环境 + vLLM，40–60 分钟）
#   bash scripts/run_m10r_pipeline.sh report   # 4. 分数 + 探索率（M10-R 的两条验收判据）
#   bash scripts/run_m10r_pipeline.sh status   # 随时看：会话活着吗、跑到哪、显存多少
#
# 会话名：m10r-env / m10r-build / m10r-train / m10r-eval
#   回去看：tmux attach -t m10r-train    脱离：Ctrl+B 松手再按 D
#
# 可覆盖的环境变量
#   KEEP_TEMPLATE=700      模板保留条数（建池）
#   LABEL=m10r-final200    评测输出目录名
#   MODEL=...              评测权重（默认 outputs/models/sft-m10r/stage-b/merged）
#   SAVE_STEPS=25          训练每 25 步存一次 checkpoint（定时关机时的中断粒度）
#   AUTO_EVAL=1            训练成功后自动接着跑 Final-200（无人值守用这个）
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
SAVE_STEPS="${SAVE_STEPS:-25}"
# 空 = 用 run_sft_curriculum 的默认值 Qwen/Qwen3.5-2B（走 HF 缓存）。
# 国内机器连不上 hub 时 transformers 会卡在 "Retrying [Retry 1/5]"，
# 这时传本地目录（如 /root/rivermind-data/hf/Qwen3.5-2B）彻底绕开网络。
BASE_MODEL="${BASE_MODEL:-}"

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
    if session_busy "${name}"; then
      echo "[skip] 会话 [${name}] 正在跑：tmux attach -t ${name}"
      return 0
    fi
    # 上一轮跑完了但会话留着（body 末尾的 exec bash），关掉重开——
    # 否则续跑永远起不来，会一直显示"已在跑"。
    tmux kill-session -t "${name}"
    echo "[reuse] 会话 [${name}] 是上一轮跑完留下的，已关闭重开"
  fi
  tmux new-session -d -s "${name}" "${body}"
  echo "[run] 会话 [${name}] 已启动 · 日志 ${LOG_DIR}/${name}.log"
  echo "      回去看：tmux attach -t ${name}（脱离：Ctrl+B 松手再按 D）"
}

# 会话「在跑」还是「跑完停在 shell」：看 pane 的 shell 还有没有子进程。
# body 末尾是 exec bash，所以跑完之后 shell 是孤零零一个，子进程数为 0。
session_busy() {
  local pid
  pid="$(tmux list-panes -t "$1" -F '#{pane_pid}' 2>/dev/null | head -1)"
  [[ -z "${pid}" ]] && return 1
  if command -v pgrep >/dev/null 2>&1; then
    pgrep -P "${pid}" >/dev/null 2>&1
  else
    ps -eo ppid= 2>/dev/null | tr -d ' ' | grep -qx "${pid}"
  fi
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
stage_done() {
  [[ -d "${MODEL_DIR}/stage-$1/merged" ]] \
    && [[ -n "$(ls -A "${MODEL_DIR}/stage-$1/merged" 2>/dev/null)" ]]
}

latest_checkpoint() {
  ls -1d "${MODEL_DIR}/stage-$1/adapter"/checkpoint-* 2>/dev/null | sort -V | tail -1
}

# 断点续跑的判据只有两条，全在磁盘上：合并好的阶段算「做完」，adapter 下的
# checkpoint-NNN 算「做到第 N 步」。所以 train 可以反复执行——它自己决定从哪接。
plan_resume() {
  local start="" resume=""
  if stage_done a && stage_done b; then
    echo "两段都已完成，没有可续的：${MODEL_DIR}/stage-b/merged" >&2
    return 1
  fi
  if stage_done a; then start="b"; fi
  if [[ -z "${start}" ]]; then
    start="a"
    resume="$(latest_checkpoint a || true)"
  else
    resume="$(latest_checkpoint b || true)"
  fi
  printf '%s\n%s\n' "${start}" "${resume}"
}

preflight_train() {
  local problems=0
  [[ -f "${CAPPED}/all.jsonl" ]] || { echo "缺 ${CAPPED}/all.jsonl（先跑 build）" >&2; problems=1; }
  [[ -f "${CAPPED}/manifest.json" ]] || { echo "缺 ${CAPPED}/manifest.json（先跑 build）" >&2; problems=1; }
  [[ "${problems}" -eq 0 ]] || return 1

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

  # 磁盘的真实开销：两段 merged 各 4–5 GB（2B bf16 全量权重），LoRA checkpoint
  # 只有几十 MB×3，日志几十 MB ⇒ 12 GB 是能跑完的下限，留点余量到 15。
  local free
  free="$(free_gb "${ROOT}/outputs")"
  if [[ -n "${free}" ]] && (( free < 12 )); then
    echo "outputs 所在盘只剩 ${free} GB；两段 merged 要 9–10 GB，至少清到 15 GB。" >&2
    return 1
  fi

  # liger-kernel 不是可选优化。全词表 CE 在 ~21k token 上要 20 GiB 瞬时 fp32 张量
  # （M5 stage-C 实测 "Tried to allocate 15.50 GiB" 直接 OOM），而 M10-R 的 A 段
  # 全是探索轨迹（翻页 → 每页渲染 20 个商品），比 stage-C 只长不短。
  # 它由 sft-accelerated extra 提供，setup.sh 漏写这个 extra 时 uv sync 会把它卸掉。
  if ! "${PY}" -c 'import liger_kernel' >/dev/null 2>&1; then
    cat >&2 <<EOF
缺 liger-kernel：48 G 卡不加它，训练会在前几步 OOM（不是慢，是崩）。
先重跑一次依赖（2–3 分钟）：
  bash scripts/run_m10r_pipeline.sh env
EOF
    return 1
  fi
  echo "[ok] liger-kernel 可用"

  # 给了本地模型目录就查一眼，别等 transformers 报一个难读的错。
  if [[ -n "${BASE_MODEL}" ]]; then
    if [[ ! -f "${BASE_MODEL}/config.json" ]]; then
      echo "BASE_MODEL=${BASE_MODEL} 下没有 config.json；它应该是本地模型目录，不是 hub id。" >&2
      return 1
    fi
    echo "[ok] 基座模型：${BASE_MODEL}（本地目录，不走网络）"
  fi
  echo "[ok] 前置检查通过（显存 ${used:-?} MiB 已用，outputs 余量 ${free:-?} GB）"
}

step_train() {
  [[ -x "${PY}" ]] || die "找不到 Python：${PY}"
  local plan start resume
  if [[ "${PIPELINE_DRY:-0}" == "1" ]]; then
    echo "[dry] 跳过显卡/磁盘检查，但仍按盘上的产物算续跑点（PIPELINE_DRY=1）"
    plan="$(plan_resume)" || plan=$'a\n'
    start="$(sed -n 1p <<<"${plan}")"
    resume="$(sed -n 2p <<<"${plan}")"
  else
    preflight_train || die "前置检查没过，见上面的提示"
    plan="$(plan_resume)" || die "没有可续的训练"
    start="$(sed -n 1p <<<"${plan}")"
    resume="$(sed -n 2p <<<"${plan}")"
    if [[ "${start}" == "b" ]]; then
      echo "[resume] A 段已完成（stage-a/merged 在盘上），从 B 段开始。"
    fi
    if [[ -n "${resume}" ]]; then
      echo "[resume] 找到中途存档：${resume}"
    else
      echo "[start] 这一段从头开始（每 ${SAVE_STEPS} 步存一次 checkpoint）"
    fi
  fi
  local extra=""
  [[ "${start}" != "a" ]] && extra="--start-stage ${start}"
  [[ -n "${resume}" ]] && extra="${extra} --resume-from-checkpoint '${resume}'"
  [[ -n "${BASE_MODEL}" ]] && extra="${extra} --base-model '${BASE_MODEL}'"
  # AUTO_EVAL=1：训练跑完接着起评测。串在同一个 tmux 里用 &&，训练失败就不会去测
  # 一个半成品权重（那样只会浪费时间，还会让人误判成模型变差）。
  local chain=""
  if [[ "${AUTO_EVAL:-0}" == "1" ]]; then
    chain=" && bash scripts/run_m10r_pipeline.sh eval"
  fi
  # expandable_segments 是历史踩坑的固定配置（碎片回收过 4.4 GiB），别省。
  # --liger-kernel 与 M5-B（62% 冠军）同一配方，也是 48 G 卡不 OOM 的前提。
  launch m10r-train \
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     bash scripts/sft_curriculum.sh \
       --source '${CAPPED}/all.jsonl' \
       --manifest '${CAPPED}/manifest.json' \
       --output-root '${MODEL_DIR}' \
       --loss-scope tool-call-only \
       --liger-kernel \
       --save-steps ${SAVE_STEPS} ${extra}${chain}"
  # 数字来自 scripts/estimate_sft_train_time.py（M5 同类 48 GB 卡三点实测外推），
  # 不要写回「1.5 + 2.5 小时」——那是按旧池 1593 行次算的，exploratory 比 strategy 长 1.5 倍。
  echo "[提示] A 段约 2.2–2.7 小时、B 段约 3.4–3.8 小时（实测标定外推）。进度：bash scripts/run_m10r_pipeline.sh status"
  echo "      被关机打断后再执行一次本命令即续跑（已完成的阶段会跳过）。"
  if [[ "${AUTO_EVAL:-0}" == "1" ]]; then
    echo "      训练成功后自动起评测（会话 m10r-eval，日志 outputs/logs/m10r-eval.log）。"
  fi
}

# ---------------------------------------------------------------- eval
# 训练刚结束时显存要过一会儿才真正释放；vLLM 抢不到会直接 OOM 退出。
wait_gpu_free() {
  local waited=0 used=""
  while (( waited < 180 )); do
    used="$(gpu_used_mib)"
    if [[ -z "${used}" ]] || (( used < 2000 )); then
      echo "[ok] 显存已释放（${used:-?} MiB）"
      return 0
    fi
    sleep 10
    waited=$(( waited + 10 ))
  done
  echo "显卡还被占 ${used} MiB（等了 180 s）。nvidia-smi 看是谁没退干净。" >&2
  return 1
}

# run_eval_m9.sh 遇到已存在的 [vllm] 会话会「跳过启动」，于是拿旧权重测新模型——
# 这种错不报错、只出错误分数，是最贵的一类 bug。这里先比对命令行里的权重路径。
preflight_eval() {
  [[ -f "${MODEL}/config.json" ]] || {
    echo "找不到权重：${MODEL}/config.json（训练没跑完？或换 MODEL=...）" >&2
    return 1
  }
  if tmux has-session -t vllm 2>/dev/null; then
    local running
    running="$(ps -eo args 2>/dev/null | grep -F 'vllm' | grep -F ' serve ' | head -1)"
    if [[ -n "${running}" && "${running}" != *"${MODEL}"* ]]; then
      cat >&2 <<EOF
已有 [vllm] 会话在服务**别的**权重，评测会静默测错模型：
  正在服务：${running}
  本次要测：${MODEL}
先换掉它：tmux kill-session -t vllm
EOF
      return 1
    fi
    echo "[ok] 已有 vLLM 在服务本次权重，跳过显存等待"
    return 0
  fi
  wait_gpu_free
}

step_eval() {
  if [[ "${PIPELINE_DRY:-0}" != "1" ]]; then
    preflight_eval || die "评测前置检查没过，见上面的提示"
  fi
  # MIN_OPENED_BEFORE_BUY 默认是 3（强制多开的诊断开关），Final-200 必须关掉，
  # 否则测出来的不是真实水平。
  # 评测后用 ';' 接 report：即使评测半途失败，也要把已有的产物摊开给人看，
  # 而不是留一个空目录让人猜。report 自己会判断缺什么。
  launch m10r-eval \
    "MIN_OPENED_BEFORE_BUY=0 MODEL='${MODEL}' LABEL='${LABEL}' \
     bash scripts/run_eval_m9.sh; bash scripts/run_m10r_pipeline.sh report"
  echo "[提示] 自带商店环境（tmux env）+ vLLM（tmux vllm）；vLLM 首次加载 20–25 分钟。"
  echo "       评测完会自动打印分数与探索率（M10-R 的验收判据）。"
}

# ---------------------------------------------------------------- report
# M10-R 的验收不是「分数涨」一条，而是「探索率抬起来 + 分数没掉」。两条都要看：
#   探索率没抬 ⇒ SFT 没起作用（只是没伤到原有能力）
#   分数掉了   ⇒ 抬过头或示范质量差
step_report() {
  local eval_dir="${ROOT}/outputs/evaluation/${LABEL}"
  local summary="${eval_dir}/summary.json"
  if [[ ! -f "${summary}" ]]; then
    echo "没有 ${summary}：评测还没跑完或失败了。" >&2
    echo "看：tail -30 outputs/logs/m10r-eval.log" >&2
    return 1
  fi
  "${PY}" - "${summary}" <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
done = int(data.get("strict_successes") or 0)
total = int(data.get("expected_tasks") or 0)
rate = float(data.get("strict_success_rate") or 0.0)
baseline = 0.62
print("== Final-200")
print(f"严格成功 {done}/{total} = {rate * 100:.1f}%")
print(f"基线（M5-B / M6 champion）62.0% = 124/200")
delta = (rate - baseline) * 100
print(f"差值 {delta:+.1f} 个百分点")
PY
  echo
  # 对照池可能不存在（比如本地没建池），缺了就只报评测集，不要让报告整个失败。
  local inputs=("${eval_dir}/trajectories.jsonl")
  [[ -f "${CAPPED}/all.jsonl" ]] && inputs+=("${CAPPED}/all.jsonl")
  [[ -f "${ROOT}/data/sft_pure_v4/all.jsonl" ]] && inputs+=("${ROOT}/data/sft_pure_v4/all.jsonl")
  "${PY}" scripts/report_behavior_rate.py "${inputs[@]}" --expect 200
}

# ---------------------------------------------------------------- status
step_status() {
  echo "== 会话"
  local any=0
  for name in m10r-env m10r-build m10r-train m10r-eval env vllm; do
    if tmux has-session -t "${name}" 2>/dev/null; then
      if session_busy "${name}"; then
        echo "  [跑着] ${name}"
        any=1
      else
        echo "  [已结束] ${name}（会话留着看输出；关掉：tmux kill-session -t ${name}）"
      fi
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
    "${PY}" scripts/show_sft_manifest.py "${CAPPED}/manifest.json" 2>/dev/null | sed 's/^/  /'
  fi

  echo
  echo "== 训练/评测进度（日志末尾）"
  for name in m10r-env m10r-build m10r-train m10r-eval; do
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
    for stage in a b; do
      local ckpt
      ckpt="$(latest_checkpoint "${stage}")"
      [[ -n "${ckpt}" ]] && echo "  续跑点（stage-${stage}）：$(basename "${ckpt}")"
    done
  fi
}

case "${1:-status}" in
  env)    step_env ;;
  build)  step_build ;;
  train)  step_train ;;
  eval)   step_eval ;;
  report) step_report ;;
  status) step_status ;;
  *)
    sed -n '2,25p' "$0" >&2
    exit 2
    ;;
esac
