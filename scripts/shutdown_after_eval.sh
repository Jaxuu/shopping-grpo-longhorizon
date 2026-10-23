#!/usr/bin/env bash
# 等 M10-R 全流程跑完（训练 → 评测 → 结果落盘），然后关机。
#
# 为什么需要它：训练 + 评测要 7–8 小时，人不可能守着。tmux 会话能扛住 SSH 断开，
# 但扛不住「跑完了还在计费」。这个脚本只做一件事——把「什么时候算真的跑完」写死：
#   1. m10r-train 会话空闲（不是"会话存在"，见 run_m10r_pipeline.sh 的 session_busy）
#   2. m10r-eval 会话出现并空闲
#   3. summary.json 能读出 expected_tasks
#   4. sync 落盘
#   5. 关机
#
# 用法
# ----
#   bash scripts/shutdown_after_eval.sh              # 起一个 tmux 会话盯着，然后关机
#   bash scripts/shutdown_after_eval.sh --dry-run    # 只打印它会做什么
#   bash scripts/shutdown_after_eval.sh --no-shutdown-on-failure
#                                                    # 训练失败时不关机（留着排查）
#   bash scripts/shutdown_after_eval.sh --kill-init   # 容器里没有 systemd 时，
#                                                    # 终止 1 号进程让容器退出
#
# 注意：容器里 shutdown 常常"命令存在但不生效"（Failed to connect to bus）。
# 脚本会先探测 systemd（/run/systemd/system 存在且 1 号进程是 systemd），
# 探测不到就打印清楚并退出——那种情况只能去平台 UI 关机。
# --kill-init 是给容器用的兜底：它能让实例停止运行，但**平台是否停止计费
# 需要在控制台确认一次**（有些平台"已停止"的容器仍然收存储费）。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PY="${PY:-${ROOT}/.venv/bin/python}"
LOG_DIR="${ROOT}/outputs/logs"
LABEL="${LABEL:-m10r-final200}"
SUMMARY="${ROOT}/outputs/evaluation/${LABEL}/summary.json"
# 阶段名由 manifest 决定（两段是 b，单阶段是 full），不能写死——写死的话单阶段
# 跑完会被这里判成「训练失败」，白等一场还关不了机。
STAGE_PLAN="${STAGE_PLAN:-reverse}"
FINAL_STAGE="${FINAL_STAGE:-}"
MANIFEST="${ROOT}/data/sft_m10r/capped/manifest.json"
if [[ -z "${FINAL_STAGE}" && -f "${MANIFEST}" ]]; then
  FINAL_STAGE="$("${PY}" -c 'import json,sys;print(list(json.load(open(sys.argv[1],encoding="utf-8"))["stages"])[-1])' \
    "${MANIFEST}" 2>/dev/null || true)"
fi
[[ -n "${FINAL_STAGE}" ]] || FINAL_STAGE="$([[ "${STAGE_PLAN}" == "single" ]] && echo full || echo b)"
FINAL_MODEL="${ROOT}/outputs/models/sft-m10r/stage-${FINAL_STAGE}/merged/config.json"
LOG="${LOG_DIR}/m10r-watch.log"
SESSION="m10r-watch"

TRAIN_MAX_SECONDS="${TRAIN_MAX_SECONDS:-43200}"   # 12 h
EVAL_MAX_SECONDS="${EVAL_MAX_SECONDS:-10800}"     # 3 h
EVAL_GRACE_SECONDS="${EVAL_GRACE_SECONDS:-900}"   # 训练结束后等评测起来的上限 15 min
SHUTDOWN_ON_FAILURE="${SHUTDOWN_ON_FAILURE:-1}"
# 容器里没有 systemd，shutdown 是空操作。--kill-init 会直接终止 1 号进程让容器退出。
# ⚠️ 这只保证"实例停止运行"，**不保证平台停止计费**——要在控制台确认一次。
KILL_INIT=0
DRY_RUN=0
WATCH_INTERNAL=0

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --no-shutdown-on-failure) SHUTDOWN_ON_FAILURE=0 ;;
    --kill-init) KILL_INIT=1 ;;
    --watch-internal) WATCH_INTERNAL=1 ;;
    *) echo "未知参数：${arg}" >&2; exit 2 ;;
  esac
done

mkdir -p "${LOG_DIR}"
log() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*"; }

# 与 run_m10r_pipeline.sh 同一判据：pane 的 shell 还有没有子进程。
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

session_exists() { tmux has-session -t "$1" 2>/dev/null; }

# 轮询到会话空闲（跑完但会话留着也算空闲）或消失；超时返回 1。
wait_idle() {
  local name="$1" limit="$2" waited=0
  while (( waited < limit )); do
    if ! session_exists "${name}"; then
      log "会话 [${name}] 已不存在（视为结束）"
      return 0
    fi
    if ! session_busy "${name}"; then
      log "会话 [${name}] 已结束"
      return 0
    fi
    sleep 30
    waited=$(( waited + 30 ))
    if (( waited % 600 == 0 )); then
      log "…[${name}] 仍在跑（已等 $(( waited / 60 )) 分钟）"
    fi
  done
  log "会话 [${name}] 等了 $(( limit / 60 )) 分钟仍未结束"
  return 1
}

detect_poweroff() {
  if [[ -d /run/systemd/system ]] && command -v systemctl >/dev/null 2>&1; then
    echo "systemctl poweroff"
    return 0
  fi
  if [[ "$(ps -p 1 -o comm= 2>/dev/null)" == "systemd" ]] && command -v shutdown >/dev/null 2>&1; then
    echo "shutdown -h now"
    return 0
  fi
  return 1
}

watch() {
  log "开始盯：train 最多 $(( TRAIN_MAX_SECONDS / 3600 )) h，eval 最多 $(( EVAL_MAX_SECONDS / 3600 )) h"
  log "评测结果：${SUMMARY}"

  # ---- 1. 等训练
  if ! session_exists m10r-train; then
    log "找不到 [m10r-train] 会话；如果训练还没启动，先跑 run_m10r_pipeline.sh train。这里退出，不关机。"
    return 2
  fi
  wait_idle m10r-train "${TRAIN_MAX_SECONDS}" || return 2

  local train_ok=1
  [[ -f "${FINAL_MODEL}" ]] || train_ok=0

  # ---- 2. 训练失败：按开关决定是否关机
  if (( ! train_ok )); then
    log "训练没有产出 stage-${FINAL_STAGE}/merged —— 视为失败。"
    if (( SHUTDOWN_ON_FAILURE )); then
      log "按默认策略仍然关机（省电费）；要保留机器排查，用 --no-shutdown-on-failure。"
    else
      log "已指定失败时不关机，退出。"
      return 2
    fi
  else
    log "训练产物在：${FINAL_MODEL}"
    # ---- 3. 等评测起来（AUTO_EVAL=1 才会自动起）
    local waited=0
    while (( waited < EVAL_GRACE_SECONDS )) && ! session_exists m10r-eval; do
      sleep 30
      waited=$(( waited + 30 ))
    done
    if ! session_exists m10r-eval; then
      log "训练结束后 $(( EVAL_GRACE_SECONDS / 60 )) 分钟内没有出现 [m10r-eval]；"
      log "说明没带 AUTO_EVAL=1，或评测前置检查没过。不关机，退出。"
      return 2
    fi
    # ---- 4. 等评测
    wait_idle m10r-eval "${EVAL_MAX_SECONDS}" || log "评测超时，仍然继续关机流程"
  fi

  # ---- 5. 校验结果（失败也照样关，但日志写清楚）
  if [[ -f "${SUMMARY}" ]]; then
    if "${PY}" - "${SUMMARY}" <<'PY' >/dev/null 2>&1
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
assert data.get("expected_tasks") == 200
PY
    then
      log "summary.json 可读，200 题齐全。"
    else
      log "summary.json 存在但内容不完整（expected_tasks 不是 200），先别信它的分数。"
    fi
  else
    log "没有 summary.json —— 评测没跑完或失败。"
  fi

  # ---- 6. 落盘再关机：不 sync 有可能丢掉最后写入的文件
  sync
  log "已 sync。"

  local cmd
  if ! cmd="$(detect_poweroff)"; then
    if (( KILL_INIT )); then
      local pid1
      pid1="$(ps -p 1 -o comm= 2>/dev/null)"
      log "容器环境（1 号进程 ${pid1}）：没有 systemd，改用终止 1 号进程让容器退出。"
      log "⚠️ 这只保证实例停止运行，平台是否停止计费要在控制台确认一次。"
      kill -TERM 1 2>/dev/null || true
      local i
      for i in $(seq 1 20); do sleep 1; done
      log "TERM 之后还活着，改用 KILL。"
      kill -KILL 1 2>/dev/null || true
      sleep 5
      log "如果这条日志还能被看到，说明容器没被终止——只能去平台 UI 关机。"
      return 3
    fi
    log "这台机器没有 systemd（1 号进程是 $(ps -p 1 -o comm= 2>/dev/null)），容器内关不掉。"
    log "请到云平台控制台手动关机，否则会一直计费。"
    log "（想让它自己退出容器：加 --kill-init；但计费是否停止需在控制台确认。）"
    return 3
  fi

  log "执行关机：${cmd}"
  if (( DRY_RUN )); then
    log "[dry-run] 不真的关机"
    return 0
  fi
  eval "${cmd}"
}

if (( WATCH_INTERNAL )); then
  # tmux 会话里执行这一支：真正开始盯。
  watch
  exit $?
fi

if (( DRY_RUN )); then
  echo "[dry-run] 会话名 ${SESSION}，日志 ${LOG}"
  echo "[dry-run] 轮询 train → eval → 校验 → sync → 关机"
  # 这一行是排错的关键：阶段名写错时，跑完 8 小时会被判成「训练失败」。
  echo "[dry-run] 认定训练完成的标志：${FINAL_MODEL}（阶段 ${FINAL_STAGE}）"
  cmd="$(detect_poweroff)" || cmd="(探测不到 systemd：默认只打印提示；加 --kill-init 则终止 1 号进程)"
  echo "[dry-run] 关机命令：${cmd}"
  exit 0
fi

command -v tmux >/dev/null 2>&1 || { echo "找不到 tmux" >&2; exit 1; }
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[skip] 会话 [${SESSION}] 已在盯：tmux attach -t ${SESSION}"
  exit 0
fi

body="cd '${ROOT}' && bash scripts/shutdown_after_eval.sh --watch-internal \
  2>&1 | tee -a '${LOG}'; echo; echo '[结束] 关机流程已走完；Ctrl+B 松手按 D 脱离'; exec bash"
tmux new-session -d -s "${SESSION}" "${body}"
echo "[run] 会话 [${SESSION}] 已启动 · 日志 ${LOG}"
echo "      回去看：tmux attach -t ${SESSION}（脱离：Ctrl+B 松手再按 D）"
