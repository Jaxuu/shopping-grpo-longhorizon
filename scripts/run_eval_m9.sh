#!/usr/bin/env bash
# 一条命令跑完 M9 诊断：起环境 → 起模型 → 等就绪 → 跑评测 → 出报告。
#
# 为什么要有这个：手工流程要在三个 tmux 会话之间切换，任何一步粘错（比如把注释行
# 一起粘进去、Ctrl+B d 按了两次）都会让人以为"卡住了"。这个脚本用 `tmux new-session -d`
# 后台起服务、自己轮询就绪状态，你只需要看它打印的进度。
#
# 用法（建议自己也放进 tmux，断线不影响）：
#   tmux new -s run
#   bash scripts/run_eval_m9.sh
#   # Ctrl+B 松手，再按 D 脱离
#
# 可调环境变量：
#   MIN_OPENED_BEFORE_BUY=5          诊断开关（0=关闭）
#   LABEL=m9-explore5                输出目录名 outputs/evaluation/<LABEL>
#   MODEL=outputs/models/.../merged  权重目录
#   GPU_MEM_UTIL=0.9                 vLLM 显存占比
#   SKIP_PREFLIGHT=1                 跳过自检
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${LABEL:-m9-explore5}"
MIN_OPENED_BEFORE_BUY="${MIN_OPENED_BEFORE_BUY:-5}"
MODEL="${MODEL:-$ROOT/outputs/models/sft-b-curriculum/stage-c/merged}"
ENV_SESSION="${ENV_SESSION:-env}"
VLLM_SESSION="${VLLM_SESSION:-vllm}"
WAIT_ENV_SECONDS="${WAIT_ENV_SECONDS:-180}"
WAIT_MODEL_SECONDS="${WAIT_MODEL_SECONDS:-2400}"
OUT_DIR="$ROOT/outputs/evaluation/$LABEL"
LOG_DIR="$ROOT/outputs/logs"

# 脚本自己的输出也留一份：会话一旦退出，终端里的报错就看不见了，只能靠这个文件。
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() {
  printf '[%s] 失败：%s\n' "$(date +%H:%M:%S)" "$*" >&2
  printf '[%s] 本次日志已保存到 %s/run.log\n' "$(date +%H:%M:%S)" "$LOG_DIR" >&2
  exit 1
}

mkdir -p "$OUT_DIR"

echo "=================================================="
echo " M9 诊断：买前至少打开 ${MIN_OPENED_BEFORE_BUY} 个候选"
echo " 权重：$MODEL"
echo " 输出：$OUT_DIR"
echo "=================================================="

[[ -f "$MODEL/config.json" ]] || die "找不到权重 $MODEL/config.json，先 ls outputs/models/ 确认真实路径"
command -v tmux >/dev/null 2>&1 || die "找不到 tmux，先 apt install tmux"

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  echo
  bash "$ROOT/scripts/preflight_eval.sh" "$MODEL" || log "自检有 FAIL 项；仍继续尝试启动，若起不来看上面提示"
fi

# ---------- 1. 商店环境（纯 CPU） ----------
echo
if tmux has-session -t "$ENV_SESSION" 2>/dev/null; then
  log "tmux 会话 [$ENV_SESSION] 已存在，跳过启动"
else
  tmux new-session -d -s "$ENV_SESSION" \
    "cd '$ROOT' && SHOPSIM_ENV_SLOTS='${SHOPSIM_ENV_SLOTS:-32}' bash scripts/start_environment.sh 2>&1 | tee '$LOG_DIR/env.log'"
  log "已在 tmux 会话 [$ENV_SESSION] 启动商店环境（日志 $LOG_DIR/env.log）"
fi

log "等待商店环境就绪（最多 ${WAIT_ENV_SECONDS}s）…"
env_ready=0
for ((i = 1; i <= WAIT_ENV_SECONDS; i++)); do
  code=$(curl -s -o /dev/null -m 2 -w '%{http_code}' http://127.0.0.1:5700/ 2>/dev/null || echo 000)
  if [[ "$code" != "000" ]]; then
    log "商店环境就绪（HTTP $code；404 属正常）"
    env_ready=1
    break
  fi
  # 会话提前消失 = 进程自己退了，不用等满
  if ! tmux has-session -t "$ENV_SESSION" 2>/dev/null; then
    echo "--- env.log 末尾 ---"
    tail -20 "$LOG_DIR/env.log" 2>/dev/null
    die "商店环境进程退出（tmux 会话 [$ENV_SESSION] 已消失）。若上面没有报错就突然结束，多半是内存不足被系统杀掉：用 free -g 看内存，再试 SHOPSIM_ENV_SLOTS=8"
  fi
  sleep 1
done
(( env_ready )) || die "商店环境 ${WAIT_ENV_SECONDS}s 内没起来，看 $LOG_DIR/env.log"

# ---------- 2. 模型服务（显卡） ----------
if tmux has-session -t "$VLLM_SESSION" 2>/dev/null; then
  log "tmux 会话 [$VLLM_SESSION] 已存在，跳过启动"
else
  tmux new-session -d -s "$VLLM_SESSION" \
    "cd '$ROOT' && GPU_MEM_UTIL='${GPU_MEM_UTIL:-0.9}' bash scripts/serve_model.sh '$MODEL' 2>&1 | tee '$LOG_DIR/vllm.log'"
  log "已在 tmux 会话 [$VLLM_SESSION] 启动 vLLM（日志 $LOG_DIR/vllm.log）"
fi

log "等待模型服务就绪（最多 $((WAIT_MODEL_SECONDS / 60)) 分钟；首次加载 20-25 分钟，显存先涨、util 为 0 是正常的）"
model_ready=0
for ((i = 1; i <= WAIT_MODEL_SECONDS; i += 10)); do
  body=$(curl -s -m 5 http://127.0.0.1:8000/v1/models 2>/dev/null || true)
  if [[ "$body" == *'"id"'* ]]; then
    log "模型服务就绪：$(printf '%s' "$body" | tr -d '\n' | cut -c1-100)"
    model_ready=1
    break
  fi
  if (( i % 60 == 0 )); then
    log "  …已等 $((i / 60)) 分钟，vLLM 最后一行日志：$(tail -1 "$LOG_DIR/vllm.log" 2>/dev/null | cut -c1-100)"
  fi
  sleep 10
done
(( model_ready )) || die "模型服务 $((WAIT_MODEL_SECONDS / 60)) 分钟内没就绪，看 $LOG_DIR/vllm.log（常见原因：显存不足→GPU_MEM_UTIL=0.85 重跑）"

# ---------- 3. 评测 ----------
echo
log "开始评测：${LABEL}，MIN_OPENED_BEFORE_BUY=${MIN_OPENED_BEFORE_BUY}"
log "进度看这个数字（0→200）：wc -l $OUT_DIR/trajectories.jsonl"
echo

MIN_OPENED_BEFORE_BUY="$MIN_OPENED_BEFORE_BUY" \
EVAL_OUTPUT_DIR="$OUT_DIR" \
bash "$ROOT/scripts/evaluate.sh" "$LABEL" || die "评测脚本返回非零，检查上面输出"

# ---------- 4. 报告 ----------
echo
log "生成分析报告…"
"$ROOT/.venv/bin/python" "$ROOT/scripts/analyze_candidate_exploration.py" \
  "$OUT_DIR/trajectories.jsonl" \
  --out "$OUT_DIR/candidate_exploration.md" --examples 20 || log "报告生成失败，但轨迹已保留"

echo
echo "=================================================="
echo " 完成。看这两处："
echo "   摘要：$OUT_DIR/summary.json"
echo "   解剖：$OUT_DIR/candidate_exploration.md"
echo "=================================================="
