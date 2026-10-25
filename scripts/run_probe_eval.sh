#!/usr/bin/env bash
# 一条命令跑完一次「探针评测」：自检 → 起环境 → 起模型 → 等就绪 → 跑评测 → 出报告。
#
# 为什么要有这个
# --------------
# 手工流程要在三个 tmux 会话之间切换：环境（:5700）、vLLM（:8000）、评测。
# 任何一步粘错（把注释行一起粘进去、Ctrl+B 按了两次、断线）都会表现为「卡住了」，
# 而 vLLM 首次加载要 20–25 分钟，白等一次就是一小时。
#
# 与 run_eval_m9.sh 的区别：那个是 M9 专用（买前必须打开 k 个候选 + 专用报告）。
# 这个只做「换一份 benchmark 再测一遍」，评测口径与历史完全一致：
# MIN_OPENED_BEFORE_BUY 固定 0，max_steps 由环境侧冻结在 35。
#
# 用法
# ----
#   # 训练集自测（300 道模型自己训练过的题）
#   MODEL=outputs/models/sft-m10r/stage-full/merged \
#   BENCHMARK=data/probes/selftest-300/tasks.jsonl \
#   LABEL=selftest-300 bash scripts/run_probe_eval.sh
#
#   # dev 集基线
#   BENCHMARK=data/evaluation/dev-300/tasks.jsonl LABEL=dev-300 bash scripts/run_probe_eval.sh
#
#   # 服务已经起着，只补一次评测（省掉 25 分钟加载）
#   SKIP_SERVE=1 BENCHMARK=... LABEL=... bash scripts/run_probe_eval.sh
#
# 可调环境变量：MODEL / BENCHMARK / LABEL / GPU_MEM_UTIL / SKIP_PREFLIGHT / SKIP_SERVE
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${LABEL:?用法：BENCHMARK=<benchmark.jsonl> LABEL=<输出名> bash scripts/run_probe_eval.sh}"
BENCHMARK="${BENCHMARK:?缺少 BENCHMARK（要评测的 tasks.jsonl 路径）}"
MODEL="${MODEL:-$ROOT/outputs/models/sft-m10r/stage-full/merged}"
ENV_SESSION="${ENV_SESSION:-env}"
VLLM_SESSION="${VLLM_SESSION:-vllm}"
WAIT_ENV_SECONDS="${WAIT_ENV_SECONDS:-180}"
WAIT_MODEL_SECONDS="${WAIT_MODEL_SECONDS:-2400}"
OUT_DIR="$ROOT/outputs/evaluation/$LABEL"
LOG_DIR="$ROOT/outputs/logs"

mkdir -p "$LOG_DIR" "$OUT_DIR"
exec > >(tee -a "$LOG_DIR/run-${LABEL}.log") 2>&1

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() {
  printf '[%s] 失败：%s\n' "$(date +%H:%M:%S)" "$*" >&2
  printf '[%s] 日志：%s/run-%s.log\n' "$(date +%H:%M:%S)" "$LOG_DIR" "$LABEL" >&2
  exit 1
}

cd "$ROOT"

# 路径都要先验，别等到服务起好 25 分钟后才因为文件不存在失败。
[[ -f "$BENCHMARK" ]] || die "benchmark 不存在：$BENCHMARK"
[[ -f "$MODEL/config.json" ]] || die "权重不存在：$MODEL/config.json（ls outputs/models/ 确认真实路径）"
bench_n=$(grep -c . "$BENCHMARK" 2>/dev/null || echo 0)

echo "=================================================="
echo " 探针评测：$LABEL"
echo " 权重：    $MODEL"
echo " 评测集：  $BENCHMARK（$bench_n 题）"
echo " 输出：    $OUT_DIR"
echo "=================================================="
if (( bench_n > 0 )); then
  # 200 题实测 40–60 分钟（串行），按这个斜率给个预期，避免以为卡死。
  log "预计耗时：模型加载约 20–25 分钟 + 评测约 $(( bench_n * 15 / 60 + 1 )) 分钟（按 200 题 50 分钟折算）"
fi

command -v tmux >/dev/null 2>&1 || die "找不到 tmux，先 apt install tmux"

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  echo
  bash "$ROOT/scripts/preflight_eval.sh" "$MODEL" || log "自检有 FAIL 项；仍继续尝试启动，若起不来看上面提示"
fi

if [[ "${SKIP_SERVE:-0}" == "1" ]]; then
  log "SKIP_SERVE=1：跳过起服务，直接用 :5700 与 :8000 上现有的进程"
else
  # ---------- 1. 商店环境（纯 CPU） ----------
  echo
  # 32 个环境实例每个都加载 23,421 条商品；内存小的机器会被 OOM killer 杀掉，
  # 表现是端口 000 + 显存 1 MiB。按可用内存自动降档。
  SLOTS="${SHOPSIM_ENV_SLOTS:-32}"
  mem_avail_gb=$(free -g 2>/dev/null | awk '/^Mem:/{print $7}')
  if [[ -n "${mem_avail_gb:-}" ]] && (( mem_avail_gb > 0 )); then
    if (( mem_avail_gb < 24 )) && (( SLOTS > 8 )); then
      SLOTS=8
      log "可用内存仅 ${mem_avail_gb} GB，环境实例数自动降到 8"
    elif (( mem_avail_gb < 48 )) && (( SLOTS > 16 )); then
      SLOTS=16
      log "可用内存 ${mem_avail_gb} GB，环境实例数自动降到 16"
    else
      log "可用内存 ${mem_avail_gb} GB，环境实例数 ${SLOTS}"
    fi
  fi

  if tmux has-session -t "$ENV_SESSION" 2>/dev/null; then
    log "tmux 会话 [$ENV_SESSION] 已存在，跳过启动"
  else
    tmux new-session -d -s "$ENV_SESSION" \
      "cd '$ROOT' && SHOPSIM_ENV_SLOTS='$SLOTS' bash scripts/start_environment.sh 2>&1 | tee '$LOG_DIR/env.log'"
    log "已在 tmux [$ENV_SESSION] 启动商店环境（实例数 $SLOTS，日志 $LOG_DIR/env.log）"
  fi

  log "等待商店环境就绪（最多 ${WAIT_ENV_SECONDS}s）…"
  env_ready=0
  for ((i = 1; i <= WAIT_ENV_SECONDS; i++)); do
    # curl 连接失败时 -w 已经打印 000，不能再 `|| echo 000`（会拼成 000000 骗过判断）。
    code=$(curl -s -o /dev/null -m 2 -w '%{http_code}' http://127.0.0.1:5700/ 2>/dev/null)
    [[ -z "$code" ]] && code=000
    if [[ "$code" != "000" ]]; then
      log "商店环境就绪（HTTP $code；404 属正常）"
      env_ready=1
      break
    fi
    if ! tmux has-session -t "$ENV_SESSION" 2>/dev/null; then
      echo "--- env.log 末尾 ---"
      tail -20 "$LOG_DIR/env.log" 2>/dev/null
      die "商店环境进程退出。若没有报错就突然结束，多半是内存不足被系统杀掉：free -g 看看，再试 SHOPSIM_ENV_SLOTS=8"
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
    log "已在 tmux [$VLLM_SESSION] 启动 vLLM（日志 $LOG_DIR/vllm.log）"
  fi

  log "等待模型服务就绪（最多 $((WAIT_MODEL_SECONDS / 60)) 分钟；首次加载 20–25 分钟，显存先涨、util 为 0 是正常的）"
  model_ready=0
  for ((i = 1; i <= WAIT_MODEL_SECONDS; i += 10)); do
    body=$(curl -s -m 5 http://127.0.0.1:8000/v1/models 2>/dev/null || true)
    if [[ "$body" == *'"id"'* ]]; then
      log "模型服务就绪：$(printf '%s' "$body" | tr -d '\n' | cut -c1-100)"
      model_ready=1
      break
    fi
    if (( i % 60 == 0 )); then
      log "  …已等 $((i / 60)) 分钟，vLLM 最后一行：$(tail -1 "$LOG_DIR/vllm.log" 2>/dev/null | cut -c1-100)"
    fi
    if ! tmux has-session -t "$VLLM_SESSION" 2>/dev/null; then
      echo "--- vllm.log 末尾 ---"
      tail -30 "$LOG_DIR/vllm.log" 2>/dev/null
      die "vLLM 进程退出。常见原因：显存不足（试 GPU_MEM_UTIL=0.85）、端口 8000 被占用、权重不完整"
    fi
    sleep 10
  done
  (( model_ready )) || die "模型服务 $((WAIT_MODEL_SECONDS / 60)) 分钟内没就绪，看 $LOG_DIR/vllm.log"
fi

# ---------- 3. 评测 ----------
echo
log "开始评测：$LABEL（$bench_n 题）"
log "进度看这个数字（0→$bench_n）：wc -l $OUT_DIR/trajectories.jsonl"
echo

BENCHMARK="$BENCHMARK" EVAL_OUTPUT_DIR="$OUT_DIR" \
  bash "$ROOT/scripts/evaluate.sh" "$LABEL" || die "评测脚本返回非零，检查上面输出"

echo
echo "=================================================="
echo " 完成：$LABEL"
echo "   摘要：$OUT_DIR/summary.json"
echo "   轨迹：$OUT_DIR/trajectories.jsonl"
if [[ "$BENCHMARK" == *selftest* ]]; then
  echo
  echo " 下一步（判「泛化问题 vs 没学会」）："
  echo "   .venv/bin/python scripts/compare_selftest.py --run-dir $OUT_DIR \\"
  echo "     --probe $(dirname "$BENCHMARK") --eval-run-dir outputs/evaluation/m10r-v2"
fi
echo "=================================================="
