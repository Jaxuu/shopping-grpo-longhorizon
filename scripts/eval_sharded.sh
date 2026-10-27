#!/usr/bin/env bash
# 用**本地 vLLM** 跑 Final-200：切片 N 并发 → 合并 → 报告 → 与基线逐题对比。
#
# 与 eval_teacher_final200.sh 的区别只有一处：那个跑远端老师（需要凭据、
# context-window 必须关），这个跑本地权重（不需要凭据、上下文窗口 24576）。
# 两者共用同一套切片/合并脚本，所以口径一致。
#
# 典型用途：同一份权重、只改系统提示词的 A/B。
#   LABEL=prompt-query-attrs \
#   SYSTEM_PROMPT_FILE=configs/prompts/query-attributes.txt \
#   bash scripts/eval_sharded.sh
#
# 常用覆盖：
#   SHARDS=8 bash scripts/eval_sharded.sh
#   BASELINE_SUMMARY=outputs/evaluation/grpo-m10h/summary.json bash scripts/eval_sharded.sh
#   FRESH=1 bash scripts/eval_sharded.sh          # 丢弃旧分片全部重跑
#   LLM_BASE_URL=http://127.0.0.1:8001/v1 bash scripts/eval_sharded.sh
#
# 中途失败不用重头跑：已完成的轨迹留在 outputs/evaluation/<LABEL>/shards/ 里，
# 重跑本脚本会跳过已完成的题。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
SHARDS="${SHARDS:-8}"
BENCHMARK="${BENCHMARK:-$ROOT/data/evaluation/tasks.jsonl}"
SHOPSIM_BASE_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-shopping-agent}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-24576}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-}"
# 采样温度。评测一律 0.0（与历史 63.5%/63.0% 同口径）；只有「给训练池打难度标签」
# 那种需要多条不同轨迹的场景才调高，此时分数**没有可比性**，只是分桶用的原始样本。
TEMPERATURE="${TEMPERATURE:-0.0}"

# LABEL 决定产物目录；用提示词变体时不给 LABEL，就按文件名自动起名，
# 避免手滑把变体写进冠军目录、把 63.5% 覆盖掉。
if [[ -z "${LABEL:-}" ]]; then
  if [[ -n "$SYSTEM_PROMPT_FILE" ]]; then
    LABEL="prompt-$(basename "$SYSTEM_PROMPT_FILE" .txt)"
  else
    LABEL="eval-$(date +%m%d-%H%M)"
  fi
fi
OUT="$ROOT/outputs/evaluation/$LABEL"
SHARD_DIR="$OUT/shards"

fail() { printf '\n[FAIL] %s\n' "$*" >&2; exit 1; }

echo "== 0/6 预检 =="
[[ -x "$PY" ]] || fail "缺少 $PY：先跑 bash scripts/setup.sh"
[[ -f "$BENCHMARK" ]] || fail "找不到评测集 $BENCHMARK"
[[ "$SHARDS" =~ ^[0-9]+$ ]] || fail "SHARDS 必须是整数，当前 '$SHARDS'"
env_code=$(curl -s -o /dev/null -m 3 -w '%{http_code}' "$SHOPSIM_BASE_URL/" 2>/dev/null)
env_code=${env_code:-000}
[[ "$env_code" != "000" ]] \
  || fail "商店环境未启动（$SHOPSIM_BASE_URL）。另开一个终端跑：bash scripts/start_environment.sh"
printf '  商店环境在线（%s -> HTTP %s；404 是正常信号）\n' "$SHOPSIM_BASE_URL" "$env_code"
if ! curl -sf -m 5 "$LLM_BASE_URL/models" >/dev/null 2>&1; then
  fail "模型服务没就绪（$LLM_BASE_URL/models 不通）。
       先起 vLLM：MODEL=outputs/models/<权重目录> bash scripts/serve_model.sh
       首次加载权重需要 20-25 分钟，期间端口不通是正常的。"
fi
printf '  模型服务在线：%s\n' "$LLM_BASE_URL"
if [[ -n "$SYSTEM_PROMPT_FILE" ]]; then
  [[ -f "$SYSTEM_PROMPT_FILE" ]] || fail "找不到提示词文件 $SYSTEM_PROMPT_FILE"
  printf '  系统提示词：%s\n' "$SYSTEM_PROMPT_FILE"
  # printf 会对格式串做 % 展开，裸百分号必须写成 %%。
  printf '  ⚠ 换提示词 = 换 agent 配置。本轮分数**不能**与 63.5%%/62.0%% 直接比较，\n'
  printf '    只能与同一提示词下的对照比。protocol 里会记指纹。\n'
else
  printf '  系统提示词：冻结版 SYSTEM_PROMPT（分数与历史可比）\n'
fi
printf '  采样温度：%s\n' "$TEMPERATURE"
if [[ "$TEMPERATURE" != "0.0" ]]; then
  printf '  ⚠ 温度非 0 ⇒ 本轮**不是评测**，只是给训练池打难度标签的采样。\n'
  printf '    分数不可与 63.5%%/63.0%% 比较，也不要写进冠军记录。\n'
fi

echo
echo "== 1/6 切片（$SHARDS 份） =="
mkdir -p "$OUT"
if [[ "${FRESH:-0}" == "1" ]]; then
  echo "  FRESH=1：清空旧分片"
  rm -rf "$SHARD_DIR"
fi
"$PY" scripts/split_eval_shards.py \
  --benchmark "$BENCHMARK" \
  --shard-dir "$SHARD_DIR" \
  --shards "$SHARDS" || fail "切片失败"

echo
echo "== 2/6 开跑（$SHARDS 并发，每题最多 35 步） =="
declare -a pids=()
for ((i = 0; i < SHARDS; i++)); do
  shard="$SHARD_DIR/$(printf 'shard-%02d' "$i")"
  args=(
    --benchmark "$shard/tasks.jsonl"
    --output "$shard/trajectories.jsonl"
    --summary "$shard/summary.json"
    --base-url "$SHOPSIM_BASE_URL"
    --model "$SERVED_MODEL_NAME"
    --llm-base-url "$LLM_BASE_URL"
    --api-key "$LLM_API_KEY"
    --context-window "$CONTEXT_WINDOW"
    --temperature "$TEMPERATURE"
    --top-p 1.0
    --max-steps 35
  )
  [[ -n "$SYSTEM_PROMPT_FILE" ]] && args+=(--system-prompt-file "$SYSTEM_PROMPT_FILE")
  "$PY" scripts/evaluate_shop_benchmark.py "${args[@]}" >"$shard/run.log" 2>&1 &
  pids+=($!)
  printf '  shard-%02d 已启动 pid=%s\n' "$i" "${pids[${#pids[@]}-1]}"
done

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    printf '  shard-%02d 完成\n' "$i"
  else
    printf '  shard-%02d 失败：tail -30 %s\n' "$i" "$SHARD_DIR/$(printf 'shard-%02d' "$i")/run.log" >&2
    failed=$((failed + 1))
  fi
done
printf '  失败分片 %d / %d\n' "$failed" "$SHARDS"
if ((failed > 0)); then
  printf '  已完成的分片轨迹都留着：修好原因后直接重跑本脚本即可续跑，不用 FRESH=1。\n' >&2
fi

echo
echo "== 3/6 合并 =="
merge_args=(
  --benchmark "$BENCHMARK"
  --shard-dir "$SHARD_DIR"
  --shards "$SHARDS"
  --output-dir "$OUT"
  --model "$SERVED_MODEL_NAME"
  --context-window "$CONTEXT_WINDOW"
)
[[ -n "$SYSTEM_PROMPT_FILE" ]] && merge_args+=(--system-prompt-file "$SYSTEM_PROMPT_FILE")
"$PY" scripts/merge_eval_shards.py "${merge_args[@]}" \
  || fail "合并失败：上面的报错说了原因；补跑缺失分片后重跑本脚本"

echo
echo "== 4/6 报告 =="
"$PY" scripts/build_eval_report.py --run-dir "$OUT" || printf '  报告生成失败，但 summary.json 已写好\n'

echo
echo "== 5/6 与基线逐题对比 =="
base_path=""
run_tasks=$("$PY" - "$OUT/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(int(summary.get("expected_tasks") or 0))
PY
)
# 只有「Final-200 口径 + 温度 0」才是评测。打标采样和小样本跑了也不能跟冠军比，
# 否则会印出「146/401 比 147/200 差 -1 题」这种分母不一致的假结论。
if [[ "$TEMPERATURE" != "0.0" ]]; then
  printf '  采样温度 %s ≠ 0 ⇒ 打标采样，不参与基线对比。\n' "$TEMPERATURE"
elif [[ "$run_tasks" != "200" ]]; then
  printf '  本轮 %s 题，不是 Final-200 口径 ⇒ 不参与基线对比。\n' "$run_tasks"
elif [[ -z "$BASELINE_SUMMARY" ]]; then
  BASELINE_SUMMARY=$("$PY" - "$ROOT" "$LABEL" <<'PY'
import json
import sys
from pathlib import Path

root, skip = Path(sys.argv[1]), sys.argv[2]
best = None
for path in sorted((root / "outputs" / "evaluation").glob("*/summary.json")):
    if path.parent.name == skip:
        continue
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if int(summary.get("expected_tasks") or 0) != 200:
        continue
    successes = int(summary.get("strict_successes") or 0)
    if best is None or successes > best[0]:
        best = (successes, str(path))
print(best[1] if best else "")
PY
)
fi
if [[ "$TEMPERATURE" != "0.0" || "$run_tasks" != "200" ]]; then
  :  # 上面已经说明跳过，这里什么都不做
elif [[ -z "$BASELINE_SUMMARY" ]]; then
  printf '  没找到可对比的 200 题 summary。用 BASELINE_SUMMARY=outputs/evaluation/<目录>/summary.json 指定。\n'
else
  base_path="$BASELINE_SUMMARY"
  [[ -f "$base_path" ]] || base_path="$ROOT/$BASELINE_SUMMARY"
  printf '  对比对象：%s\n' "$base_path"
  "$PY" scripts/compare_teacher_vs_ours.py \
    --teacher-summary "$OUT/summary.json" \
    --ours-summary "$base_path" \
    --teacher-name "$LABEL" \
    --ours-name "基线" \
    --output-dir "$OUT" || printf '  逐题对比失败（通常是基线不是 200 题口径）\n'
fi

echo
echo "== 6/6 判读 =="
"$PY" - "$OUT/summary.json" "$base_path" <<'PY'
import json
import sys
from pathlib import Path

run = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
base = None
try:
    base = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
except Exception:
    pass
got = int(run["strict_successes"])
total = int(run.get("expected_tasks") or 0)
rate = float(run.get("strict_success_rate") or 0.0)
print(f"  本轮 {got}/{total} = {rate * 100:.1f}%")
if total != 200:
    print(f"  本轮 {total} 题，不是 Final-200 口径 ⇒ 不做基线比较（打标/小样本不参与冠军判定）。")
elif base and int(base.get("expected_tasks") or 0) == 200:
    was = int(base["strict_successes"])
    print(f"  基线 {was}/200 = {was / 200 * 100:.1f}%   差 {got - was:+d} 题")
    print(f"  二项噪声 σ ≈ 6.9 题：|差| < 7 题时只能说「同一水平」，不能说「涨了」")
PY

echo
echo "产物："
echo "  $OUT/summary.json            本轮成绩"
echo "  $OUT/report.html             逐题报告"
echo "  $OUT/teacher-vs-ours.md      与基线的逐题差集"
