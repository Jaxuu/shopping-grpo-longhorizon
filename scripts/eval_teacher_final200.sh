#!/usr/bin/env bash
# 用老师模型跑 Final-200 Clean：切片 8 并发 → 合并 → 报告 → 与冠军的逐题差集。
#
# 为什么是切片：`evaluate_shop_benchmark.py` 的采集循环是**串行**的（没有 workers），
# 200 题单进程要 1.5-3 小时。切成 8 份各起一个进程，共享同一个商店环境
# （环境实例池 32 个，8 并发安全），大约 20-40 分钟跑完。
#
# 不需要显卡：商店环境是纯 CPU，模型是远端 API。
#
# 用法（凭据二选一，缺了脚本会当场问你）：
#   bash scripts/eval_teacher_final200.sh
#   OPENAI_BASE_URL=... OPENAI_API_KEY=... bash scripts/eval_teacher_final200.sh
#
# 常用覆盖：
#   SHARDS=8 TEACHER_MODEL=deepseek-v4-flash bash scripts/eval_teacher_final200.sh
#   OURS_SUMMARY=outputs/evaluation/<冠军目录>/summary.json bash scripts/eval_teacher_final200.sh
#   TOKENIZER_PATH=outputs/models/<任意完整权重目录> bash scripts/eval_teacher_final200.sh
#   FRESH=1 bash scripts/eval_teacher_final200.sh        # 丢弃旧分片，全部重跑
#   TEACHER_TIMEOUT=300 bash scripts/eval_teacher_final200.sh  # 老师响应慢时放宽单步超时
#
# 中途失败不用重头跑：已完成的轨迹留在 outputs/evaluation/<LABEL>/shards/ 里，
# 重跑本脚本会自动跳过已完成的题。
#
# 口径说明：老师历史成绩 139/200 跑在**旧** Final-200 上，和当前 Final-200 Clean
# 不是同一份卷子。这个脚本产出的数字才是能直接与 63.5% 对比的那个。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
LABEL="${LABEL:-teacher-final200}"
SHARDS="${SHARDS:-8}"
BENCHMARK="${BENCHMARK:-$ROOT/data/evaluation/tasks.jsonl}"
SHOPSIM_BASE_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"
TEACHER_MODEL="${TEACHER_MODEL:-deepseek-v4-flash}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
OUT="$ROOT/outputs/evaluation/$LABEL"
SHARD_DIR="$OUT/shards"
OURS_SUMMARY="${OURS_SUMMARY:-}"

fail() { printf '\n[FAIL] %s\n' "$*" >&2; exit 1; }

echo "== 0/6 预检 =="
[[ -x "$PY" ]] || fail "缺少 $PY：先跑 bash scripts/setup.sh"
[[ -f "$BENCHMARK" ]] || fail "找不到评测集 $BENCHMARK"
[[ "$SHARDS" =~ ^[0-9]+$ ]] || fail "SHARDS 必须是整数，当前 '$SHARDS'"
env_code=$(curl -s -o /dev/null -m 3 -w '%{http_code}' "$SHOPSIM_BASE_URL/" 2>/dev/null)
env_code=${env_code:-000}
if [[ "$env_code" == "000" ]]; then
  fail "商店环境未启动（$SHOPSIM_BASE_URL）。另开一个终端跑：bash scripts/start_environment.sh"
fi
printf '  商店环境在线（%s -> HTTP %s；404 是正常信号）\n' "$SHOPSIM_BASE_URL" "$env_code"
if [[ -z "$TOKENIZER_PATH" ]]; then
  # 优先用冠军权重目录里的 tokenizer：与历史 63.5% 的截断口径同源。
  for candidate in \
    "$ROOT/outputs/models/sft-m10r/stage-full/merged" \
    "$ROOT/outputs/models/sft-b-curriculum/stage-c/merged"; do
    if [[ -f "$candidate/tokenizer.json" || -f "$candidate/tokenizer_config.json" ]]; then
      TOKENIZER_PATH="$candidate"
      break
    fi
  done
fi
if [[ ! -f "$TOKENIZER_PATH/tokenizer.json" && ! -f "$TOKENIZER_PATH/tokenizer_config.json" ]]; then
  fail "找不到本地 tokenizer（试过冠军权重目录）。用 TOKENIZER_PATH=<完整权重目录> 指定。
       它用来给 observation 做截断计数，必须存在。"
fi
printf '  tokenizer: %s\n' "$TOKENIZER_PATH"

echo
echo "== 1/6 老师凭据（现在放） =="
# 凭据只在本进程内使用，不落盘、不写进任何产物文件。
if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
  [[ -t 0 ]] || fail "非交互环境下请用环境变量给凭据：OPENAI_BASE_URL=... OPENAI_API_KEY=..."
  read -r -p "  老师 base_url（形如 https://xxx/v1）： " OPENAI_BASE_URL
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  [[ -t 0 ]] || fail "非交互环境下请用环境变量给凭据：OPENAI_BASE_URL=... OPENAI_API_KEY=..."
  read -r -s -p "  老师 API Key（输入不回显，回车结束）： " OPENAI_API_KEY
  echo
fi
[[ -n "${OPENAI_BASE_URL:-}" ]] || fail "base_url 为空"
[[ -n "${OPENAI_API_KEY:-}" ]] || fail "api_key 为空"
printf '  模型: %s\n' "$TEACHER_MODEL"
printf '  端点: %s\n' "$OPENAI_BASE_URL"

echo
echo "== 2/6 切片（$SHARDS 份） =="
mkdir -p "$OUT"
# 默认保留已跑完的分片轨迹（collect_tasks 会跳过已完成的 task），
# 中途失败只要重跑本脚本就能续。要全部重来用 FRESH=1。
if [[ "${FRESH:-0}" == "1" ]]; then
  echo "  FRESH=1：清空旧分片"
  rm -rf "$SHARD_DIR"
fi
"$PY" scripts/split_eval_shards.py \
  --benchmark "$BENCHMARK" \
  --shard-dir "$SHARD_DIR" \
  --shards "$SHARDS"

echo
echo "== 3/6 开跑（$SHARDS 并发，每题最多 35 步） =="
declare -a pids=()
for ((i = 0; i < SHARDS; i++)); do
  shard="$SHARD_DIR/$(printf 'shard-%02d' "$i")"
  "$PY" scripts/evaluate_shop_benchmark.py \
    --benchmark "$shard/tasks.jsonl" \
    --output "$shard/trajectories.jsonl" \
    --summary "$shard/summary.json" \
    --base-url "$SHOPSIM_BASE_URL" \
    --model "$TEACHER_MODEL" \
    --llm-base-url "$OPENAI_BASE_URL" \
    --api-key "$OPENAI_API_KEY" \
    --context-window 0 \
    --observation-token-budget 1536 \
    --tokenizer-path "$TOKENIZER_PATH" \
    --temperature 0.0 \
    --top-p 1.0 \
    --max-steps 35 \
    --timeout "${TEACHER_TIMEOUT:-180}" \
    >"$shard/run.log" 2>&1 &
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
echo "== 4/6 合并 200 题 =="
"$PY" scripts/merge_eval_shards.py \
  --benchmark "$BENCHMARK" \
  --shard-dir "$SHARD_DIR" \
  --shards "$SHARDS" \
  --output-dir "$OUT" \
  --model "$TEACHER_MODEL" || fail "合并失败：上面的报错说了原因；补跑缺失分片后重跑本脚本"

echo
echo "== 5/6 报告 =="
"$PY" scripts/build_eval_report.py --run-dir "$OUT" || printf '  报告生成失败，但 summary.json 已写好\n'

echo
echo "== 6/6 与冠军逐题对比 =="
if [[ -z "$OURS_SUMMARY" ]]; then
  OURS_SUMMARY=$("$PY" - "$ROOT" "$LABEL" <<'PY'
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

if [[ -z "$OURS_SUMMARY" ]]; then
  printf '  没找到可对比的冠军 summary（200 题口径）。用 OURS_SUMMARY=outputs/evaluation/<目录>/summary.json 指定后重跑这一步。\n'
else
  # 自动探测给的是相对仓库根的路径；显式传的可能是绝对路径。
  ours_path="$OURS_SUMMARY"
  [[ -f "$ours_path" ]] || ours_path="$ROOT/$OURS_SUMMARY"
  printf '  对比对象：%s\n' "$ours_path"
  "$PY" scripts/compare_teacher_vs_ours.py \
    --teacher-summary "$OUT/summary.json" \
    --ours-summary "$ours_path" \
    --teacher-name "$TEACHER_MODEL" \
    --ours-name "冠军" \
    --output-dir "$OUT"
fi

echo
echo "产物："
echo "  $OUT/summary.json            老师成绩（Final-200 Clean）"
echo "  $OUT/report.html             逐题报告"
echo "  $OUT/teacher-vs-ours.md      「老师对而我们错」的清单与行为对照"
echo
echo "注意：仓库里老师的 139/200（69.5%）跑在旧 Final-200 上，不要和这里的数字混着说。"
