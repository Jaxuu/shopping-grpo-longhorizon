#!/usr/bin/env bash
# 同一模型、同一套题，跑 N 次（带温度），再把 N 份轨迹拿去多数票。
#
# 为什么要这么做
# --------------
# 现在的评测是贪心解码（temperature=0），同一题跑一百次也是同一个答案，
# 投票毫无意义。把温度调到 0.7 之后每次采样走的路不一样，错误会分散，
# 而正确答案只有一个 —— 这正是多数票能涨分的机制（self-consistency）。
#
# 预登记判据（跑之前就定死，跑完不许改）
# --------------------------------------
# 脚本最后会打印「并集」= N 次里只要有一次做对就算对，也就是 oracle@N。
#   1. 并集 < 140/200（70%）  → 这条路的物理上限都不到 70%，直接放弃，不要再调温度。
#   2. 并集 ≥ 140 且多数票 ≥ 140 → 达成，把「多数票」写进结论。
#   3. 并集 ≥ 140 但多数票 < 140 → 涨分空间存在但选择规则不够好，再谈改进选择规则。
#
# 代价
# ----
# 200 题 × 3.4 s ≈ 11 分钟/次。6 次 ≈ 70 分钟。纯推理，不动权重。
#
# 前置条件
# --------
# vLLM 与商店环境必须已经在跑（tmux 里的 vllm / env 两个会话）。
# 本脚本只做「循环调评测 + 汇总」，不会重启 vLLM（重启要 20 分钟，别浪费）。
#
# 用法::
#
#   bash scripts/run_self_consistency.sh sc            # 6 次，温度 0.7
#   SAMPLES=10 bash scripts/run_self_consistency.sh sc # 10 次
#   TEMPERATURE=1.0 bash scripts/run_self_consistency.sh sc-hot
#
# 断点续跑：已经跑满的采样目录会被跳过，中断了重跑同一条命令即可。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${1:-sc}"
SAMPLES="${SAMPLES:-6}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-1.0}"
BENCHMARK="${BENCHMARK:-$ROOT/data/evaluation/tasks.jsonl}"

cd "$ROOT"

if [[ ! -f "$BENCHMARK" ]]; then
  echo "找不到考卷：$BENCHMARK" >&2
  exit 1
fi
EXPECTED="$(wc -l < "$BENCHMARK" | tr -d ' ')"

# 采样目录用同一个前缀，汇总时只认这个前缀，不会把历史评测目录混进来。
DIRS=()
for i in $(seq 1 "$SAMPLES"); do
  DIRS+=("$ROOT/outputs/evaluation/${LABEL}-s${i}")
done

echo "== 自洽投票：${SAMPLES} 次采样，温度 ${TEMPERATURE}，top_p ${TOP_P}，考卷 ${EXPECTED} 题"
echo

for i in $(seq 1 "$SAMPLES"); do
  OUT="${ROOT}/outputs/evaluation/${LABEL}-s${i}"
  TRAJ="$OUT/trajectories.jsonl"
  if [[ -f "$TRAJ" ]]; then
    GOT="$(wc -l < "$TRAJ" | tr -d ' ')"
    if [[ "$GOT" -ge "$EXPECTED" ]]; then
      echo "-- 第 ${i}/${SAMPLES} 次已完成（${GOT} 题），跳过"
      continue
    fi
    echo "-- 第 ${i}/${SAMPLES} 次只跑了 ${GOT}/${EXPECTED} 题，重跑"
  fi
  echo "-- 第 ${i}/${SAMPLES} 次开始 $(date '+%H:%M:%S')"
  EVAL_OUTPUT_DIR="$OUT" \
  BENCHMARK="$BENCHMARK" \
  TEMPERATURE="$TEMPERATURE" \
  TOP_P="$TOP_P" \
    bash scripts/evaluate.sh "${LABEL}-s${i}"
done

echo
echo "== 汇总"
.venv/bin/python scripts/eval_ensemble.py "${DIRS[@]}" \
  --reference "$BENCHMARK" \
  --out "outputs/analysis/${LABEL}-ensemble.json"

echo
echo "判据：并集（oracle@${SAMPLES}）到不了 140/200 就放弃这条路；"
echo "      并集到了但多数票没到，问题在选择规则而不是采样次数。"
