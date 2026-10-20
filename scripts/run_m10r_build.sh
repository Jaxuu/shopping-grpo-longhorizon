#!/usr/bin/env bash
# M10-R：把筛出来的「答案不在第 1 名」样本拼成一份可训练的 SFT 池。
#
# 全部纯 CPU，不花显卡也不花 API。三步串起来：
#   1. build   —— 合并 + 配比 + 契约校验（写 all.jsonl / difficulty_labels.jsonl）
#   2. label   —— 给新题补离线难度标签（课程清单要用）
#   3. manifest—— 生成课程清单（含与评测集的零重叠校验）
#
# 用法：
#   bash scripts/run_m10r_build.sh                  # 默认不降采样（第 1 名占 52%）
#   bash scripts/run_m10r_build.sh --target-share 0.60   # 强干预臂
#   bash scripts/run_m10r_build.sh --dry-run        # 只打印训练命令，不训练
#
# 训练（要显卡）另跑：
#   bash scripts/sft_curriculum.sh --source data/sft_m10r/all.jsonl \
#       --manifest data/sft_m10r/manifest.json \
#       --output-root outputs/models/sft-m10r \
#       --loss-scope tool-call-only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PY="${PY:-${ROOT}/.venv/bin/python}"
INCREMENT="${INCREMENT:-outputs/m10-rank2/kept.jsonl}"
OUT_DIR="${OUT_DIR:-data/sft_m10r}"
POOL="${POOL:-data/sft_pure_v4/all.jsonl}"
LABELS="${LABELS:-data/sft_pure_v4/difficulty_labels.jsonl}"
EVALUATION="${EVALUATION:-data/evaluation/tasks.jsonl}"

if [[ ! -f "${INCREMENT}" ]]; then
  echo "没有增量文件 ${INCREMENT}；先跑 bash scripts/run_m10_rank2.sh collect/filter。" >&2
  exit 1
fi

# 只透传配比旋钮；--dry-run 是本脚本自己的开关
BUILD_ARGS=()
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --target-share) BUILD_ARGS+=(--target-share "$2"); shift 2 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

echo "[1/3] 合并与配比"
"${PY}" scripts/build_m10r_pool.py \
  --pool "${POOL}" \
  --increment "${INCREMENT}" \
  --out-dir "${OUT_DIR}" \
  --labels "${LABELS}" \
  --evaluation "${EVALUATION}" \
  "${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}"

echo
echo "[2/3] 给新题补离线难度标签"
"${PY}" scripts/label_sft_difficulty_offline.py \
  --input "${OUT_DIR}/all.jsonl" \
  --output "${OUT_DIR}/difficulty_labels.jsonl"

echo
echo "[3/3] 生成课程清单"
"${PY}" scripts/prepare_sft_curriculum.py \
  --source "${OUT_DIR}/all.jsonl" \
  --labels "${OUT_DIR}/difficulty_labels.jsonl" \
  --evaluation "${EVALUATION}" \
  --output "${OUT_DIR}/manifest.json"

echo
TRAIN_CMD=(
  bash scripts/sft_curriculum.sh
  --source "${OUT_DIR}/all.jsonl"
  --manifest "${OUT_DIR}/manifest.json"
  --output-root outputs/models/sft-m10r
  --loss-scope tool-call-only
)
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "训练命令（--dry-run 只打印，不执行）："
  printf '  %q' "${TRAIN_CMD[@]}"; echo
  "${PY}" scripts/run_sft_curriculum.py \
    --source "${OUT_DIR}/all.jsonl" \
    --manifest "${OUT_DIR}/manifest.json" \
    --output-root outputs/models/sft-m10r \
    --loss-scope tool-call-only \
    --dry-run
  exit 0
fi

echo "完成。训练命令（要显卡，M5-B 同款 3 阶段课程 + tool-call-only loss）："
printf '  %q' "${TRAIN_CMD[@]}"; echo
