#!/usr/bin/env bash
# M10-R：把筛出来的「答案不在第 1 名」样本拼成一份可训练的 SFT 池。
#
# 全部纯 CPU，不花显卡也不花 API。三步串起来：
#   1. build   —— 合并 + 配比 + 契约校验（写 all.jsonl / difficulty_labels.jsonl）
#   2. label   —— 给新题补离线难度标签（课程清单要用）
#   3. manifest—— 生成课程清单（含与评测集的零重叠校验）
#
# 增量来源有五批，存在哪批并哪批。**顺序 = 去重优先级**（同题只留一条）：
#   outputs/m10-paging-requery/kept.jsonl 翻页题里老师换词找到的真实轨迹（探针 raw 白捡）
#   outputs/m10-paging/kept.jsonl        合成「翻页才找到」（next_page 的唯一来源）
#   outputs/m10-compare/kept.jsonl       采集「先开第 1 名再对比」
#   outputs/m10-requery/kept.jsonl       采集「换查询重搜」
#   outputs/m10-rank2/kept.jsonl         采集「gold 名次≥2」（面最广但不教探索动作，垫底）
# 可用 PAGING_REQUERY_INCREMENT / PAGING_INCREMENT / COMPARE_INCREMENT /
# REQUERY_INCREMENT / RANK2_INCREMENT 覆盖。
# 注意：requery / compare 以前是脚本合成的，2026-10-21 探针判定后改走真实采集，
# 文件名不变但内容来源变了（详见 plans/m10r-runbook.md §3.0）。
# ⚠️ 各批清单同源、必然重叠（paging ⊂ rank2、探针翻页题 ⊂ paging），
# 所以合并必须去重——见 scripts/merge_m10r_increments.py。
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
# 增量来源按「去重优先级」排列：同一道题被多批覆盖时，留那条教了更稀缺动作的轨迹。
# 真实采集优先于合成；rank2 面最广但不教任何探索动作，所以垫底。
INCREMENT_SOURCES=(
  "${PAGING_REQUERY_INCREMENT:-outputs/m10-paging-requery/kept.jsonl}"
  "${PAGING_INCREMENT:-outputs/m10-paging/kept.jsonl}"
  "${COMPARE_INCREMENT:-outputs/m10-compare/kept.jsonl}"
  "${REQUERY_INCREMENT:-outputs/m10-requery/kept.jsonl}"
  "${RANK2_INCREMENT:-outputs/m10-rank2/kept.jsonl}"
)
OUT_DIR="${OUT_DIR:-data/sft_m10r}"
POOL="${POOL:-data/sft_pure_v4/all.jsonl}"
LABELS="${LABELS:-data/sft_pure_v4/difficulty_labels.jsonl}"
EVALUATION="${EVALUATION:-data/evaluation/tasks.jsonl}"

mkdir -p "${OUT_DIR}"
STAGED="${OUT_DIR}/increment_merged.jsonl"
echo "[0/3] 合并增量并去重（同题只留一条，顺序即优先级）"
"${PY}" scripts/merge_m10r_increments.py \
  --out "${STAGED}" \
  "${INCREMENT_SOURCES[@]}"
INCREMENT="${STAGED}"

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

echo "[1/3] 合并与配比（去重后增量 $(wc -l < "${STAGED}") 行）"
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
# 桶与阶段的分布直接打出来：新采的探索类轨迹几乎全进 strategy 桶，
# 而 strategy 只在最后阶段才出现——这件事不该靠人记。
"${PY}" scripts/show_sft_manifest.py "${OUT_DIR}/manifest.json"

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
