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
# 模板轨迹保留条数（1,912 → 约 1,250 的那一刀）与课程分桶口径。
KEEP_TEMPLATE="${KEEP_TEMPLATE:-700}"
BUCKET_MODE="${BUCKET_MODE:-behavior}"
# reverse = 倒序两段（M10-R v1，已证伪：B 段把 A 段的翻页从 30% 压过冲到 1%）；
# single  = 单阶段全量（M10-R v2）。
STAGE_PLAN="${STAGE_PLAN:-reverse}"
# 单阶段必须从 62% 那个已收敛的基线热启动（M5 从原始底座用了 423 步才到 62%，
# 单阶段只有 136 步）。这里只用于**打印**训练命令，真正透传在 run_m10r_pipeline.sh。
WARM_START="${WARM_START:-outputs/models/sft-b-curriculum/stage-c/merged}"

mkdir -p "${OUT_DIR}"
STAGED="${OUT_DIR}/increment_merged.jsonl"
echo "[0/4] 合并增量并去重（同题只留一条，顺序即优先级）"
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

echo "[1/4] 合并与配比（去重后增量 $(wc -l < "${STAGED}") 行）"
"${PY}" scripts/build_m10r_pool.py \
  --pool "${POOL}" \
  --increment "${INCREMENT}" \
  --out-dir "${OUT_DIR}" \
  --labels "${LABELS}" \
  --evaluation "${EVALUATION}" \
  "${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}"

echo
echo "[2/4] 给新题补离线难度标签"
"${PY}" scripts/label_sft_difficulty_offline.py \
  --input "${OUT_DIR}/all.jsonl" \
  --output "${OUT_DIR}/difficulty_labels.jsonl"

echo
echo "[3/4] 砍掉多余的模板轨迹（改配权，不是丢数据）"
# 1,912 条里 71% 是「搜一次、开第一个、买」。旧课程又把它练 2–3 遍，曝光比到 5:1，
# 结果只会让模型更确定地「开第一个」，GRPO 需要的组内方差被压没。
# 探索轨迹一条不动，模板留到 KEEP_TEMPLATE 条。
CAPPED="${OUT_DIR}/capped"
"${PY}" scripts/downsample_sft_pool.py \
  --pool "${OUT_DIR}/all.jsonl" \
  --labels "${OUT_DIR}/difficulty_labels.jsonl" \
  --keep-template "${KEEP_TEMPLATE}" \
  --out-dir "${CAPPED}"

echo
if [[ "${STAGE_PLAN}" == "single" ]]; then
  echo "[4/4] 生成课程清单（单阶段：全量分布 1 个 epoch）"
else
  echo "[4/4] 生成课程清单（倒序两段：先探索，后全量）"
fi
"${PY}" scripts/prepare_sft_curriculum.py \
  --source "${CAPPED}/all.jsonl" \
  --labels "${CAPPED}/difficulty_labels.jsonl" \
  --evaluation "${EVALUATION}" \
  --bucket-mode "${BUCKET_MODE}" \
  --stage-plan "${STAGE_PLAN}" \
  --output "${CAPPED}/manifest.json"

echo
# 桶与阶段的分布直接打出来：哪一类数据被练几遍，是这次干预的核心变量。
"${PY}" scripts/show_sft_manifest.py "${CAPPED}/manifest.json"

echo
TRAIN_CMD=(
  bash scripts/sft_curriculum.sh
  --source "${CAPPED}/all.jsonl"
  --manifest "${CAPPED}/manifest.json"
  --output-root outputs/models/sft-m10r
  --loss-scope tool-call-only
)
# 单阶段漏了 --base-model 就会从 Qwen3.5-2B 重新学「怎么购物」，136 步不够。
BASE_ARGS=()
if [[ "${STAGE_PLAN}" == "single" ]]; then
  BASE_ARGS=(--base-model "${WARM_START}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "训练命令（--dry-run 只打印，不执行）："
  printf '  %q' "${TRAIN_CMD[@]}" "${BASE_ARGS[@]}"; echo
  "${PY}" scripts/run_sft_curriculum.py \
    --source "${CAPPED}/all.jsonl" \
    --manifest "${CAPPED}/manifest.json" \
    --output-root outputs/models/sft-m10r \
    --loss-scope tool-call-only \
    "${BASE_ARGS[@]}" \
    --dry-run
  exit 0
fi

echo "完成。训练命令（要显卡；$( [[ "${STAGE_PLAN}" == "single" ]] && echo "单阶段全量" || echo "倒序两段课程" ) + tool-call-only loss）："
printf '  %q' "${TRAIN_CMD[@]}" "${BASE_ARGS[@]}"; echo
