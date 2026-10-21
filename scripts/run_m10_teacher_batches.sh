#!/usr/bin/env bash
# M10-R 真实采集：把探针判「可采集」的两批从老师轨迹里收上来。
#
# 背景（2026-10-21 探针全量 480 条，120 题 × 4 次，温度 0.9）
# -------------------------------------------------------
# | 场景 | 动作率 | 可收率 | 判定 |
# |---|---|---|---|
# | 翻页 | 15.6% | 1.9% | 采不到 ⇒ 合成（老师有更省事的替代策略：换词重搜）|
# | 换词重搜 | 83.1% | 25.0% | **可采集** |
# | 对比候选 | 81.9% | 27.5% | **可采集** |
#
# 所以这两批改走真实采集：收的是**老师自己的策略**，不是脚本规定的动作序列。
# 翻页批仍走 scripts/generate_m10_paging.py 合成。
#
# 第三批是白捡的：翻页题 40 道里老师买对 52 次，其中 49 次没翻页而是换词找到的
# （这批题 gold 在第 21–60 名，第一页看不到）。这批轨迹已经在探针 raw 里，
# 用 paging_requery 模式重筛即可，**不花一分钱、不跑采集**。
#
# 用法
# ----
#   bash scripts/run_m10_teacher_batches.sh requery         # 采集 + 筛选
#   bash scripts/run_m10_teacher_batches.sh compare
#   bash scripts/run_m10_teacher_batches.sh paging_requery  # 只筛选，用探针已有的 raw
#   bash scripts/run_m10_teacher_batches.sh requery status   # 盘点已跑多少、还剩多少（不花钱）
#   bash scripts/run_m10_teacher_batches.sh requery select  # 只筛选（已有 raw.jsonl）
#   bash scripts/run_m10_teacher_batches.sh requery collect # 只采集
#
# 断点续跑：raw.jsonl 是事实来源，重跑同一命令会跳过已完成的 (task, attempt)。
# 已经探针采过的题会跳过，不重复花钱；筛选时把探针 raw 一起并进来。
# **续跑前先跑 status**：余额/网络中断会留下没到终态的错误行，
# 不清掉就会被当成已完成而永久跳过（status 会提示要不要 purge）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SCENARIO="${1:-}"
MODE="${2:-all}"
PY="${PY:-${ROOT}/.venv/bin/python}"
ENV_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"
MODEL="${OPENAI_MODEL:-deepseek-v4-flash}"

ATTEMPTS="${BATCH_ATTEMPTS:-4}"
TEMPERATURE="${BATCH_TEMPERATURE:-0.9}"
WORKERS="${BATCH_WORKERS:-8}"
PROBE_RAW="${PROBE_RAW:-outputs/m10-teacher-probe/raw.jsonl}"

# 可选覆盖项：目标动作（与题源场景解耦）与「只收清单里某场景的题」。
TARGETS_SCENARIO=""
ACTION=""
SELECT_ONLY=0
SELECT_SCENARIO=""

case "${SCENARIO}" in
  requery)
    TARGETS="data/collection/m10_requery_targets.jsonl"
    OUT_DIR="${BATCH_OUT_DIR:-outputs/m10-requery}"
    SELECT_SCENARIO="requery"
    ;;
  compare)
    TARGETS="data/collection/m10_compare_targets.jsonl"
    OUT_DIR="${BATCH_OUT_DIR:-outputs/m10-compare}"
    SELECT_SCENARIO="compare"
    ;;
  paging_requery)
    # 题源场景是翻页，但收的是换词轨迹 ⇒ 筛选用 --scenario paging --action requeried
    TARGETS="data/collection/m10_teacher_probe_targets.jsonl"
    OUT_DIR="${BATCH_OUT_DIR:-outputs/m10-paging-requery}"
    TARGETS_SCENARIO="paging"
    ACTION="requeried"
    SELECT_ONLY=1
    SELECT_SCENARIO="paging"
    ;;
  *)
    echo "用法：bash scripts/run_m10_teacher_batches.sh {requery|compare|paging_requery} [all|collect|select|status]" >&2
    exit 2
    ;;
esac

check_environment() {
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 --noproxy '*' "${ENV_URL}/" || true)"
  if [[ "${code}" != "404" ]]; then
    cat >&2 <<EOF
环境服务不可用：${ENV_URL} 返回 ${code}（期望 404）
先起：tmux new -s env && cd ${ROOT} && bash scripts/start_environment.sh
EOF
    exit 1
  fi
  echo "[ok] 环境服务在线：${ENV_URL}"
}

check_teacher() {
  if [[ -z "${OPENAI_BASE_URL:-}" || -z "${OPENAI_API_KEY:-}" ]]; then
    cat >&2 <<'EOF'
老师 API 没配。先 source 环境文件（不要在命令行里贴 key）：
  source ~/.m2_collect.env
EOF
    exit 1
  fi
  echo "[ok] 老师模型：${MODEL} @ ${OPENAI_BASE_URL}"
}

# 探针已经采过的题不再重复花钱。清单是同一个池子抽的，重叠部分直接用探针的轨迹。
pending_targets() {
  local out="$1"
  "${PY}" - "${TARGETS}" "${PROBE_RAW}" "${out}" <<'PY'
import json, sys

targets_path, probe_path, out_path = sys.argv[1:4]
done = set()
try:
    with open(probe_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                done.add(int(json.loads(line)["task_id"]))
except FileNotFoundError:
    pass

kept = []
with open(targets_path, encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        if int(json.loads(line)["task_id"]) in done:
            continue
        kept.append(line)

with open(out_path, "w", encoding="utf-8") as handle:
    handle.write("\n".join(kept) + ("\n" if kept else ""))
print(f"  探针已采 {len(done)} 题，本批还需 {len(kept)} 题")
PY
}

collect() {
  if [[ "${SELECT_ONLY}" -eq 1 ]]; then
    echo "[skip] ${SCENARIO} 不采集：这批轨迹已经在 ${PROBE_RAW} 里，直接筛选。"
    return 0
  fi
  check_environment
  check_teacher
  mkdir -p "${OUT_DIR}"
  local targets="${OUT_DIR}/pending_targets.jsonl"
  pending_targets "${targets}"
  local tasks total
  tasks="$(grep -c . "${targets}" || true)"
  if [[ "${tasks}" -eq 0 ]]; then
    echo "[skip] 这批题都被探针采过了，直接筛选即可："
    echo "  bash scripts/run_m10_teacher_batches.sh ${SCENARIO} select"
    return 0
  fi
  total=$(( tasks * ATTEMPTS ))
  echo "[run] ${SCENARIO}：${tasks} 题 × ${ATTEMPTS} 次 = ${total} 条，temperature ${TEMPERATURE}，${WORKERS} 并发"
  "${PY}" scripts/collect_sft_data.py \
    --tasks "${targets}" \
    --output-dir "${OUT_DIR}" \
    --model "${MODEL}" \
    --base-url "${ENV_URL}" \
    --attempts-per-task "${ATTEMPTS}" \
    --workers "${WORKERS}" \
    --target-accepted "${total}" \
    --temperature "${TEMPERATURE}" \
    --max-steps 35
}

select_batch() {
  local merged="${OUT_DIR}/raw_all.jsonl"
  mkdir -p "${OUT_DIR}"
  : > "${merged}"
  if [[ -f "${PROBE_RAW}" ]]; then
    cat "${PROBE_RAW}" >> "${merged}"
    echo "[merge] 并入探针轨迹 ${PROBE_RAW}"
  fi
  if [[ -f "${OUT_DIR}/raw.jsonl" ]]; then
    cat "${OUT_DIR}/raw.jsonl" >> "${merged}"
    echo "[merge] 并入本批轨迹 ${OUT_DIR}/raw.jsonl"
  fi
  if [[ ! -s "${merged}" ]]; then
    echo "没有可用的 raw.jsonl，先跑 collect。" >&2
    exit 1
  fi
  local extra=()
  if [[ -n "${ACTION}" ]]; then
    extra+=(--action "${ACTION}")
  fi
  if [[ -n "${TARGETS_SCENARIO}" ]]; then
    extra+=(--targets-scenario "${TARGETS_SCENARIO}")
  fi
  "${PY}" scripts/select_teacher_pattern_batch.py \
    "${merged}" \
    --scenario "${SELECT_SCENARIO}" \
    --targets "${TARGETS}" \
    ${extra[@]+"${extra[@]}"} \
    --out-dir "${OUT_DIR}"
  echo
  echo "[next] 并入池子：bash scripts/run_m10r_build.sh --dry-run"
}

case "${MODE}" in
  all)     collect; echo; select_batch ;;
  collect) collect ;;
  select)  select_batch ;;
  status)
    # 只读盘点：这批跑了多少、有没有需要清理的错误行、还剩多少条。
    if [[ ! -f "${OUT_DIR}/raw.jsonl" ]]; then
      echo "还没有任何轨迹：${OUT_DIR}/raw.jsonl 不存在。" >&2
      exit 1
    fi
    mkdir -p "${OUT_DIR}"
    # 必须对照「本批待采清单」而不是全量清单：探针已采过的题不在本批里，
    # 拿全量清单比会把它们算成「未开始」，把剩余量吹大一倍（2026-10-22 实测：
    # 报 254 条，实际 94 条）。
    pending="${OUT_DIR}/pending_targets.jsonl"
    pending_targets "${pending}"
    "${PY}" scripts/inspect_collection_raw.py "${OUT_DIR}/raw.jsonl" \
      --targets "${pending}" --attempts "${ATTEMPTS}"
    echo
    echo "[提示] 本批全量清单 ${TARGETS}（含探针已采的题）；上面只算本批还要补多少。"
    ;;
  *)
    echo "未知模式：${MODE}（可选 all / collect / select / status）" >&2
    exit 2
    ;;
esac
