#!/usr/bin/env bash
# 盘点磁盘占用，标出「不能删」的冻结件。
#
# 为什么要有这个
# --------------
# 每轮训练都会往 outputs/models 写 10 GB 以上（两段 merged 各 4–5 GB + 按步存盘的
# LoRA checkpoint），而「哪些是冠军权重 / 冻结起点」只存在于人的记忆里。删错一个
# 目录，62% 的对照基线就没了，之后所有比较都失去意义。
#
# 所以保护清单写死在代码里（见 PROTECT），脚本只允许删「候选」，并且**祖先目录
# 不会被删**——比如 sft-b-curriculum 下面还有 stage-c/merged，它自己就只标
# 「含冻结件」，不会整目录被删掉。
#
# 用法
# ----
#   bash scripts/audit_disk_usage.sh              # 只读盘点（默认）
#   bash scripts/audit_disk_usage.sh --delete     # 删掉所有「候选」
#
# 退出码：0 正常；1 有候选但没加 --delete（只是提示，不算失败）→ 仍返回 0。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DELETE=0
for arg in "$@"; do
  case "${arg}" in
    --delete) DELETE=1 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "未知参数：${arg}" >&2; exit 2 ;;
  esac
done

# ------------------------------------------------------------------ 保护清单
# 每一条都是「删了就没法复现结论」的东西。改动这里之前先想清楚。
PROTECT=(
  # 冠军权重（Final-200 62.0%）与 M10-R 的起点
  "outputs/models/sft-b-curriculum/stage-c/merged"
  # GRPO champion：grpo-m6/global_step_50
  "outputs/models/grpo-m6/global_step_50"
  # 注意：m10h 的三份 GRPO 产物（grpo-m10h / -merged / -s50-merged，约 30 GB）**不保护**。
  # 那次实验是 62%→62% 的空转，结论已入 docs/project-history.md，逐题证据留在
  # outputs/evaluation/grpo-m10h*（几十 MB）。删掉只失去「再起一次那个模型」的能力。
  # 本轮实验产物（stage-a 还要单独评测）；连 adapter 一起留，否则没法续跑。
  # stage-full 是单阶段方案（M10-R v2）的产物目录，命名规则与 a/b 一致。
  "outputs/models/sft-m10r/stage-a"
  "outputs/models/sft-m10r/stage-b"
  "outputs/models/sft-m10r/stage-full"
  # 证据归档（M6-R 崩溃现场）
  "outputs/archive"
  # 评测结果与训练日志：分数、失败解剖、每步 loss 的唯一原始记录，都只有几十 MB
  "outputs/evaluation"
  "outputs/logs"
  # 数据池：重建代价是几小时采集 + 几十块 API 费用
  "data/sft_pure_v4"
  "data/sft_m10r"
  "data/collection"
  "data/evaluation"
  "data/sft_curriculum"
  "data/grpo"
  "data/sft"
  # 探针集（dev-300 与训练集自测卷）：只有 1 MB，但没了就做不了「尺子」层面的任何比较。
  # 曾经被误判成候选——`--delete` 会把它删掉，所以显式列进来。
  "data/probes"
)

is_protected() {
  local path="$1" p
  for p in "${PROTECT[@]}"; do
    [[ "${path}" == "${p}" || "${path}" == "${p}"/* ]] && return 0
  done
  return 1
}

contains_protected() {
  local path="$1" p
  for p in "${PROTECT[@]}"; do
    [[ "${p}" == "${path}"/* ]] && return 0
  done
  return 1
}

classify() {
  local path="$1"
  if is_protected "${path}"; then
    echo "保留"
  elif contains_protected "${path}"; then
    echo "含冻结件"
  else
    echo "候选"
  fi
}

size_mb() {
  du -sm "$1" 2>/dev/null | cut -f1
}

CANDIDATES=()

# 递归展开：大目录（≥300 MB）继续往下拆，小目录整块显示。
report() {
  local path="$1" depth="$2"
  local mb label
  mb="$(size_mb "${path}")"
  [[ -z "${mb}" ]] && return
  label="$(classify "${path}")"
  printf '%7s MB  %-52s %s\n' "${mb}" "${path#"${ROOT}/"}" "${label}"
  if [[ "${label}" == "候选" ]]; then
    CANDIDATES+=("${path}")
    return
  fi
  if (( mb >= 300 && depth < 4 )); then
    local child
    for child in "${path}"/*/; do
      [[ -d "${child}" ]] || continue
      report "${child%/}" $((depth + 1))
    done
  fi
}

show_fs() {
  local label="$1" path="$2"
  [[ -e "${path}" ]] || return 0
  printf '%-24s %s\n' "${label}" "$(df -h "${path}" 2>/dev/null | tail -1)"
}

echo "== 磁盘（分别看，因为 outputs 常常是软链到另一块盘）"
show_fs "仓库（代码 + .venv）" .
show_fs "outputs（权重/评测）" outputs
show_fs "data（池子）" data
echo
echo "== 占用明细（只展开大目录）"
echo "   大小        路径                                                 处理"
shopt -s nullglob
for base in outputs/models outputs/evaluation outputs/logs outputs/analysis outputs/pytest-tmp data; do
  [[ -d "${base}" ]] || continue
  # 整块都是候选就整块列，不往下拆成几十行噪音。
  if [[ "$(classify "${base}")" == "候选" ]]; then
    report "${base}" 0
    continue
  fi
  children=("${base}"/*/)
  if (( ${#children[@]} == 0 )); then
    report "${base}" 0
    continue
  fi
  for child in "${children[@]}"; do
    report "${child%/}" 1
  done
done

# 缓存单独列，不进候选：模型缓存删了训练要重新下载（国内网络会卡在 Retry）。
echo
echo "== 缓存（不参与自动清理）"
for cache in "${HOME}/.cache/huggingface" "/root/rivermind-data/hf"; do
  [[ -d "${cache}" ]] && printf '%7s MB  %s\n' "$(size_mb "${cache}")" "${cache}"
done

# 仓库以外的占用：盘满的时候，往往是这些东西而不是 outputs 在吃空间。
# 只读、不参与删除；列出来是为了不把「删了仓库里的东西还是不够」这种情况留到最后才发现。
echo
echo "== 仓库以外（只读，不参与清理）"
for extra in \
  "${ROOT}/.venv" \
  "${ROOT}/.tmp-pytest" \
  "${HOME}/.cache/uv" \
  "${HOME}/.cache/pip" \
  "${HOME}/.cache/vllm" \
  "${HOME}/.triton" \
  "${HOME}/.nv" \
  "${HOME}/.ollama" \
  "/var/lib/docker"; do
  [[ -e "${extra}" ]] || continue
  printf '%7s MB  %s\n' "$(size_mb "${extra}")" "${extra}"
done

MOUNT="$(df -P "${ROOT}" 2>/dev/null | tail -1 | awk '{print $6}')"
if [[ -n "${MOUNT}" && -d "${MOUNT}" && "${MOUNT}" != "/" ]]; then
  echo
  echo "== ${MOUNT} 顶层占用（前 12，只读）"
  du -sm "${MOUNT}"/* 2>/dev/null | sort -rn | head -12 | while read -r mb path; do
    printf '%7s MB  %s\n' "${mb}" "${path}"
  done
fi

if (( ${#CANDIDATES[@]} == 0 )); then
  echo
  echo "没有可删的候选目录。"
  exit 0
fi

total=0
for path in "${CANDIDATES[@]}"; do
  mb="$(size_mb "${path}")"
  [[ -n "${mb}" ]] && total=$((total + mb))
done

echo
echo "== 候选（共 ${#CANDIDATES[@]} 项，合计约 ${total} MB ≈ $((total / 1024)) GB）"
for path in "${CANDIDATES[@]}"; do
  echo "  ${path#"${ROOT}/"}"
done

echo
echo "== 大头（不能删，列出来是为了知道空间都去哪了）"
for p in "${PROTECT[@]}"; do
  [[ -e "${p}" ]] || continue
  mb="$(size_mb "${p}")"
  [[ -n "${mb}" ]] && (( mb >= 500 )) && printf '%7s MB  %s\n' "${mb}" "${p}"
done

if (( DELETE == 0 )); then
  echo
  echo "以上只是清单，什么都没删。确认无误后执行："
  echo "  bash scripts/audit_disk_usage.sh --delete"
  exit 0
fi

echo
echo "== 开始删除"
for path in "${CANDIDATES[@]}"; do
  # 二次保险：删除前再判一次，防止 report 的分类被后续改动影响。
  if [[ "$(classify "${path}")" != "候选" ]]; then
    echo "  [skip] ${path}（分类已变，跳过）"
    continue
  fi
  echo "  [del ] ${path#"${ROOT}/"}（$(size_mb "${path}") MB）"
  rm -rf -- "${path}"
done
echo
show_fs "outputs（删除后）" outputs
