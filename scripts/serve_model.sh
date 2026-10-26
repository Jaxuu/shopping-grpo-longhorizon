#!/usr/bin/env bash
# 起一个 OpenAI 兼容的 vLLM 服务，供评测脚本调用。
#
#   bash scripts/serve_model.sh                                  # 原始底座（HF 仓库）
#   bash scripts/serve_model.sh outputs/models/<权重目录>          # 本地权重
#
# 本地路径先做存在性检查：vLLM 把不存在的路径当 HF repo id 去解析，报出来的
# 是 `OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'`，
# 看不出真实原因是「目录没了」。这里提前拦住并列出可用的权重目录。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 相对路径一律按仓库根解析，避免在别的目录里调用时找不到权重。
cd "$ROOT"

MODEL="${1:-Qwen/Qwen3.5-2B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-shopping-agent}"
LLM_PORT="${LLM_PORT:-8000}"
# 22G 卡上 2B 推理够用；若 vLLM 报 KV cache 不足，降到 0.85 或 0.8 重试。
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"

fail() { printf '\n[FAIL] %s\n' "$*" >&2; exit 1; }

if [[ ! -x "$ROOT/.venv/bin/vllm" ]]; then
  fail "vLLM is not installed. Run: bash scripts/setup.sh"
fi

# 判断这是本地路径还是 HF 仓库 id：显式路径前缀、或以 outputs/ 开头、或目录真实存在。
is_local=0
case "$MODEL" in
  /* | ./* | ../* | outputs/*) is_local=1 ;;
  *) [[ -d "$MODEL" ]] && is_local=1 ;;
esac

if [[ "$is_local" == "1" && ! -f "$MODEL/config.json" ]]; then
  printf '\n[FAIL] 本地权重目录不可用：%s\n' "$MODEL" >&2
  if [[ -d "$MODEL" ]]; then
    printf '       目录在，但没有 config.json —— 可能只合并了一半，或者被清理掉了。\n' >&2
  else
    printf '       目录不存在。\n' >&2
  fi
  printf '\n当前可用的权重目录（含 config.json）：\n' >&2
  found=0
  while IFS= read -r cfg; do
    [[ -z "$cfg" ]] && continue
    found=1
    printf '  %-52s %s\n' "$(dirname "$cfg")" "$(du -sh "$(dirname "$cfg")" 2>/dev/null | cut -f1)" >&2
  done < <(find outputs/models -maxdepth 4 -name config.json 2>/dev/null | sort)
  if [[ "$found" == "0" ]]; then
    printf '  （outputs/models 下没有找到任何可加载的权重）\n' >&2
  fi
  printf '\n磁盘余量：\n' >&2
  df -h . | tail -1 >&2
  exit 1
fi

if [[ "$is_local" == "1" ]]; then
  printf '本地权重：%s\n' "$MODEL"
else
  printf 'HF 仓库：%s（本地没有同名目录，按远端 id 解析）\n' "$MODEL"
fi
printf '端口 %s / 服务名 %s / 上下文 %s / 显存占用比 %s\n' \
  "$LLM_PORT" "$SERVED_MODEL_NAME" "24576" "$GPU_MEM_UTIL"
printf '首次加载权重 20-25 分钟，期间端口不通是正常的。\n\n'

exec "$ROOT/.venv/bin/vllm" serve "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --port "$LLM_PORT" \
  --max-model-len 24576 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
