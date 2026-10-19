#!/usr/bin/env bash
# 评测前自检（只读，不改任何东西）。
#
# 为什么要有这个：克隆实例后最常见的三种故障都不报错、只是卡住——
#   1) 数据盘没挂上，outputs 软链指向不存在的目录 → 评测把轨迹写到系统盘或直接失败；
#   2) .venv 里的绝对路径与当前实例不一致 → python 能启动但 vllm 找不到；
#   3) 显存不够 → vLLM 起服务要 20+ 分钟才报 OOM，白等。
# 这三件事都能在 10 秒内查完，所以先查再跑。
#
# 用法：
#   bash scripts/preflight_eval.sh
#   bash scripts/preflight_eval.sh <模型目录>          # 指定权重路径
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-$ROOT/outputs/models/sft-b-curriculum/stage-c/merged}"
DATA_DIR="${DATA_DIR:-/root/rivermind-data}"

FAIL=0
ok()   { printf '  [ OK ] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; FAIL=1; }
warn() { printf '  [warn] %s\n' "$*"; }

echo "== 1. 显卡 =="
if command -v nvidia-smi >/dev/null 2>&1; then
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  if [[ -n "${total:-}" ]]; then
    ok "GPU: ${name}，显存 ${total} MiB（空闲 ${free} MiB）"
    # 2B 模型 bf16 权重 4-5 GB + 24k 上下文 KV cache，实测峰值约 10-14 GB。
    if (( total < 16000 )); then
      bad "显存 ${total} MiB 偏小：2B 推理峰值约 10-14 GB，建议 >=16 GB"
    fi
    if (( ${free:-0} < 14000 )); then
      warn "当前空闲仅 ${free} MiB，起 vLLM 前先确认没有别的进程占卡：nvidia-smi"
    fi
  else
    bad "nvidia-smi 无法读取显存信息"
  fi
else
  bad "找不到 nvidia-smi（驱动未装或不在 PATH）"
fi

echo "== 2. CPU / 内存（环境服务是纯 CPU 密集型） =="
cpus=$(nproc 2>/dev/null || echo 0)
if (( cpus >= 8 )); then ok "CPU 核数 ${cpus}"; else warn "CPU 核数仅 ${cpus}：商店环境要起 32 个实例，检索会变慢"; fi
mem_gb=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')
if [[ -n "${mem_gb:-}" ]] && (( mem_gb >= 16 )); then
  ok "内存 ${mem_gb} GB"
else
  warn "内存 ${mem_gb:-?} GB：商品库 23,421 条 + 32 个环境实例会吃掉不少"
fi

echo "== 3. 数据盘与 outputs 软链 =="
if [[ -d "$DATA_DIR" ]]; then
  ok "数据盘 ${DATA_DIR} 存在"
  df -h "$DATA_DIR" | awk 'NR==2{printf "        可用 %s / 总 %s（已用 %s）\n", $4, $2, $5}'
else
  bad "数据盘 ${DATA_DIR} 不存在：检查挂载，或改 DATA_DIR=... 重跑"
fi
if [[ -L "$ROOT/outputs" ]]; then
  target=$(readlink -f "$ROOT/outputs" || true)
  if [[ -d "$target" ]]; then ok "outputs -> ${target}"; else bad "outputs 软链指向不存在的目录：${target}"; fi
elif [[ -d "$ROOT/outputs" ]]; then
  warn "outputs 是真实目录（不是软链）：产物会写系统盘，注意空间"
else
  bad "outputs 不存在"
fi
avail_kb=$(df -Pk "$ROOT/outputs" 2>/dev/null | awk 'NR==2{print $4}')
if [[ -n "${avail_kb:-}" ]] && (( avail_kb < 5000000 )); then
  warn "outputs 所在分区剩余不足 5 GB（约 ${avail_kb} KB）"
fi

echo "== 4. 主环境 .venv =="
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  ok "python: $("$ROOT/.venv/bin/python" -V 2>&1)"
else
  bad "缺少 $ROOT/.venv/bin/python：跑 bash scripts/setup.sh"
fi
if [[ -x "$ROOT/.venv/bin/vllm" ]]; then
  ok "vllm: $("$ROOT/.venv/bin/vllm" --version 2>&1 | tail -1)"
else
  bad "缺少 .venv/bin/vllm：grpo 依赖没装全，跑 bash scripts/setup.sh"
fi

echo "== 5. 商店环境 .venv-shopsim =="
SHOPSIM_PY="$ROOT/environments/ShopSimulator/.venv-shopsim/bin/python"
if [[ -x "$SHOPSIM_PY" ]]; then
  ok "shopsim python: $("$SHOPSIM_PY" -V 2>&1)"
else
  bad "缺少 .venv-shopsim：跑 bash scripts/setup.sh"
fi
INDEX="$ROOT/environments/ShopSimulator/shop_env/search_engine/products.sqlite3"
if [[ -f "$INDEX" ]]; then ok "检索索引存在"; else bad "缺少检索索引：跑 bash scripts/setup.sh"; fi

echo "== 6. 评测权重 =="
if [[ -f "$MODEL/config.json" ]]; then
  ok "权重目录 ${MODEL}"
  du -sh "$MODEL" 2>/dev/null | awk '{printf "        大小 %s\n", $1}'
else
  bad "找不到 ${MODEL}/config.json：先 ls outputs/models/ 确认真实路径"
fi

echo "== 7. 端口 =="
# 商店环境：根路径 404 就是"活着"的正常信号。
env_code=$(curl -s -o /dev/null -m 3 -w '%{http_code}' "http://127.0.0.1:5700/" 2>/dev/null || echo 000)
if [[ "$env_code" == "000" ]]; then
  warn "商店环境未启动（:5700）：bash scripts/start_environment.sh"
else
  ok "商店环境在线（:5700 -> HTTP ${env_code}；404 是正常信号）"
fi

# 模型服务：只有 /v1/models 返回 OpenAI 格式才算真的就绪，端口能连不算。
model_body=$(curl -s -m 5 "http://127.0.0.1:8000/v1/models" 2>/dev/null || true)
if [[ -z "$model_body" ]]; then
  warn "模型服务未就绪（:8000/v1/models 无响应）：vLLM 初始化约 20-25 分钟，先看 tmux 里的日志"
elif [[ "$model_body" == *'"id"'* ]]; then
  ok "模型服务就绪：$(printf '%s' "$model_body" | tr -d '\n' | cut -c1-120)"
else
  warn "8000 端口有响应但不是 OpenAI 兼容接口：$(printf '%s' "$model_body" | tr -d '\n' | cut -c1-100)"
fi

echo
if (( FAIL )); then
  echo "结论：有 FAIL 项，先按上面提示修，不要直接起服务。"
  exit 1
fi
echo "结论：自检通过，可以起服务。"
