#!/usr/bin/env bash
# 离线行为多样性探针入口（纯推理，不训练，单卡即可）。
#
#   bash scripts/probe_diversity.sh --make-task-list 50 --out-dir outputs/probe/t07
#   bash scripts/probe_diversity.sh --tasks outputs/probe/t07/tasks.jsonl \
#       --out-dir outputs/probe/t07 --attempts-per-task 8 --temperature 0.7 --workers 8
#   bash scripts/probe_diversity.sh --report-only --out-dir outputs/probe/t07
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec "$ROOT/.venv/bin/python" scripts/probe_group_diversity.py "$@"
