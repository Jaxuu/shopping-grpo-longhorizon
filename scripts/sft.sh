#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-2B}"
ADAPTER_DIR="${SFT_ADAPTER_DIR:-$ROOT/outputs/models/sft-lora}"
MERGED_DIR="${SFT_MERGED_DIR:-$ROOT/outputs/models/sft-merged}"

# M4 rebuild: the Pure V4 pool is data/sft_pure_v4/all.jsonl (1,252 rows). The
# legacy data/sft/ pool (1,000 rows) is no longer a valid training input.
# The train/validation split lives in the curriculum manifest, so both data flags
# point at the same file and the manifest selects the task ids (stage c = the
# whole pool). M5 overrides SFT_TRAIN to run the D_full / D_toolonly A/B.
TRAIN_DATA="${SFT_TRAIN:-data/sft_pure_v4/all.jsonl}"
CURRICULUM_MANIFEST="${SFT_CURRICULUM_MANIFEST:-data/sft_curriculum/manifest.json}"
CURRICULUM_STAGE="${SFT_CURRICULUM_STAGE:-c}"

cd "$ROOT"
"$ROOT/.venv/bin/python" scripts/train_lora_sft.py \
  --model "$BASE_MODEL" \
  --train "$TRAIN_DATA" \
  --validation "$TRAIN_DATA" \
  --curriculum-manifest "$CURRICULUM_MANIFEST" \
  --curriculum-stage "$CURRICULUM_STAGE" \
  --output "$ADAPTER_DIR" \
  --dtype auto \
  --gradient-checkpointing \
  --attention-implementation sdpa

exec "$ROOT/.venv/bin/python" scripts/merge_lora_adapter.py \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER_DIR" \
  --output "$MERGED_DIR" \
  --bf16
