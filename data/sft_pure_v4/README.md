# Pure DeepSeek-v4 SFT pool

This directory contains the deduplicated union of the current DeepSeek-v4
teacher set and the new portion of the supplied `merged` package, plus the M2
high-ambiguity collection increment merged in M4-8.

- `all.jsonl` — **D_full**: 1,252 clean, unique-task, gold-purchase trajectories
  (1,152 pre-M4 + 100 from `data/collection/accepted.jsonl`). This is the
  canonical SFT training file.
- `all_toolonly.jsonl` — **D_toolonly**: identical rows and task ids, with every
  assistant `content` blanked so the assistant-turn loss supervises tool calls
  only. Built by `scripts/build_sft_variants.py` (M4-6).
- `variants_report.json`: D_full / D_toolonly hashes and the stripped-text counts.
- `difficulty_labels.jsonl`: intrinsic task difficulty and separate observed
  trajectory-complexity labels, **all 1,252 rows from `deepseek-v4-flash`**
  (`label_input_mode` = `full_trajectory` / `compact_tool_content`). The 100 M2
  rows were re-labelled by the teacher in M4 (49 batches, 0 failures, ~1.12M
  tokens). `scripts/label_sft_difficulty_offline.py` is kept only as a
  deterministic offline fallback for when no LLM endpoint is reachable.
- `duplicate_report.json`: source, quality features, and the selected/discarded
  row for every duplicate task.
- `metadata.json`: counts, hashes, labeling provenance, the M2 collection
  provenance, the guard-repair record, the final split, and mix feasibility.

The active SFT recipe keeps the natural difficulty mix. After M4 the observed
distribution is 22.8% simple / 66.5% medium / 10.8% hard; forcing 30% / 50% / 20%
would still discard valid rows merely because hard examples are scarce (pool-level
resampling was rejected in M4 — see `plans/m4-rebuild-report.md` §6). The
deterministic split and cumulative training stages live in
[`../sft_curriculum/`](../sft_curriculum/README.md).

Trajectories containing an action-guard rejection are excluded from this pool.
The M2 increment arrived with 41 such rows; M4 repaired them by deleting the
illegal `(assistant tool call, rejection observation)` pair — every rejection
observation states the action was *not executed*, so the environment state is
unchanged and the repaired path is a legal execution ending in `gold_purchase`.
