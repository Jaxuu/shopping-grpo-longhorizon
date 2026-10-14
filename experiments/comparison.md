# Baseline → SFT → GRPO

All three models were evaluated with one deterministic rollout on the same 200
held-out tasks.

| Model | Done | Strict success | Purchase success | Mean reward | Mean steps | Guard rejections |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B baseline | 18.0% | 0.0% | 0.0% | -0.1105 | 5.875 | 752 |
| LoRA SFT | 96.5% | 60.5% | 60.5% | 0.4729 | 12.335 | 52 |
| GRPO step 100 | 96.5% | 62.0% | 62.5% | 0.5158 | 11.850 | 38 |

## Interpretation

SFT provides the dominant gain. It teaches the base model to use the action
protocol, continue through multi-step shopping tasks and reach valid terminal
states. GRPO then adds a smaller improvement: three additional strict successes,
one valid alternative purchase, fewer wrong purchases, fewer loops and fewer
guard rejections than SFT.

The result supports a practical training order: first establish reliable tool
behavior with SFT, then use online RL for constraint satisfaction and policy
refinement.

## Limits

- Each model has one rollout per task, so the table does not estimate sampling
  variance.
- The SFT and GRPO recipes target large single GPUs; results may differ with
  other distributed layouts or dependency versions.
- Seven SFT and GRPO tasks ended without a Reward v3 terminal record and remain
  in the denominator.
- Full model weights and rollout logs are generated artifacts, not committed
  repository files.

Machine-readable settings and summaries are stored beside each stage:

- [`baseline/`](baseline/)
- [`sft/`](sft/)
- [`grpo/`](grpo/)
- [`sft-m5/`](sft-m5/)

## M5 SFT loss-scope ablation (2026-10-14)

Ablation for GRPO base selection: two loss scopes on the SFT Pure V4 pool
(1,115 train rows, 3-stage curriculum, LoRA r=16), re-evaluated with the
baseline on the same Final-200 Clean tasks, one deterministic rollout each,
Reward v3+v4 dual contract, code commit 3bf597a. Not directly comparable to
the archived table above (different data pool and reward consumer).

| Model | Loss scope | Done | Strict success | Purchase success | Mean reward | Mean steps | Guard rejections |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B baseline (re-run) | — | 22.0% | 0.0% | 0.0% | -0.1391 | 5.57 | 706 |
| M5-A (full / assistant) | assistant | 98.5% | 59.0% | 59.0% | 0.4359 | 8.07 | 17 |
| **M5-B (full / tool-call-only)** | tool-call-only | **100.0%** | **62.0%** | **62.0%** | **0.4943** | 8.41 | **3** |

Selection: M5-B wins on every reported metric and is chosen as the GRPO base.
The baseline re-run reproduces 0.0% strict/purchase success, confirming the
evaluation harness under the Reward v4 consumer. See [`sft-m5/`](sft-m5/)
for configs, summaries and limitations.
