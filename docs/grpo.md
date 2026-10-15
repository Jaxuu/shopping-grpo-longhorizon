# GRPO with veRL

## Purpose

SFT teaches the action format and a strong initial policy. GRPO then samples
fresh trajectories in ShopSimulator and optimizes the terminal Reward signal.
The goal is to improve constraint satisfaction and termination behavior without
requiring a learned reward model.

## Integration boundary

veRL is installed from the pinned `verl==0.8.0` package. This repository does
not vendor the veRL source tree. Project-owned integration code lives in:

```text
src/shopping_grpo/training/grpo/
  adapter/              AgentLoop and ShopSimulator tools
  compat.py             narrow runtime compatibility hook
  dynamic_sampling.py   bounded non-zero-reward sampling
```

`scripts/setup.sh` applies one SHA-256-checked patch needed to connect the
bounded dynamic sampler to veRL 0.8.0. Setup fails rather than patching an
unknown veRL version.

## Inputs

- Initial policy: `outputs/models/sft-b-curriculum/stage-c/merged` (the M5-B
  curriculum endpoint). `scripts/train_grpo.py` still defaults to the older
  `outputs/models/sft-merged`, so **always pass `--model` explicitly**.
- Train set: `data/grpo/train.parquet` (906 tasks)
- Validation set: `data/grpo/validation.parquet` (49 tasks)
- Environment: ShopSimulator Environment v2.1
- Reward: `shopsimulator-reward-v4` at rollout time (see
  [`data/environment.json`](../data/environment.json)); the frozen task pool was
  collected under `shopsimulator-reward-v3` and consumers accept both

Task counts and hashes are recorded in
[`data/grpo/metadata.json`](../data/grpo/metadata.json).

## Run

Inspect the resolved command first:

```bash
bash scripts/grpo.sh \
  --model outputs/models/sft-b-curriculum/stage-c/merged \
  --output outputs/models/grpo \
  --dry-run
```

Train:

```bash
bash scripts/grpo.sh \
  --model outputs/models/sft-b-curriculum/stage-c/merged \
  --output outputs/models/grpo
```

Important defaults (see [`configs/grpo.yaml`](../configs/grpo.yaml)):

| Setting | Value |
|---|---|
| Algorithm | GRPO |
| Rollouts per prompt | 4 |
| Rollout temperature / top-p | 0.7 / 0.9 |
| Train / validation batch | 4 / 2 |
| Policy learning rate | `5e-6` |
| LoRA rank / alpha | 16 / 32 |
| Maximum response length | 16,384 |
| Maximum model length | 24,576 |
| Maximum training steps | 50 |
| Save / validation frequency | 25 / 25 |
| Clip ratio low / high | 0.20 / 0.28 |
| KL reward / KL loss | disabled / disabled |
| Policy entropy measurement | disabled (`entropy_coeff` is 0, so it only wasted wall-clock) |

`train_batch_size` (4), `clip_ratio_high` (0.28), `max_response_length` (16384),
`rollout.enforce_eager` (true) and `actor.calculate_entropy` (false) are **command-line
overrides**, not YAML values — the YAML keeps the original author's settings
(2 / 0.20 / 20480 / unset / true) so that `experiments.json`'s `grpo_baseline` still
describes the original recipe. Forgetting one of them silently runs a different
configuration; verify the `command` array in the audit JSON before letting the job
continue.

Two settings are load-bearing and must stay aligned:

- `trainer.save_freq` must be **≤ and divide** `trainer.test_freq`.
  Validation happens only on optimizer-updated steps, so a validation point with
  no matching `global_step_*` checkpoint is wasted.
- `trainer.test_freq` must be small enough to yield enough validation points to
  pick from. The shipped default (`25` over 50 steps) deliberately accepts the
  bare minimum of two points to keep the disk footprint at ~8.4 GB; if a finer
  curve is worth ~32 extra minutes of validation and ~21 GB, use `10 / 10`
  instead (see the table below).

The final step is always checkpointed and always validated: the sampling patch
gates both on `is_last_step or step % freq == 0`, where
`is_last_step = step >= trainer.total_training_steps`. So with
`total_training_steps=50` the run stops by itself and step 50 always has both a
`global_step_50/` checkpoint and a `val/*` score, whatever the frequencies are.

Each `global_step_N/` holds the **full model state dict**, not just the LoRA
adapter: `FSDPCheckpointManager` calls `model.state_dict()` without any LoRA
filtering, and `verl/model_merger` relies on that — it strips `.base_layer.` from
the remaining keys to recover the base weights, which only works because they are
present. For a 2B model in bf16 that is roughly **4 GB per checkpoint**, plus
~150 MB of Adam state for the LoRA parameters and a few MB of config/tokenizer
under `huggingface/` (not weights — `checkpoint.save_contents` defaults to
`[model, optimizer, extra]`, so no second HF copy is written).

`trainer.save_freq` therefore drives the disk footprint directly, and it must
divide `trainer.test_freq` (see above): with `test_freq=25` only `save_freq` of
1, 5, or 25 keeps every validation point backed by a checkpoint.

Nothing prunes old checkpoints (`trainer.max_actor_ckpt_to_keep` is unset), so
the footprint is simply `total_training_steps / save_freq` copies of ~4.2 GB:

| `save_freq` / `test_freq` | checkpoints kept | disk | validation points with a checkpoint |
|---|---|---|---|
| **25 / 25** (default) | 2 (step 25, 50) | **~8.4 GB** | 25, 50 |
| 10 / 10 | 5 | ~21 GB | 10, 20, 30, 40, 50 |
| 5 / 25 | 10 | ~42 GB | 25, 50 |

Note that `5 / 25` and `25 / 25` produce the **same** checkpoint-selection
candidates — the extra 34 GB only buys finer crash recovery, not a better pick.
A keep-limit is not a safe substitute for a larger `save_freq` either: it retains
the *most recent* N, so it would evict step 25 long before step 50 and checkpoint
selection would lose that candidate. Add ~4 GB for the exported merged model.

Resuming a finished or interrupted run appends to the same
`training_diagnostics.jsonl` (the writer opens it in append mode), so earlier
`val/*` records survive and checkpoint selection still sees the whole curve. Set
`trainer.total_training_steps` to the new, larger target and pass `--resume`
together with **exactly the same** overrides as the first segment. With the
default `save_freq=25` an interruption costs up to 24 steps of progress, since
the newest checkpoint may be that far behind.

Dynamic sampling can generate at most three batches to find a useful update and
permits at most ten consecutive skipped updates. These bounds prevent an
all-equal reward batch from causing an unbounded resampling loop.

Each run also appends `training_diagnostics.jsonl` under its output directory.
`generation_batch` records contain every generated rollout, its public tool
sequence, terminal result, reward breakdown, Guard rejection reasons and group
keep/drop decision. `optimizer_step` records preserve the scalar veRL metrics —
**including the `val/*` validation metrics** — as well as entropy, PPO KL, clip
fractions, response lengths and effective-group rates. `skipped_update` records
make zero-signal attempts visible even though they do not advance the optimizer
step.

Advanced overrides may be appended after `--`. Any change to `save_freq` /
`test_freq` must still satisfy `save_freq ≤ test_freq` and
`test_freq % save_freq == 0`:

```bash
bash scripts/grpo.sh -- \
  trainer.total_training_steps=20 \
  trainer.save_freq=10 \
  trainer.test_freq=10
```

## Checkpoint selection

The last training step is **not** assumed to be the best one. RL runs commonly
degrade late (entropy collapse, reward hacking, policy drift), so the exported
policy is the checkpoint with the best validation score.

Validation metrics live in `<output>/training_diagnostics.jsonl`; use
[`scripts/select_grpo_checkpoint.py`](../scripts/select_grpo_checkpoint.py) to
read them instead of eyeballing the console log:

```bash
# 1. Confirm which val/* keys this run actually produced
.venv/bin/python scripts/select_grpo_checkpoint.py outputs/models/grpo --list-metrics

# 2. Pick the best checkpoint
.venv/bin/python scripts/select_grpo_checkpoint.py outputs/models/grpo

# 3. Machine-readable, for scripts and reports
.venv/bin/python scripts/select_grpo_checkpoint.py outputs/models/grpo --json
```

The selector decides in three layers:

1. Primary metric within `best − 1/val_size` is treated as a tie. With 49
   validation tasks the resolution is one task, so a strict-equality tie test
   would be meaningless.
2. A tie is broken by the secondary metric — a continuous signal on the same
   validation split, finer-grained than a 0/1 success rate.
3. If the secondary metric also ties, the **later** step wins (more training).

**Never select on the frozen evaluation set.** That set is the exam; choosing a
checkpoint by its score leaks information and inflates the final number. The
validation pool is disjoint from both the training pool and the evaluation pool.

## Export

veRL checkpoints are not directly served by the evaluation launcher. Export the
actor directory that the selector printed:

```bash
bash scripts/export_grpo.sh \
  outputs/models/grpo/global_step_50/actor \
  outputs/models/grpo-merged
```

The selector prints this command with the chosen step filled in, so prefer
copying it from there over hand-writing it.
