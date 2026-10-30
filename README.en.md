# Shopping GRPO

<div align="center">

**English** · [简体中文](README.md)

<br />

Reproducible post-training and evaluation for long-horizon shopping agents

<br />

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![LoRA SFT](https://img.shields.io/badge/Post--training-LoRA%20SFT-7B61FF)](docs/sft.md)
[![veRL](https://img.shields.io/badge/veRL-0.8.0-0E8A16)](https://github.com/verl-project/verl)
[![ShopSimulator](https://img.shields.io/badge/Environment-ShopSimulator%20v2.1-4C78A8)](https://arxiv.org/pdf/2601.18225)
[![Benchmark](https://img.shields.io/badge/Benchmark-Final--200--Clean-F59E0B)](docs/evaluation-dataset.md)

<br />

Teacher rollouts and LoRA SFT → online GRPO with veRL → auditable comparison on
a frozen benchmark

<br />

**Current result: Qwen3.5-2B + LoRA SFT + GRPO = 128 / 200 (64.0%)**
　·　**+64 percentage points** over the untrained 2B baseline (0.0%)
　·　**+1 task** over SFT alone (63.5%), with **zero regression** — the set GRPO
solves is a strict superset of the set SFT solves

</div>

> **This repository is a fork of [YYHDBL/shopping-grpo-longhorizon](https://github.com/YYHDBL/shopping-grpo-longhorizon).**
> The upstream author built the codebase, data protocol, Reward v3 and the Final-200 Clean
> benchmark; this fork adds twelve stages (M1–M12) of training-recipe optimisation and
> per-task evaluation. See [project history](docs/project-history.md) for results and
> negative findings. Upstream attribution is at the end of this file.

![Shopping GRPO project overview](docs/images/project-overview-pipeline.png)

## What is ShopSimulator?

[ShopSimulator](https://arxiv.org/pdf/2601.18225) is a large-scale Chinese
shopping environment for evaluating long-horizon LLM agents. A task describes
what a user wants—including category, budget, brand, model, functions and
product options—but the agent must discover the right item through interaction.

In this project the agent can search products, open candidates, inspect details,
select variants, buy, or stop when no acceptable item can be verified. Success
therefore requires more than producing a plausible answer: the agent must gather
evidence, obey constraints, choose the correct variant and terminate correctly.

The frozen Environment v2.1 source and product archive are embedded under
[`environments/ShopSimulator/`](environments/ShopSimulator/), so this repository
does not depend on a separately running third-party project.

![ShopSimulator overview](docs/images/shopsimulator-overview.png)

## The four stages

| Stage | What happens | Entry point | Details |
|---|---|---|---|
| Baseline | Evaluate the untouched base model | `bash scripts/baseline.sh` | [Evaluation](docs/evaluation.md) |
| SFT | Learn tool use from accepted teacher trajectories | `bash scripts/sft_curriculum.sh` | [SFT](docs/sft.md) |
| GRPO | Optimize terminal Reward v3 with online rollouts | `bash scripts/grpo.sh` | [GRPO](docs/grpo.md) |
| Evaluation | Run the curated Final-200 Clean protocol | `bash scripts/evaluate.sh NAME` | [Evaluation](docs/evaluation.md) |

The checked-in SFT data was produced by a separate collection stage documented
in [Data collection](docs/data-collection.md). The custom constraint-aware
reward is specified in [Reward v3](docs/reward-v3.md).

```mermaid
flowchart LR
    A[Teacher rollouts] --> B[Reward v3 filtering]
    B --> C[Action-only SFT data]
    C --> D[LoRA SFT]
    D --> E[Online GRPO with veRL]
    F[Frozen ShopSimulator v2.1] --> E
    G[Final-200 Clean held-out tasks] --> H[Shared evaluation pipeline]
    I[Baseline] --> H
    D --> H
    E --> H
```

### How the SFT data was collected

The current collection used `deepseek-v4-flash` as a teacher (official alias, routed to
**DeepSeek-V4.1-Flash**) in ShopSimulator
Environment v2.1. It produced 2,498 raw trajectories, of which 1,026 passed the
strict acceptance filter. This project then ran two data-rebuild rounds (see
[project history](docs/project-history.md)), ending with a **1,210-row pool**
(1,089 train / 121 validation), of which **144 rows (11.9%) contain `next_page`
paging demonstrations** — the main lever that moved SFT from 60.5% to 63.5%.
SFT, GRPO and Final-200 Clean task IDs are pairwise disjoint. Dataset hashes and
the audit are in [Data collection](docs/data-collection.md).

The resumable collection entry point is:

```bash
python scripts/collect_sft_data.py \
  --tasks data/grpo/train.jsonl \
  --output-dir outputs/sft-collection \
  --target-accepted 1000 \
  --workers 4
```

### How GRPO is trained

GRPO starts from the merged SFT model. veRL generates four online trajectories
per prompt in ShopSimulator, while deterministic Reward v3 scores the terminal
purchase, constraint satisfaction and termination behavior. No additional
LLM-as-a-Judge reward model is used for training.

The repository pins `verl==0.8.0` instead of copying its source. It keeps only
the project-specific AgentLoop, tool adapter, runtime compatibility code and a
small SHA-256-checked patch. See the [GRPO guide](docs/grpo.md) for details.

### How evaluation works

Formal evaluation combines deterministic checks with two LLM-as-Judge roles.
DeepSeek V4 Flash curates a frozen requirement Rubric from code-generated
category, brand, model, function, option and price candidates. It may select and
deduplicate candidates, but cannot invent fields or expected values. The same
Rubric is shared by Baseline, SFT and GRPO.

After each Actor completes a rollout, code normalizes events and computes
Reward, legality, repetition, context and infrastructure checks. Valid
trajectories then go to DeepSeek V4 Pro with the original Query, frozen Rubric,
Actor-visible trajectory, neutral terminal flags and allowlisted behavioral
metrics. Reward values, hidden Gold fields, raw observations, success labels and
other models' results are excluded.

```mermaid
flowchart TD
    A[Benchmark task ID] --> B[Private TaskFacts]
    B --> C[Code-generated candidates]
    C --> D[V4 Flash frozen Rubric]
    A --> E[Actor rollout]
    E --> F[Normalization and hard checks]
    F -->|valid| G[Judge-safe payload]
    D --> G
    G --> H[V4 Pro requirement and five-dimension judgment]
    F -->|infrastructure invalid| I[not_judged]
    H --> J[Four-panel aggregation]
    I --> J
    J --> K[Paired Baseline / SFT / GRPO comparison]
```

V4 Pro scores Search Strategy, Candidate Utilization, Evidence Verification,
Decision Quality and Termination Efficiency independently on a 0/1/2 scale. It
also assesses each Rubric and assigns errors from a frozen taxonomy. The final
report keeps Reward/terminal, requirement Rubric, trajectory quality and
deterministic behavior as four separate panels—there is no composite score.
Failed, missing and not-judged tasks remain in the Final-200 Clean denominator. The
[evaluation guide](docs/evaluation.md) contains the complete prompts, one
worked benchmark example, input-isolation rules and aggregation contract. An
historical static view is available in the [Final-200 Benchmark Dashboard](docs/evaluation-dashboard.html); the current task set is documented in [Final-200 Clean](docs/evaluation-dataset.md).

> **Reserved figure — Training and evaluation pipeline.** A full-width diagram
> showing teacher data collection, LoRA SFT, online GRPO rollouts and the shared
> held-out evaluation path, with artifacts produced at each boundary.

## Differences from upstream: what changed and what did not

This repository is a fork of
[YYHDBL/shopping-grpo-longhorizon](https://github.com/YYHDBL/shopping-grpo-longhorizon),
diverging at upstream `449bba2` (2026-09-22). **Upstream already ships a complete training
recipe** — a three-stage SFT curriculum, GRPO dynamic sampling, Reward v3 and the Final-200
Clean benchmark. The work here is incremental on top of that recipe, not a rebuild.

Every row below can be checked against the upstream files.

### Teacher model (unchanged)

| Item | Value | Upstream source |
|---|---|---|
| Model | `deepseek-v4-flash` (official alias, routed to **DeepSeek-V4.1-Flash**) | `docs/data-collection.md` |
| Endpoint | `https://api.deepseek.com/v1` via `OPENAI_BASE_URL` / `OPENAI_API_KEY` | same |
| Sampling temperature | 0.0 | same |
| Acceptance | every trajectory really executes environment actions, then is strictly accepted on the Reward v3 terminal result | same |
| Yield | 2,498 raw trajectories → 1,026 accepted (41.1%) | same |

> Collection must use the literal `deepseek-v4-flash`, not the current official name
> `deepseek-flash`: the client prefix-matches `deepseek-v4` to decide whether to send
> `thinking: disabled`. Renaming silently falls back to the default thinking mode, which drops
> `reasoning_content` across turns and breaks tool calling.

### Not changed

| Category | Content | Upstream source |
|---|---|---|
| Environment | ShopSimulator Environment v2.1 source and the 23,421-product corpus | `environments/ShopSimulator/` |
| Protocol | tool schema v2, observation v2, frozen `SYSTEM_PROMPT` | `configs/tools.json`, `configs/prompts/` |
| Benchmark | `data/evaluation/tasks.jsonl`, 200 tasks, one deterministic rollout each | same file upstream |
| Evaluation settings | temperature 0.0, `max_steps=35`, top-20 rendered results, BM25 recall 150 | `docs/evaluation.md` |
| **SFT three-stage curriculum** | A→B→C, 1 epoch per stage, lr **1e-4 → 7e-5 → 5e-5**, LoRA re-opened on the previous stage's merged weights | `data/sft_curriculum/manifest.json`, `docs/sft.md` |
| **SFT structural hyperparameters** | LoRA r=16 / α=32 / dropout 0.05, `max_length=24576`, batch 1 × grad-accum 8, SDPA, gradient checkpointing | `docs/sft.md` |
| **GRPO group size and sampling** | `n=4`, rollout temperature 0.7 / top-p 0.9 | `configs/grpo.yaml`, `docs/grpo.md` |
| Reward decision logic | hard gates (category + budget), `0.35/0.25/0.25/0.15` weighting, discrete terminal table | `docs/reward-v3.md` |

**The three-stage curriculum, the LoRA structure and the group size are all upstream designs;
this fork did not change them.**

### What changed

**1. Reward: v3 → v4 (M3); v5 written but never enabled (M7)**

| Version | Change | Effect on Final-200 score |
|---|---|---|
| **v4** | ① Budget parser rewritten: Chinese numerals (including elided forms like 两千五), colloquial units (块/块钱/圆), unprefixed ranges, 万 abbreviations, full-width and `\xa0` normalisation; price-context gating rejects false hits such as `SPF50+`, `6g`, `60厘米`. ② Two **information-only layers**: soft price proximity and process evidence `shopping-process-evidence-v1` | **None.** Hard-gate semantics are byte-identical; both layers only write to `evidence` and never enter the terminal decision. Budget parse rate on the benchmark is 93.5% |
| **v5** | `wrong_purchase` from a constant −0.85 to the band `[−0.85, −0.66]`, varying with `match_score` × budget proximity | **None, and never used for a final model**: Gate 3 measured a gradient share of 0.3% against a 5% threshold, so M7 was closed |

In one line: **we changed whether budgets parse correctly and whether the training signal is
fine-grained enough — not what counts as a correct purchase.**

**2. SFT: larger pool, new supervision scope, extra final leg for the champion**

The curriculum and its hyperparameters follow upstream; what changed is which rows are read
and which tokens are supervised:

| Item | Upstream | This fork |
|---|---|---|
| Main pool `data/sft_pure_v4/all.jsonl` | **1,192 rows** | **1,252 rows** (M2 top-up collection + M4 rebuild) |
| Manifest split | train 1,073 / val 119 | train 1,126 / val 126 |
| Three-stage curriculum | A→B→C, 1 epoch each, lr 1e-4→7e-5→5e-5 | **identical** (epochs/lr in the manifest untouched) |
| Behavioural coverage | dominated by "search once → open the first → buy" (89.2%) | M10-R adds 144 rows (11.9%) of `next_page` demonstrations, plus returning to an earlier candidate and rejecting over-budget items; pool 1,210 rows (1,089 / 121) |
| Loss scope | global assistant (natural-language `content` also supervised) | new per-turn `--loss-scope` switch; the champion uses `tool-call-only` (M5 ablation: 62.0% vs 59.0%) |
| Champion's final leg | curriculum ends at stage C | M10-R v2 adds **1 epoch at lr 5e-5** on top of `stage-c/merged`, reaching 63.5% |

**3. GRPO: new training pool plus six overrides**

Upstream `configs/grpo.yaml` and `docs/grpo.md` agree on the values, so the "Upstream" column
below is the real value in both places:

| Parameter | Upstream | Used here | Reason |
|---|---|---|---|
| Training pool `data/grpo/train.jsonl` | **1,000 tasks** | 906 after the M4 rebuild → M12 **223 medium-difficulty tasks** (1–3 successes out of 4 rollouts) | all-correct / all-wrong groups have zero within-group advantage and only burn compute |
| `train_batch_size` | 2 | **4** | both feed 8 sequences per update (4×4÷2 = 2×4÷1); equivalent learning, but 4 is faster (189 vs 227 s/update) |
| `max_response_length` | 20480 | **16384** | measured `response_length/max = 7500`, never saturated; saves ~25% memory |
| `optim.lr` | **1e-6** | **1e-5** | 1e-6 does not move the model (M6 was a zero result); at 1e-5 `ppo_kl` rises monotonically from 0.0038 to 0.0055 |
| `clip_ratio_high` | 0.20 | **0.28** | asymmetric with `clip_ratio_low`, loosening positive updates |
| `calculate_entropy` | true (logging only, not in the loss) | **false** | under bypass it both OOMs and never emits the metric |
| `total_training_steps` | 500 | **50** | 223 tasks × 50 steps × batch 4 = 200 prompt-passes ≈ 0.9 epochs |
| `save_freq` / `test_freq` | 50 / 50 | **25 / 25** | 50 steps only yields two checkpoints; 25-step granularity makes the mid-run validation visible |
| `rollout.n` (group size) | 4 | **4 (unchanged)** | changing it changes the algorithm itself |

**4. Evaluation harness: upstream has only a single-process entry point**

| Script | Purpose |
|---|---|
| `scripts/eval_sharded.sh` | shards the 200 tasks across 8 workers; resumes from completed tasks; merges → report → **per-task diff against a named baseline** |
| `scripts/compare_teacher_vs_ours.py` | per-task comparison of any two `summary.json` files, emitting the gap list and behaviour contrast |
| `scripts/label_pool_difficulty.py` | labels a pool with "how many of N rollouts succeeded" and keeps only medium-difficulty tasks |
| `scripts/build_candidate_pool.py` | samples unused corpus rows while enforcing zero task-id overlap with SFT / GRPO / benchmark |

Three guards live in the scripts rather than in prose: subtracting from a historical baseline
is **refused** when temperature is non-zero or the run is not 200 tasks; `LABEL=` must be
explicit (otherwise a champion directory gets overwritten); `BASELINE_SUMMARY=` must be
explicit (auto-selection picks the highest-scoring directory, which may not be the intended one).

### Comparability: what can and cannot be compared

**Final scores are comparable.** Benchmark, environment, tools, system prompt, temperature and
step cap are all frozen, with one deterministic rollout per task — the same exam paper.

**Per-change attribution is not.** Upstream's 60.5% → 62.0% and this fork's 62.0% → 64.0%
differ in **several** variables (SFT pool, GRPO pool, reward version), so it is not valid to
claim "our GRPO is 2 points better than the author's". Attribution inside this fork is
closed-loop: each stage moves one variable under the same reward.

| Model | SFT pool | GRPO pool | Reward | Strict success |
|---|---|---|---|---:|
| Qwen3.5-2B untrained | — | — | v3 | 0.0% |
| **Upstream** LoRA SFT | 1,192 | — | v3 | 60.5% |
| **Upstream** GRPO step 100 | 1,192 | 1,000 | v3 | 62.0% |
| This fork M5-B (three-stage curriculum, loss=tool-call-only) | 1,252 | — | v3+v4 | 62.0% |
| **This fork M10-R v2 (best SFT)** | 1,210 | — | v4 | **63.5%** |
| **This fork M12 (SFT + GRPO)** | 1,210 | 223 medium | v4 | **64.0%** |
| Qwen3.8-27B (reference, not produced here) | — | — | v3 | 73.0% |

End-to-end gain is **+2.0pp / +4 tasks** (62.0% → 64.0%), which as noted is not a
single-variable result. **The only attributable increment is M12's per-task 1-in / 0-out**,
described in the next section.

### Test history

Twelve stages (M1 solvability audit → M12 medium-difficulty GRPO); the per-stage
"what was done / result / artifacts" table is in the Chinese README and in
[project history §2](docs/project-history.md). **Eight of them are negative results**
(M6 zero gain, M6-R divergence, M7 Gate 3 failure, M8 target invalidated offline,
M9 direction rejected, M10-R v3 regression, M10-H hard-pool idling, M11 query A/B
cancelling 5-for-5) — those carry more information than the scores.

---

## Results

All numbers are on **Final-200 Clean**: 200 held-out tasks, one deterministic
rollout per task (temperature 0). Strict success requires a `gold_purchase`
terminal state. Full protocol and artifact hashes:
[evaluation update log](docs/evaluation-updates.md).

### Main line

| Model | Strict success | Δ vs previous | Artifact |
|---|---:|---:|---|
| Qwen3.5-2B baseline (no post-training) | 0.0% | — | — |
| + LoRA SFT (initial) | 60.5% | +60.5pp | `experiments/sft/` |
| + LoRA SFT (behavior-coverage rebuild, M10-R v2) | **63.5% (127/200)** | +3.0pp | `sft-m10r/stage-full/merged` |
| **+ GRPO (medium-difficulty pool, M12, 50 steps)** | **64.0% (128/200)** | **+0.5pp** | `grpo-m12-merged` |
| `deepseek-v4-flash` (teacher, reference ceiling) | 73.5% (147/200) | — | — |
| Qwen3.8-27B (reference, not produced here) | 73.0% | — | — |

### Per-task delta of GRPO — the core finding

| Item | Count |
|---|---:|
| GRPO correct | **128** |
| SFT baseline correct | 127 |
| Both correct | 127 |
| **Only GRPO correct** | **1** |
| **Only SFT correct** | **0** |

**The 128 tasks GRPO solves are a strict superset of the 127 the SFT model
solves** — zero regression, one net gain. This is the first non-zero delta across
twelve stages. The three earlier GRPO attempts either tied (M6), crashed at step
48 (M6-R), or lost one task (M11 probe, whose 126 successes were a strict
*subset* of the baseline's 127 — the opposite direction).

> **Honest caveat**: +1 task sits inside binomial noise (σ ≈ 6.9 tasks). At the
> aggregate level 64.0% and 63.5% are indistinguishable. The defensible claim is
> "unique successes went from 0 to 1", not "success rate improved by 0.5pp".
> Statistical separation would require ≥134 tasks (+7).

### Where the 0.5pp came from

The first two GRPO rounds spun in place for a measurable reason: the training log
showed `group/all_equal_ratio` **0.25–0.50** and `no_purchase_success_ratio`
**0.375–0.75** — a quarter to a half of all groups had four identical rewards
(all correct or all wrong), so the within-group advantage was identically zero.
Those prompts burned compute without producing a gradient direction.

M12 changed **only the training data**: two sampling rounds labeled each
candidate task by "how many of 4 rollouts succeed", and only the
**medium-difficulty** ones (1–3 of 4, 223 tasks) were kept. Every other
hyperparameter stayed fixed.

| Metric | Before | After |
|---|---:|---:|
| `no_purchase_success_ratio` (all-wrong groups) | 0.375–0.75 | **0.00** |
| `all_equal_ratio` (zero-gradient groups) | 0.25–0.50 | 0.25 |
| Final-200 | 127 | **128** |

**All-wrong groups were eliminated entirely** — direct evidence that the pool
swap worked. The residual 0.25 are all *all-correct* groups: medium-difficulty
tasks degrade into easy ones during training. That is the structural ceiling of
this route.

### Training reward does rise — but that is fitting, not capability

This is the most easily misread part of the project, so it gets its own section.
Reward metrics **rise sharply** during training:

| Metric | step 25 | step 50 | Change |
|---|---:|---:|---:|
| `reward/shaped_mean` (= `critic/score/mean`) | 0.177 | **0.356** | **+101%** |
| `reward/purchase_success_rate` (training set) | 0.500 | **0.563** | +12.5% rel. |
| `reward/strict_mean` | 0.500 | 0.563 | +12.5% rel. |
| val mean@1 (49 held-out tasks) | 0.4477 | **0.4972** | +11.1% rel. |
| **Final-200 (the actual exam)** | — | **127 → 128** | **+0.8% rel.** |

**Three levels of decay, each an order of magnitude**: training reward +101% →
held-out val +11% → real exam +0.8%.

Why, level by level:

1. **`shaped_mean` doubling is mostly *not* "more tasks solved"** — it is the
   disappearance of the **−0.85 all-wrong groups**. A wrong purchase costs
   −0.85; with all-wrong groups falling from 37%–75% to zero, the mean is
   mechanically lifted. That is a structural change, not capability growth.
2. **Training-set purchase success 0.5 → 0.563 (+6.3pp) is a real improvement**,
   but only on the distribution of **223 medium-difficulty tasks**.
3. **Only +1 task transfers to Final-200.** Those 223 tasks were selected by
   labelling with the *then-current* model; learning them well is partly "learn
   the tasks that were labelled learnable", not "get better on unseen tasks".

⇒ **Reward is a training signal, not a capability measure.** That is why every
conclusion in this project rests on Final-200 alone — not on training reward and
not on val (the M11 probe gained +12.3% on val while losing one task on Final-200).

### Paths tried and closed

| Direction | Result |
|---|---|
| Change GRPO recipe (lr 5e-6 → 2e-5) | val below start, circuit-breaker at step 48 |
| Force "open k candidates before buying" at inference | same 9 tasks **7/9 → 0/9** (p≈0.016), direction rejected |
| Teach paging (144 synthetic demonstrations) | 63.5%, but only 7 of 21 paging tasks actually paged; per-turn rationales dropped it to 61.5% |
| Rewrite search queries via system prompt | 127/200, net 0 (5 gained, 5 lost) |
| Hard-task pool (201 tasks with gold beyond rank 20) | 63.5%, only 2 tasks changed per-task ⇒ no signal |
| Reward v5 (`wrong_purchase` as a band) | Gate 3 gradient share 0.3% ≪ 5% threshold, never trained |

Full attribution and interpretation rules: [`docs/project-history.md`](docs/project-history.md)
and [`docs/lessons-learned.md`](docs/lessons-learned.md). Machine-readable
training configs and result summaries live in [`experiments/`](experiments/).
These are reported results, not a promise that different hardware or dependency
versions will produce bit-identical training.

## Training hardware and time

### GRPO (veRL 0.8, single GPU)

**GPU memory is the hard constraint.** Measured on M12 (50 steps):

| Item | Measured |
|---|---:|
| `actor/perf/max_memory_allocated_gb` | **41.3 GB** |
| `actor/perf/max_memory_reserved_gb` | **57.1 GB** |

⇒ **A 48 GB card does not fit** (47.9 GB usable). A previous 48 GB smoke run peaked
at 47,641 / 48,507 = 98.2% and crashed in `logprobs_from_logits → F.log_softmax`
(the 248,320-row vocabulary output layer is the culprit). **Minimum 64 GB,
recommended 80 GB+**, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is
mandatory.

| Stage | Time |
|---|---:|
| vLLM initialisation (GPU util stays at 0 — this is normal) | 20–25 min |
| Per step (`timing_s/step`) | **140–149 s** |
| ├ `update_actor` | **103–105 s** |
| ├ `gen` (rollout) | 12–19 s |
| └ `save_checkpoint` | 17 s |
| **50 steps including initialisation** | **2 h 23 min** |

**The wall-clock bottleneck is the actor update, not rollout** — shortening
responses or raising the batch size saves little.

### Evaluation (run separately)

A 2B model needs far less memory to evaluate than to train; **a 48 GB card is
enough**. One hard constraint: **do not use a Blackwell / sm_120 card** (e.g.
RTX 6000D). With torch CUDA < 12.9, `vllm serve` fails with
`FlashInfer requires sm75 or higher`. This project therefore splits training and
evaluation across two machines.

| Stage | Time |
|---|---:|
| vLLM loading the merged weights | 20–25 min |
| Final-200 evaluation (8 shards) | **~25 min** |
| Merging an FSDP checkpoint (11 GB → 4.2 GB weights) | ~15 min |

### SFT (LoRA)

| Item | Measured |
|---|---:|
| Per step | ~74 s |
| Total steps | 133 (1,061 kept examples ÷ batch 8) |
| Full training | ~2.7 h |

### Other stages

| Stage | Estimated time |
|---|---:|
| Teacher collection (2,498 raw trajectories) | Depends on endpoint concurrency and rate limits |
| Difficulty labelling for 1,000 tasks (2 rounds, 8-way) | **~4 h** |
| Final-200 Clean evaluation (Base) | ~20 min |
| Final-200 Clean evaluation (SFT/GRPO) | ~25 min |
| LLM Judge scoring for 200 trajectories | ~30–60 min |

## Requirements

- Linux with an NVIDIA GPU and a compatible CUDA driver;
- [`uv`](https://docs.astral.sh/uv/);
- about **60 GB** of free disk for environments, weights and generated artifacts
  (**each GRPO checkpoint is ~11 GB**);
- **SFT**: 48 GB of GPU memory is enough;
- **GRPO**: minimum **64 GB**, recommended **80 GB+**
  (measured `allocated 41.3 GB` / `reserved 57.1 GB`);
- **Evaluation**: 48 GB is enough, but **do not use a Blackwell / sm_120 card**
  (torch CUDA < 12.9 makes `vllm serve` fail with `FlashInfer requires sm75 or higher`).

The main environment uses Python 3.12. ShopSimulator is isolated on Python 3.10.
`uv` creates both environments. veRL is **installed as the pinned
`verl==0.8.0` dependency**; its source is not copied into this repository. Only
the Shopping Agent adapter and a small version-checked patch live here.

## Quick start

Run every command from the repository root.

### 1. Install

```bash
bash scripts/setup.sh
```

This installs the pinned SFT and GRPO dependencies, creates the isolated
ShopSimulator environment, verifies and expands the product archive, builds the
search index and applies the version-checked veRL patch.

### 2. Start ShopSimulator

Keep this terminal running:

```bash
bash scripts/start_environment.sh
```

The service listens on `http://127.0.0.1:5700`.

### 3. Evaluate the baseline

Start the base model server in a second terminal:

```bash
bash scripts/serve_model.sh Qwen/Qwen3.5-2B
```

Evaluate it in a third terminal:

```bash
bash scripts/baseline.sh
```

Stop the model server before training so it releases the GPU.

### 4. Train and evaluate SFT

The main path is the three-stage curriculum (A→B→C, 1 epoch each, lr 1e-4→7e-5→5e-5):

```bash
# print the six train/merge commands without loading a model
bash scripts/sft_curriculum.sh --dry-run

# actual training
bash scripts/sft_curriculum.sh

bash scripts/serve_model.sh outputs/models/sft-curriculum/stage-c/merged
bash scripts/evaluate.sh sft
```

`scripts/sft.sh` is the single-stage entry point (`data/sft/` 1,000 rows, 3 epochs, lr 1e-4).
It exists only to reproduce the historical 60.5% baseline and is not the main path.

Stop the model server again before GRPO.

### 5. Train GRPO

First inspect the fully resolved launcher without starting CUDA or Ray:

```bash
bash scripts/grpo.sh --dry-run
```

Then train:

```bash
bash scripts/grpo.sh
```

Choose a checkpoint using validation metrics and export its actor:

```bash
bash scripts/export_grpo.sh \
  outputs/models/grpo/global_step_100/actor \
  outputs/models/grpo-merged
```

Evaluate it:

```bash
bash scripts/serve_model.sh outputs/models/grpo-merged
bash scripts/evaluate.sh grpo
```

Generated checkpoints, rollouts and logs are written under `outputs/`, which is
ignored by Git.

## Reward design (v3 → v4)

The decision rule is a deterministic terminal reward; it does not rely on another
language model for subjective judgment:

- category and budget are hard gates;
- brand, model, core functions and key options use weights of
  `0.35 / 0.25 / 0.25 / 0.15`;
- an exact target purchase with full satisfaction receives `1.0`;
- a fully satisfying alternative item receives `0.55`;
- partial satisfaction receives a continuous score capped at `0.25`;
- wrong purchases, premature abstention, repeat loops and maximum-step
  termination receive distinct negative rewards;
- insufficient evidence sets `reward_valid=false`, rather than being treated as
  a valid neutral zero.

![Reward V3 decision rules](docs/images/reward-v3-decision-rules.png)

**This fork made exactly two changes to the reward, neither touching the decision above:**

| Version | Change | Effect on scores |
|---|---|---|
| v4 (M3) | budget parser rewritten (Chinese numerals, colloquial units, 万 abbreviations, full-width whitespace, price-context gating) plus two information-only layers (soft price proximity, process evidence) | none — hard gates are byte-identical, the new layers only write to `evidence` |
| v5 (M7) | `wrong_purchase` moved from a constant −0.85 to the `[−0.85, −0.66]` band | none, and never enabled (Gate 3 gradient share 0.3% < 5%) |

The complete formula, termination rules and evidence requirements are in the
[Reward v3 design guide](docs/reward-v3.md); the rationale for each change is in
[Reward v4](docs/reward-v4.md) and [Reward v5](docs/reward-v5.md).

## Repository map

```text
configs/                         current GRPO, AgentLoop and tool configuration
data/
  sft/                           SFT trajectories (1,026-row base pool, rebuilt here to 1,210: 1,089 / 121)
  grpo/                          ready-to-train JSONL and veRL Parquet
  evaluation/                    curated Final-200 Clean held-out set
docs/                            data, SFT, GRPO, evaluation and Reward documentation
environments/ShopSimulator/      embedded environment and product archive
experiments/
  baseline/                      baseline config and result summary
  sft/                           SFT config and result summary
  grpo/                          GRPO config and result summary
scripts/                         thin user-facing entry points
src/shopping_grpo/
  collection/                    Teacher acceptance and SFT data construction
  environment/                   HTTP client, tools, actions and observations
  training/sft/                  SFT dataset masking and collation
  training/grpo/                 veRL adapter, compatibility and sampling logic
  evaluation/                    hard checks, Rubrics, trajectory Judge and metrics
tests/                           focused unit, launcher and packaging checks
```

The project keeps focused checks for the CPU smoke path, offline trajectory
evaluation, GRPO launcher arguments, non-editable wheel installation, Reward
aggregation and the frozen environment manifest. Cleanup does not mean deleting
tests that protect the public workflow.

## Configuration

Most users only need these environment variables:

| Variable | Default |
|---|---|
| `BASE_MODEL` | `Qwen/Qwen3.5-2B` |
| `SHOPSIM_BASE_URL` | `http://127.0.0.1:5700` |
| `LLM_BASE_URL` | `http://127.0.0.1:8000/v1` |
| `SERVED_MODEL_NAME` | `shopping-agent` |
| `SFT_ADAPTER_DIR` | `outputs/models/sft-lora` |
| `SFT_MERGED_DIR` | `outputs/models/sft-merged` |

Advanced GRPO overrides can be appended after `--`:

```bash
bash scripts/grpo.sh -- \
  trainer.total_training_steps=20 \
  trainer.save_freq=10
```

SwanLab logging is opt-in:

```bash
export SWANLAB_API_KEY=...
bash scripts/grpo.sh --logger swanlab
```

## Documentation

**Read these two first**:

- [Project history and issue ledger](docs/project-history.md) — timeline, results
  across all twelve stages, negative-result attribution, evaluation protocol
- [Lessons learned](docs/lessons-learned.md) — closed paths, interpretation rules,
  hard constraints, engineering pitfalls

Stage documents:

- [Data collection and dataset provenance](docs/data-collection.md)
- [LoRA SFT](docs/sft.md)
- [GRPO with veRL](docs/grpo.md)
- [Held-out evaluation](docs/evaluation.md)
- [Final-200 Clean evaluation dataset](docs/evaluation-dataset.md)
- [Final-200 Benchmark Dashboard (historical)](docs/evaluation-dashboard.html)
- [Reward v3 design](docs/reward-v3.md)
- [Auditable experiment results](experiments/comparison.md)

## Future improvement plan

### 1. Do attribution before adding training compute (zero GPU cost)

The teacher `deepseek-v4-flash` scores **147/200**, we score 128/200, and the
**union of both is 165/200 = 82.5%** — this is complementary blindness, not a
uniform deficit. The real headroom is the **38 tasks the teacher solves and
we do not**:

| Failure mode | Tasks |
|---|---:|
| A Correct product never appears in search results | 13 |
| B Product appears but is never opened | 15 |
| **C Opened but bought wrong** | **0** |
| D Right product, wrong variant | 9 |
| E Never purchased | 1 |

**C = 0 means "which candidate to open" is the only decision point** — once the
right product is opened, the model does not mis-buy. The next step should be an
offline attribution pass over these 38 tasks (reads existing trajectories only,
no GPU), before deciding whether to change data or reward.

### 2. Pool expansion can work, but it is an arms race

**Why it could work (mechanism, not guesswork)**: after 0.9 epochs over the 223
tasks, `all_equal_ratio` stalls at **0.25**, and the residual is entirely
*all-correct* groups — medium-difficulty tasks degrade into easy ones. Adding
**new** medium-difficulty tasks pushes that ratio back down and lets more compute
produce a useful gradient.

**Why it could fail**: the "medium" label on new tasks is produced by the
*current* champion. After training, the model is stronger and those new tasks
degrade too. ⇒ Expansion is not a one-off fix; **every round needs fresh
labelling**, and the cost grows with each round.

**Real budget**: medium-difficulty tasks are only **14%–17%** of the candidate
pool, and labelling 1,000 tasks for 2 rounds takes **~4 hours**.
⇒ **200 new medium tasks = 1,200 labelled (4 h) + 4.7 h training ≈ 9 hours.**

**Where the ceiling is**: of the 72 tasks Final-200 still fails, only **38 are
solvable by the teacher** (`deepseek-v4-flash`); the other 34 it also fails.
⇒ **The realistic target of expansion is not 65%, but approaching
128 + 38 = 166/200 = 83%** (minus the ~20 unsolvable tasks in the benchmark, so
the true ceiling is lower).

**My read**: expansion is the only route with a mechanism behind it, but it is an
arms race — ~9 hours per 200 medium tasks buys one more round of useful gradient,
not a permanent gain. **Do the attribution in item 1 first**: determine whether
those 38 tasks fail because of missing training coverage or because of the 2B
capability limit, then decide whether expansion is worth funding.

### 3. Evaluation noise is the actual ceiling

The binomial noise on 200 tasks is **σ ≈ 6.9 tasks**, which means **any
improvement smaller than 7 tasks cannot be statistically confirmed**. The last two
rounds produced +1 and +3 tasks — both drowned in noise. **Expanding the
evaluation set (or averaging multiple rollouts per model) raises confidence more
than further fine-tuning does.**

### 4. Other directions

- **Scale up the model**: 2B has a structural disadvantage on tasks requiring
  product knowledge (e.g. deciding which specification word to add).
- **Finer-grained credit assignment**: terminal reward is sparse; key intermediate
  steps ("open the n-th candidate") receive no signal.
- **Mixed teachers**: a single teacher has limited behavioral coverage.

## Upstream project and acknowledgements

This repository is a fork of
[YYHDBL/shopping-grpo-longhorizon](https://github.com/YYHDBL/shopping-grpo-longhorizon).
The upstream project provides the infrastructure this work builds on: the ShopSimulator
environment wrapper, tool schema, data protocol, Reward v3, the Final-200 Clean benchmark
and the veRL training scaffold. **The changes here are confined to the post-training
recipe and the data pools** (SFT data reconstruction, a medium-difficulty GRPO pool,
evaluation scripts and per-task attribution tooling). Environment, reward and evaluation
protocol are untouched, so scores remain directly comparable across both repositories.

Upstream itself builds on the
[ShopSimulator paper](https://arxiv.org/pdf/2601.18225) and source project,
[veRL](https://github.com/verl-project/verl), and
[Qwen](https://github.com/QwenLM/Qwen3).

The evaluation protocol and Benchmark construction were also informed by
[VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-world Applications](https://arxiv.org/pdf/2509.26490)
and
[EComAgentBench: Benchmarking Shopping Agents on Long-Horizon Tasks with Distributed Hidden Intent](https://arxiv.org/pdf/2606.17698).

The repository organisation was informed by
[qiqihezh/agentic-grpo-longhorizon](https://github.com/qiqihezh/agentic-grpo-longhorizon).
Thanks to the [OpenCode Go plan](https://dev.opencode.ai/go) for supporting the upstream
development workflow.

### Contributors

**Upstream author**

<a href="https://github.com/Guochangwei917">
  <img src="https://github.com/Guochangwei917.png?size=80" width="80px;" alt="Guochangwei917"/>
</a>

**Maintained in this fork**

<a href="https://github.com/Jaxuu">
  <img src="https://github.com/Jaxuu.png?size=80" width="80px;" alt="Jaxuu"/>
</a>
