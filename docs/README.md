# Documentation

**Start here for project history**: [project-history.md](project-history.md) — timeline,
problem ledger, reward version evolution, negative results and the metric conventions
used for judging experiments. It is the entry point for writing a project overview
or an archive report.

Follow the guides in workflow order:

1. [Data collection](data-collection.md) explains how the checked-in SFT data
   was produced and audited.
2. [SFT](sft.md) trains the first useful shopping agent.
3. [GRPO](grpo.md) improves that model with online environment reward.
4. [Evaluation](evaluation.md) compares baseline, SFT and GRPO fairly.
5. [Final-200 Clean evaluation dataset](evaluation-dataset.md) defines the current
   curated benchmark and its update record.

Reward specification, by version: [v3](reward-v3.md) (baseline contract) →
[v4](reward-v4.md) (parser coverage + informational layers) →
[v5](reward-v5.md) (current; `wrong_purchase` soft band).
