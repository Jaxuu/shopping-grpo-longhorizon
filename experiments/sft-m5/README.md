# M5 SFT loss-scope ablation

M5 目的：在 SFT Pure V4 数据池上消融两种 loss 口径，为 GRPO 选出 base 模型。
三个模型于 2026-10-14 在同一套 Final-200 Clean 任务上各跑一次确定性 rollout
（Environment v2.1，Reward v3+v4 双口径消费端，代码 commit 3bf597a）。

| 模型 | Loss 口径 | Done | Strict success | Purchase success | Mean reward | Mean steps | Guard rejections |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B baseline（复测） | — | 22.0% | 0.0% | 0.0% | -0.1391 | 5.57 | 706 |
| M5-A（full / assistant） | assistant | 98.5% | 59.0% | 59.0% | 0.4359 | 8.07 | 17 |
| **M5-B（full / tool-call-only）** | tool-call-only | **100.0%** | **62.0%** | **62.0%** | **0.4943** | 8.41 | **3** |

## 结论

- **管线校验**：baseline 复测 strict/purchase 均为 0.0%，与 v3 口径的历史结果一致，
  mean reward 微小漂移（-0.139 vs -0.111）来自 Reward v4 消费端；评测链路可信。
- **选型**：M5-B 胜出，作为 GRPO base。配对比较 B 赢 18 题 / A 赢 11 题，
  且 B 在全部次要指标（reward、wrong_purchase 23 vs 30、guard 3 vs 17、
  done 100% vs 98.5%、repeat_loop 1% vs 2%）上一致占优。
- tool-call-only loss 的训练 loss 数值（~0.04）远低于 assistant 口径（~0.37），
  这是口径差异（仅对工具调用载荷计 loss）而非过拟合信号，评测结果已证实其泛化。

## 限制

- 每题仅一次确定性 rollout，无方差估计；3 个百分点的 strict 差距未做显著性检验，
  选型结论以多指标一致性支撑。
- 本表与 README 历史表口径不同：历史 SFT/GRPO 行使用旧数据池（448/800 条训练样本）
  与 Reward v3；M5 使用 SFT Pure V4 池（1,115 条）与 v3+v4 双口径，不直接互比。
- 完整轨迹与 HTML 报告为生成产物，保留在 outputs/（不入库）。

目录内容：

- `baseline/` 原模型复测 config 与 summary
- `sft-a/` assistant 口径配方 config 与 summary
- `sft-b/` tool-call-only 口径配方 config 与 summary（胜者）
