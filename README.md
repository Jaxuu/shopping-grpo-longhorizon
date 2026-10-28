# Shopping GRPO

<div align="center">

**简体中文** · [English](README.en.md)

<br />

面向长程购物 Agent 的可复现后训练与评测项目

<br />

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![LoRA SFT](https://img.shields.io/badge/Post--training-LoRA%20SFT-7B61FF)](docs/sft.md)
[![veRL](https://img.shields.io/badge/veRL-0.8.0-0E8A16)](https://github.com/verl-project/verl)
[![ShopSimulator](https://img.shields.io/badge/Environment-ShopSimulator%20v2.1-4C78A8)](https://arxiv.org/pdf/2601.18225)
[![Benchmark](https://img.shields.io/badge/Benchmark-Final--200--Clean-F59E0B)](docs/evaluation-dataset.md)

<br />

教师轨迹与 LoRA SFT → veRL 在线 GRPO → Final-200 Clean Benchmark 的可审计对比

<br />

**当前成绩：Qwen3.5-2B + LoRA SFT + GRPO = 128 / 200（64.0%）**
　·　相对 2B 未后训练基线 **0.0%** 提升 64 个百分点
　·　相对纯 SFT（63.5%）再 **+1 题**，且**零回退**（做对的题是 SFT 的严格超集）

</div>

![Shopping GRPO project overview](docs/images/project-overview-pipeline.png)

## ShopSimulator 是什么？

[ShopSimulator](https://arxiv.org/pdf/2601.18225) 是一个用于评估长程购物
Agent 的大规模中文购物环境。每个任务会给出一段用户需求，其中可能包含商品类别、
预算、品牌、型号、核心功能以及颜色、尺寸、容量、套餐等具体规格。

Agent 不能只生成一句“推荐购买某商品”，而是必须真正与环境交互：

1. 根据需求搜索商品；
2. 打开并比较候选商品；
3. 查看描述、参数和可选规格；
4. 选择正确的商品变体；
5. 购买满足约束的商品，或者在证据充分时合理终止。

这类任务同时考察指令理解、工具调用、长上下文管理、约束满足和终止决策。项目内嵌
了冻结的 ShopSimulator Environment v2.1 源码和商品数据，位于
[`environments/ShopSimulator/`](environments/ShopSimulator/)，不需要用户再单独
克隆或修改一份环境仓库。

![ShopSimulator 环境概览](docs/images/shopsimulator-overview.png)

## 项目做了什么？

项目按照一条连续的后训练流水线组织：

```mermaid
flowchart LR
    A[教师模型采集轨迹] --> B[Reward v3 回放过滤]
    B --> C[Action-only SFT 数据]
    C --> D[LoRA SFT]
    D --> E[veRL 在线 GRPO]
    F[ShopSimulator v2.1] --> E
    G[Final-200 Clean 测试任务] --> H[统一评估流水线]
    I[Base Model] --> H
    D --> H
    E --> H
```

| 阶段 | 目标 | 入口 | 详细文档 |
|---|---|---|---|
| Baseline | 测量原始 Qwen3.5-2B 的工具使用能力 | `bash scripts/baseline.sh` | [评估](docs/evaluation.md) |
| SFT | 从高质量教师轨迹学习合法、完整的购物行为 | `bash scripts/sft.sh` | [SFT](docs/sft.md) |
| GRPO | 在真实环境 Rollout 中优化 Reward v3 | `bash scripts/grpo.sh` | [GRPO](docs/grpo.md) |
| Evaluation | 使用同一批 Final-200 Clean 留出任务公平比较模型 | `bash scripts/evaluate.sh NAME` | [评估](docs/evaluation.md) |

### SFT 数据是怎么收集的？

当前数据使用 `deepseek-v4-flash` 作为教师模型，在 ShopSimulator
Environment v2.1 中采集：

- 共获得 2,498 条原始任务轨迹；
- 每条轨迹在采集时都真实执行环境动作，再按 Reward v3 终局结果验收；
- 其中 1,026 条通过严格验收，构成 SFT 基础池；
- 本项目在此基础上做了两轮数据重构（见下方「12 个阶段做了什么」），最终池
  **1,210 条**（train 1,089 / val 121），其中 **144 条（11.9%）含 `next_page`
  翻页示范**——这是 SFT 从 60.5% 提到 63.5% 的主要手段；
- 所有池与 GRPO 训练池、Final-200 Clean 评测集保持 task_id 零重叠。

仓库已提供可断点续跑的采集入口：

```bash
python scripts/collect_sft_data.py \
  --tasks data/grpo/train.jsonl \
  --output-dir outputs/sft-collection \
  --target-accepted 1000 \
  --workers 4
```

SFT 只在 Assistant 动作 token 上计算 Loss，用户指令和环境 Observation 会被
Mask。这样模型学习的是可执行的工具策略，而不是背诵环境返回内容。数据哈希、接受率
和采集审计见[数据采集文档](docs/data-collection.md)。

### GRPO 是怎么训练的？

GRPO 从合并后的 SFT 模型开始。veRL 在 ShopSimulator 中为每个 Prompt 在线生成
四条轨迹，环境用确定性的 Reward v3 评估最终购买结果、约束满足程度和终止行为。
训练不使用额外的 LLM-as-a-Judge Reward Model。

本仓库没有复制 veRL 源码，而是固定安装 `verl==0.8.0`，并保留项目自己的
AgentLoop、工具适配层、运行时兼容代码和一个带 SHA-256 校验的小补丁。详细配置见
[GRPO 文档](docs/grpo.md)。

### 评估流水线是怎么设计的？

正式评估由“代码硬检查 + 两个 LLM-as-Judge + 固定分母聚合”组成。两个 Judge
职责不同：

- **DeepSeek V4 Flash 是 Rubric Curator。** 代码先根据每道题的 Query 和私有
  TaskFacts 提取品类、品牌、型号、功能、规格和价格候选；Flash 只能从候选中选择
  用户真正要求的约束、去重并标注 hard/soft，不能创造新的字段或期望值。生成的
  Rubric 冻结一次，由 Baseline、SFT 和 GRPO 共用。
- **DeepSeek V4 Pro 是 Trajectory Judge。** 它读取用户 Query、冻结 Rubric、
  Actor 实际看到的完整轨迹、中性终局状态和白名单代码指标，逐条判断需求是否满足，
  并从搜索策略、候选利用、证据核验、决策质量、终止效率五个维度分别打 0/1/2 分。

这里的 Rubric 是逐任务评分标准，不是向量检索式 RAG。

```mermaid
flowchart TD
    A["Benchmark test_id"] --> B["私有 TaskFacts"]
    B --> C["代码提取 Rubric 候选"]
    C --> D["V4 Flash 整理并冻结 Rubric"]
    A --> E["Actor + ShopSimulator Rollout"]
    E --> F["轨迹规范化 + Action Guard + 确定性硬检查"]
    F -->|基础设施无效| G["not_judged，仍计入 Final-200 Clean 分母"]
    F -->|检查通过| H["移除 Reward、Gold、raw observation"]
    D --> H
    H --> I["V4 Pro 逐需求判断 + 五维评分 + 错误分类"]
    G --> J["四面板结果拼装"]
    I --> J
    J --> K["Reward / Rubric / Trajectory / Deterministic"]
    K --> L["Baseline、SFT、GRPO 按 task_id 配对比较"]
```

以 Final-200 中的 `task_id=8187` 为例，Query 要求“一对卡通-永结同心款的高档
酒红色木梳、礼盒、陪嫁、20 元左右”。代码生成 7 条候选，V4 Flash 冻结为 5 条
Rubric；SFT Actor 用 10 步完成搜索、详情核验、规格选择和购买；V4 Pro 最终给出
`搜索策略 2 / 候选利用 1 / 证据核验 1 / 决策质量 2 / 终止效率 2`，并为每项判断
引用真实的 `event_id`。

Pro 看不到 Reward 分数、Gold 商品私有字段、raw Observation、成功标签或其他模型
结果，因此不能根据答案倒推轨迹质量。最终结果分为四个独立面板：

1. Environment Reward 与终局；
2. Query Rubric 的 hard/soft 满足情况和 Reward disagreement；
3. Pro Judge 五维分布与错误类型；
4. 步数、工具、Guard、重复、上下文和基础设施指标。

四部分不会合成一个总分。缺失、报错和 `not_judged` 任务仍保留在 Final-200 Clean 分母中。
完整数据流、两个模型的完整 Prompt、输入隔离规则、示例 Rubric 和最终统计口径见
[评估流水线文档](docs/evaluation.md)。当前集的筛选依据见
[Final-200 Clean 说明](docs/evaluation-dataset.md)；[Final-200 Dashboard（历史）](docs/evaluation-dashboard.html)
只保留为历史归档。



## 实验结果

所有结果都在 **Final-200 Clean** 上测：200 道留出任务、每题 1 次确定性 rollout
（temperature 0）、严格口径只认 `gold_purchase` 终局。完整协议、失败分布和产物哈希见
[评测更新记录](docs/evaluation-updates.md)。

### 主线成绩

| 模型 | 严格成功率 | 相对上一行 | 产物 |
|---|---:|---:|---|
| Qwen3.5-2B Baseline（未后训练） | 0.0% | — | — |
| + LoRA SFT（初版） | 60.5% | +60.5pp | `experiments/sft/` |
| + LoRA SFT（行为覆盖重构，M10-R v2） | **63.5%（127/200）** | +3.0pp | `sft-m10r/stage-full/merged` |
| **+ GRPO（中等题池，M12，50 步）** | **64.0%（128/200）** | **+0.5pp** | `grpo-m12-merged` |
| `deepseek-v4-flash`（教师，参考上界） | 73.5%（147/200） | — | — |
| Qwen3.8-27B（参考，非本项目产出） | 73.0% | — | — |

### GRPO 的逐题增量（本项目最核心的结论）

| 项 | 题数 |
|---|---:|
| GRPO 做对 | **128** |
| SFT 基线做对 | 127 |
| 两边都对 | 127 |
| **只有 GRPO 对** | **1** |
| **只有 SFT 对** | **0** |

**GRPO 做对的 128 题是 SFT 127 题的严格超集**——它没有牺牲任何一道原来做对的题，
而是净增 1 题。这是本项目 12 个阶段里**第一次出现非零增量**：此前的 3 轮 GRPO
要么打平（M6）、要么崩掉（M6-R）、要么退化 1 题（M11 探针，做对的 126 题是基线的
真子集）。

> **诚实说明**：+1 题落在二项噪声内（σ ≈ 6.9 题）。**总分层面 64.0% 与 63.5% 不可区分**，
> 能站住的表述是「独有正确题从 0 变成 1」，而不是「成功率提升了 0.5 个百分点」。
> 要在统计上区分，需要 ≥134 题（+7）。

### 这 0.5 个百分点是怎么来的

GRPO 前两轮空转的真因在训练日志里：`group/all_equal_ratio` **0.25–0.50**、
`no_purchase_success_ratio` **0.375–0.75** ⇒ 四分之一到一半的组「4 条轨迹全对」
或「4 条全错」，组内优势恒为 0，**这批题只耗算力不产方向**。

M12 的做法是**只换训练数据**：用两轮采样给候选池打难度标签，只保留「4 条轨迹里
对 1–3 条」的**中等题**（223 道），其余超参一个不动。结果：

| 指标 | 换池前 | 换池后 |
|---|---:|---:|
| `no_purchase_success_ratio`（全错组） | 0.375–0.75 | **0.00** |
| `all_equal_ratio`（零梯度组） | 0.25–0.50 | 0.25 |
| Final-200 | 127 | **128** |

**全错组被彻底清零**，这是「换池子」生效的直接证据。残留的 0.25 全部是「全对组」——
中等题在训练中会退化成简单题，这是该路线的天然上限。

### 试过但没走通的路（同样是资产）

| 方向 | 结果 |
|---|---|
| 换 GRPO 配方（lr 5e-6 → 2e-5） | val 低于起点，step 48 熔断 |
| 推理期强制「买前开 k 个候选」 | 同批 9 题 **7/9 → 0/9**（p≈0.016），方向否决 |
| 教模型「翻页」（144 条合成示范） | 63.5%，但该翻的题只翻 7 道；逐回合补理由反而掉到 61.5% |
| 换系统提示词改查询 | 127/200，净 0（5 换 5 双向抵消） |
| 换难题池（201 道 gold 在 20 名外的题） | 63.5%，逐题只改 2 道 ⇒ 空转 |
| Reward v5（`wrong_purchase` 改 band） | Gate 3 梯度份额 0.3% ≪ 门槛 5%，未开训 |

完整的失败归因、判读铁律与硬约束见 [`docs/project-history.md`](docs/project-history.md)
与 [`docs/lessons-learned.md`](docs/lessons-learned.md)。机器可读的训练配置、结果摘要
和限制说明位于 [`experiments/`](experiments/)。

### 12 个阶段做了什么

| 阶段 | 内容 | 结果 |
|---|---|---|
| **M1** 可解性审计 | 三层审计（有解性 / 检索可达性 / 评分可评性），23,421 条语料 | 建立准入契约，200 题冻结 |
| **M2–M4** 数据与奖励 | 补采、Reward v4、数据重建 | SFT 池成形，四池 task_id 零重叠 |
| **M5** SFT 消融 | assistant vs tool-call-only loss | **62.0%**（tool-call-only 胜，选为 GRPO 起点） |
| **M6 / M6-R** GRPO | 按作者配方 / 把 lr 提到 2e-5 | 62.0% 打平 / step 48 熔断 |
| **M7** Reward v5 | `wrong_purchase` 由常数改 band | Gate 3 梯度份额 0.3% ≪ 5%，**未开训** |
| **M8** 多候选核验 | 计划补 100 条多开轨迹 | 靶点被离线解剖否决，**未开训** |
| **M9** 推理期强制探索 | 买前必须打开 k 个候选 | 同批 9 题 **7/9 → 0/9**，方向否决 |
| **M10 / M10-R** SFT 重构 | 补翻页 / 候选筛选 / 超预算示范 | **63.5%（127/200）**，最长一段时间的冠军 |
| **M10-H** GRPO 难题池 | 201 道「gold 在 20 名外」的题并入池，100 步 | 63.5%，逐题只改 2 道 ⇒ 空转 |
| **M11** 查询提示词 A/B | 只改系统提示词第 3 条 | 127/200，净 0（5 换 5 抵消） |
| **M12** 中等题 GRPO | 223 道「4 条里对 1–3 条」的题，50 步 | **64.0%（128/200），首次非零增量** |

## 训练硬件与耗时

### GRPO（veRL 0.8，单卡）

**显存是硬门槛。** M12 实测（50 步）：

| 项 | 实测值 |
|---|---:|
| `actor/perf/max_memory_allocated_gb` | **41.3 GB** |
| `actor/perf/max_memory_reserved_gb` | **57.1 GB** |

⇒ **48 GB 卡装不下**（可用 47.9 GB）。历史上在 48 GB 上冒烟过，峰值 47,641 / 48,507
= 98.2%，崩在 `logprobs_from_logits → F.log_softmax`（词表 248,320 的输出层是元凶）。
**最低 64 GB，推荐 80 GB 以上**，并且必须设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

| 阶段 | 耗时 |
|---|---:|
| vLLM 初始化（此阶段 GPU util=0 属正常） | 20–25 分钟 |
| 单步（`timing_s/step`） | **140–149 秒** |
| ├ 其中 `update_actor` | **103–105 秒** |
| ├ 其中 `gen`（rollout） | 12–19 秒 |
| └ 其中 `save_checkpoint` | 17 秒 |
| **50 步合计（含初始化）** | **2 小时 23 分钟** |

**墙钟瓶颈在 actor 更新，不在 rollout**——缩短 response 或加大 batch 都省不了多少。

### 评测（与训练分开跑）

2B 模型的评测显存需求远低于训练，**48 GB 卡足够**。但有一条硬约束：
**不能用 Blackwell / sm_120 的卡**（如 RTX 6000D）。torch 的 CUDA < 12.9 会让
`vllm serve` 报 `FlashInfer requires sm75 or higher`，评测根本起不来。
本项目因此把训练与评测拆到两台机器上做。

| 环节 | 耗时 |
|---|---:|
| vLLM 加载合并后的权重 | 20–25 分钟 |
| Final-200 评测（8 分片并发） | **约 25 分钟** |
| 合并 FSDP checkpoint（11 GB → 4.2 GB 权重） | 约 15 分钟 |

### SFT（LoRA）

| 项 | 实测 |
|---|---:|
| 单步 | ~74 秒 |
| 总步数 | 133（kept 1,061 条 ÷ batch 8） |
| 完整训练 | ~2.7 小时 |

### 其他环节

| 环节 | 耗时估算 |
|---|---:|
| Teacher 采集（2,498 条原始轨迹） | 取决于接口并发与限流 |
| 给 1,000 道题打难度标签（2 轮，8 并发） | **约 4 小时** |
| 200 任务评测（Base） | ~20 分钟 |
| 200 任务评测（SFT/GRPO） | ~25 分钟 |
| LLM Judge 评分 200 条轨迹 | ~30–60 分钟 |

## 环境要求

- Linux；
- NVIDIA GPU 和兼容的 CUDA Driver；
- [`uv`](https://docs.astral.sh/uv/)；
- 约 **60 GB** 可用磁盘空间（依赖、模型权重、运行产物；**每个 GRPO checkpoint 约 11 GB**）；
- **SFT**：48 GB 显存足够；
- **GRPO**：最低 **64 GB**，推荐 **80 GB 以上**（实测 `allocated 41.3 GB` / `reserved 57.1 GB`）；
- **评测**：48 GB 足够，但**不能用 Blackwell / sm_120 的卡**（torch CUDA < 12.9 会让
  `vllm serve` 报 `FlashInfer requires sm75 or higher`）。

主训练环境使用 Python 3.12，ShopSimulator 使用隔离的 Python 3.10 环境。
`scripts/setup.sh` 会通过 `uv` 创建并安装两套环境。

## 快速开始

以下命令都在仓库根目录执行。

### 1. 安装

```bash
bash scripts/setup.sh
```

该脚本会安装固定版本的 SFT、veRL 和 vLLM 依赖，创建独立的 ShopSimulator
环境，校验并解压商品数据，构建搜索索引，并应用经过版本和哈希检查的 veRL 补丁。

### 2. 启动 ShopSimulator

在第一个终端运行并保持服务：

```bash
bash scripts/start_environment.sh
```

服务默认监听 `http://127.0.0.1:5700`。

### 3. 运行 Baseline

在第二个终端启动基础模型：

```bash
bash scripts/serve_model.sh Qwen/Qwen3.5-2B
```

在第三个终端评估：

```bash
bash scripts/baseline.sh
```

开始训练前请停止模型服务，释放 GPU 显存。

### 4. 训练并评估 SFT

```bash
bash scripts/sft.sh
bash scripts/serve_model.sh outputs/models/sft-merged
bash scripts/evaluate.sh sft
```

完成评估后再次停止模型服务。

### 5. 训练 GRPO

先只解析并打印最终命令，不启动 CUDA 或 Ray：

```bash
bash scripts/grpo.sh --dry-run
```

开始训练：

```bash
bash scripts/grpo.sh
```

复现 M12（当前冠军）的完整配置：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/grpo.sh \
  --model outputs/models/sft-m10r/stage-full/merged \
  --train-data data/grpo/train_medium.parquet \
  --output outputs/models/grpo-m12 \
  -- \
  data.max_response_length=16384 \
  actor_rollout_ref.actor.calculate_entropy=false \
  actor_rollout_ref.rollout.enforce_eager=true \
  data.train_batch_size=4 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.optim.lr=1.0e-5 \
  trainer.total_training_steps=50 \
  trainer.save_freq=25 trainer.test_freq=25
```

几个**必设**项，不是可选项：

| 参数 | 理由 |
|---|---|
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 不设会显著抬高 `reserved`，48–64 GB 卡直接 OOM |
| `actor.calculate_entropy=false` | bypass 模式下算熵会 OOM，且该指标不出现 |
| `rollout.enforce_eager=true` | 规避部分新卡的 FlashInfer 兼容问题 |
| `train_batch_size=4` | 每次更新喂 8 条序列（4×4÷2）；batch 2 效果等价但更慢（227 vs 189 s/更新） |

根据验证集指标选择 Checkpoint，并导出 veRL Actor：

```bash
bash scripts/export_grpo.sh \
  outputs/models/grpo/global_step_50/actor \
  outputs/models/grpo-merged
```

> **续训**：数据不变时加 `--resume`（保留优化器状态与 dataloader 位置）；
> **换训练数据时必须重起**（新输出目录 + 用已合并的权重当 `--model`），
> 因为 checkpoint 里的 `data.pt` 是旧数据集的 sampler 状态。

启动并评估导出的模型：

```bash
bash scripts/serve_model.sh outputs/models/grpo-merged
bash scripts/evaluate.sh grpo
```

每次 `evaluate.sh` 完成后会自动生成 `outputs/evaluation/grpo/report.html`。对已有评测结果补生成报告：

```bash
bash scripts/report.sh grpo
```

批量生成所有单模型报告和综合比较报告：

```bash
bash scripts/report_all.sh
```

报告生成器按评测目录读取 `summary.json` 和 `trajectories.jsonl`，模型名与评测参数会从结果中自动填入，因此换模型或换评测标签不需要修改报告代码。

Checkpoint、Rollout 和日志统一写入 Git 忽略的 `outputs/`。

## Reward v3 简介

Reward v3 是一个确定性的终局 Reward，不依赖另一个大模型进行主观判断：

- 类别和预算是 Hard Gate；
- 品牌、型号、核心功能、关键规格按照 `0.35 / 0.25 / 0.25 / 0.15` 加权；
- 完全满足并命中目标商品得到 `1.0`；
- 完全满足的替代商品得到 `0.55`；
- 部分满足按照连续分数计算，最高 `0.25`；
- 错误购买、过早放弃、重复循环和达到最大步数都会获得不同负奖励；
- 证据不足时标记为 `reward_valid=false`，不会伪装成有效的零分样本。

![Reward V3 decision rules](docs/images/reward-v3-decision-rules.png)

完整公式、终止条件和证据要求见 [Reward v3 设计文档](docs/reward-v3.md)。

## 仓库结构

```text
configs/                         当前 GRPO、AgentLoop 和工具配置
data/
  sft/                           SFT 轨迹（基础池 1,026 条 → 本项目重构为 1,210 条：1,089 / 121）
  grpo/                          JSONL 与 veRL Parquet 数据
  evaluation/                    Final-200 Clean 留出任务
docs/                            数据、SFT、GRPO、评估与 Reward 文档
environments/ShopSimulator/      内嵌环境源码和商品数据
experiments/
  baseline/                      Baseline 配置与结果
  sft/                           SFT 配置与结果
  grpo/                          GRPO 配置与结果
scripts/                         面向用户的薄入口脚本
src/shopping_grpo/
  collection/                    Teacher 轨迹验收与 SFT 数据构造
  environment/                   环境客户端、动作、工具和 Observation
  training/sft/                  SFT 数据渲染与 Mask
  training/grpo/                 veRL AgentLoop、适配和动态采样
  evaluation/                    硬检查、Rubric、轨迹 Judge 和指标汇总
tests/                           核心单元、入口和 Wheel 安装检查
```

## 常用配置

| 环境变量 | 默认值 |
|---|---|
| `BASE_MODEL` | `Qwen/Qwen3.5-2B` |
| `SHOPSIM_BASE_URL` | `http://127.0.0.1:5700` |
| `LLM_BASE_URL` | `http://127.0.0.1:8000/v1` |
| `SERVED_MODEL_NAME` | `shopping-agent` |
| `SFT_ADAPTER_DIR` | `outputs/models/sft-lora` |
| `SFT_MERGED_DIR` | `outputs/models/sft-merged` |

GRPO 的高级 Hydra 参数可以追加在 `--` 后：

```bash
bash scripts/grpo.sh -- \
  trainer.total_training_steps=20 \
  trainer.save_freq=10
```

SwanLab 默认关闭，需要时显式启用：

```bash
export SWANLAB_API_KEY=...
bash scripts/grpo.sh --logger swanlab
```

## 文档导航

**先读这两份**：

- [项目历程与问题台账](docs/project-history.md) — 时间线、12 个阶段的完整结果、负结果归因、判据体系
- [经验台账](docs/lessons-learned.md) — 已关闭的路、判读铁律、硬约束、工程坑

分阶段文档：

- [数据采集与数据来源](docs/data-collection.md)
- [LoRA SFT](docs/sft.md)
- [使用 veRL 进行 GRPO](docs/grpo.md)
- [留出集评估](docs/evaluation.md)
- [Final-200 Clean 测试集说明](docs/evaluation-dataset.md)
- [Final-200 Benchmark Dashboard（历史）](docs/evaluation-dashboard.html)
- [Reward v3 设计](docs/reward-v3.md)
- [可审计实验结果](experiments/comparison.md)

## 后续改进计划

### 1. 先做归因，不要先加训练量（零显卡成本）

教师 `deepseek-v4-flash` 是 **147/200**，我们 128/200，**两边并集 165/200 = 82.5%**
⇒ 这不是全面落后，而是互补盲区。真正的增量空间在 **38 道「我们错、老师对」的题**上：

| 失败模式 | 题数 |
|---|---:|
| A 正确商品没进检索结果 | 13 |
| B 商品在结果里但没打开 | 15 |
| **C 打开过还买错** | **0** |
| D 买对商品但选错规格 | 9 |
| E 没买 | 1 |

**C = 0 说明「打开哪一个」是唯一决策点**——一旦打开对了，就不会买错。
下一步应先离线归因这 38 题（只读已有轨迹，不需要显卡），再决定改数据还是改奖励。

### 2. 扩池要按真实预算做

中等题（4 条轨迹里对 1–3 条）只占候选池的 **14%–17%**，而打标 1,000 道题 2 轮需要
**约 4 小时**。⇒ **要 200 道新的中等题，得打标 1,200 道**（4 小时）+ 训练 4.7 小时。
任何「扩池」计划都应该按这个预算规划，而不是「抽 200 道就行」。

### 3. 评测集的信噪比才是天花板

200 题的二项噪声 **σ ≈ 6.9 题**，意味着**任何小于 7 题的改进都无法在统计上被证实**。
本项目最后两轮的真实增益是 +1 题和 +3 题，都淹没在噪声里。
**扩充评测集（或对同一模型多次采样取均值）比继续微调更能提高结论的可信度。**

### 4. 其他方向

- **扩大模型规模**：2B 在需要商品知识的题上（例如判断该补哪个规格词）有结构性劣势，
  参数量可能是真正的杠杆点。
- **更细粒度的信用分配**：终局奖励稀疏，关键中间步骤（「打开第几个候选」）拿不到信号。
- **混合 Teacher**：单一教师的行为覆盖有限，多教师混合可以扩大行为分布。

## Star History

<a href="https://www.star-history.com/?repos=YYHDBL%2Fshopping-grpo-longhorizon&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=YYHDBL/shopping-grpo-longhorizon&type=date&theme=dark&legend=top-left&sealed_token=wgQ1K2TiIB2luvZFJ54oMEhME-cxYmFv_wNoNXnT7lMZHsuQUy7NThQAG2VwpEeiUBoRxd09ASiB60cvvBaEvqVqyv49wYKZSF2H_Jft3Iq1ZZ0c5Sk2SQQejxHxMQwayMTRroOh5JhcWgXk6w8HHwjP6UgTquINRr40c7XysMi_j2BksVwqOWSIz8Ny" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=YYHDBL/shopping-grpo-longhorizon&type=date&legend=top-left&sealed_token=wgQ1K2TiIB2luvZFJ54oMEhME-cxYmFv_wNoNXnT7lMZHsuQUy7NThQAG2VwpEeiUBoRxd09ASiB60cvvBaEvqVqyv49wYKZSF2H_Jft3Iq1ZZ0c5Sk2SQQejxHxMQwayMTRroOh5JhcWgXk6w8HHwjP6UgTquINRr40c7XysMi_j2BksVwqOWSIz8Ny" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=YYHDBL/shopping-grpo-longhorizon&type=date&legend=top-left&sealed_token=wgQ1K2TiIB2luvZFJ54oMEhME-cxYmFv_wNoNXnT7lMZHsuQUy7NThQAG2VwpEeiUBoRxd09ASiB60cvvBaEvqVqyv49wYKZSF2H_Jft3Iq1ZZ0c5Sk2SQQejxHxMQwayMTRroOh5JhcWgXk6w8HHwjP6UgTquINRr40c7XysMi_j2BksVwqOWSIz8Ny" />
 </picture>
</a>

## 引用与致谢

本项目建立在
[ShopSimulator 论文](https://arxiv.org/pdf/2601.18225)及其开源环境、
[veRL](https://github.com/verl-project/verl) 和
[Qwen](https://github.com/QwenLM/Qwen3) 之上。

评测协议和 Benchmark 构建还参考了
[VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-world Applications](https://arxiv.org/pdf/2509.26490)
以及
[EComAgentBench: Benchmarking Shopping Agents on Long-Horizon Tasks with Distributed Hidden Intent](https://arxiv.org/pdf/2606.17698)。

仓库结构和教程呈现参考了
[qiqihezh/agentic-grpo-longhorizon](https://github.com/qiqihezh/agentic-grpo-longhorizon)。
感谢 [OpenCode Go 套餐](https://dev.opencode.ai/go) 对开发工作的支持。

### Contributors

<a href="https://github.com/Guochangwei917">
  <img src="https://github.com/Guochangwei917.png?size=80" width="80px;" alt="Guochangwei917"/>
</a>
