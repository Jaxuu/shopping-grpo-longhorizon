# 项目历程与问题台账

> 用途：这是**可入库、可长期保存**的项目主记录，供写项目介绍、复盘与归档引用。
> 它只收录结论与证据指针，细节留在各阶段的专门文档里（见 §1 记录地图）。
> 最后更新：2026-10-18（M7 / Reward v5 落地后）。

---

## 0. 一句话历程

**Baseline 0.0% → SFT 62.0%（+62pp）→ GRPO 三次尝试均无显著增益（作者 +1.5pp，p≈0.664；
我们两轮 0 / 负），瓶颈最终定性为「奖励信号吞吐」而非超参，处置落在 Reward v5 的
`wrong_purchase` 档位 shaping 上。**

---

## 1. 记录地图（哪个文件记录什么）

| 层 | 位置 | 内容 | 是否入库 |
|---|---|---|---|
| **主记录** | `docs/project-history.md`（本文件） | 时间线、问题台账、负结果、判据体系 | ✅ |
| 阶段文档 | `docs/reward-v3.md` / `reward-v4.md` / `reward-v5.md` | 奖励口径的逐版规格与变更理由 | ✅ |
| 阶段文档 | `docs/data-collection.md` / `sft.md` / `grpo.md` / `evaluation*.md` | 各阶段的流程与协议 | ✅ |
| 实验产物 | `experiments/{baseline,sft,sft-m5,grpo}/`、`comparison.md` | 配置 + 摘要 + 对照表（小体积） | ✅ |
| 审计产物 | `audit/*.jsonl|json`、`audit/reward-v4-hashes.txt`、`reward-v5-hashes.txt` | 准入判定、冻结哈希（机器可读） | ✅ |
| **工作记录** | `plans/`（19 份） | 逐阶段 runbook、证据计划、决策、交接 | ❌ gitignored |
| 会话记忆 | `.workbuddy-ai/memory/` | 每日工作日志 + 长期约定 | ❌ gitignored |

⚠️ **归档风险（已识别）**：`plans/` 与 `.workbuddy-ai/` 不进 git，只存在于本地/服务器
工作目录。因此本文件刻意把 `plans/` 里**有长期价值的结论**（尤其 M6/M6-R/M7 的
失败归因与判据）摘录成表，确保即使工作目录丢失，结论仍在。

---

## 2. 时间线与结果

| 阶段 | 做了什么 | 结果 | 落点 |
|---|---|---|---|
| **M1** 可解性审计 | 三层审计（有解性 / 检索可达性 / 评分可评性），23,421 条商品语料、2,442 条任务 | 建立准入契约 `shopping-task-admission-v1`；评测集 200 题冻结 | `plans/audit-report.md`、`audit/` |
| **M2** 补采 | 高歧义题补采 100 条 SFT | 并入 SFT 池 | `plans/m2-collection-runbook.md` |
| **M3** Reward v4 | 预算解析覆盖扩展（parser v4）+ 两个纯信息层（soft 价格、过程证据）+ 版本同步 | 评测集预算解析率 93.5%；hard gate 语义逐字节不变 | `docs/reward-v4.md` |
| **M4** 数据重建 | 并入 M2 增量、生成 D_full / D_toolonly、刷新 metadata | SFT 1,252 行；GRPO train 906 / val 49；评测 200 冻结 | `plans/m4-rebuild-report.md` |
| **M5** SFT 消融 | 两种 loss scope 重训（A=assistant / B=tool-call-only） | **M5-B 62.0%** 全指标最优 ⇒ 选为 GRPO 起点 | `experiments/sft-m5/` |
| **M6** GRPO 第一轮 | 按作者配方（lr 5e-6、batch 4、n 4、clip .28、50 步） | **62.0%**，与 M5-B 打平（123/124 题相同）⇒ 零结果 | `plans/m6-grpo-runbook.md` |
| **M6-R** GRPO 第二轮 | 唯一变量 lr 5e-6→2e-5 | val@25 **0.3425** < 起点 0.3599；step 48 熔断 ⇒ 负结果，不采纳 | `outputs/archive/m6r-crash48/` |
| **M7** Reward v5 | `wrong_purchase` 由常数 −0.85 改为 band `[−0.85, −0.66]` | 已实现、待过三个 Gate 后开跑 | `docs/reward-v5.md` |

### 2.1 各阶段评测成绩（Final-200 Clean，每题 1 次确定性 rollout）

| 模型 | Done | Strict | Purchase | Mean reward | Mean steps | Guard 拒绝 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B baseline | 18.0% | 0.0% | 0.0% | −0.1105 | 5.88 | 752 |
| LoRA SFT（M3 期） | 96.5% | 60.5% | 60.5% | 0.4729 | 12.34 | 52 |
| GRPO step 100（作者配方） | 96.5% | 62.0% | 62.5% | 0.5158 | 11.85 | 38 |
| M5-A（assistant loss） | 98.5% | 59.0% | 59.0% | 0.4359 | 8.07 | 17 |
| **M5-B（tool-call-only loss）** | **100.0%** | **62.0%** | **62.0%** | **0.4943** | 8.41 | **3** |
| M6（我们的 50 步） | 96.5% | 62.0% | 62.0% | — | — | — |
| Qwen3.8-27B（参考，非本项目产出） | 99.5% | 73.0% | 73.0% | 0.6354 | 6.15 | 28 |

**增益归因**：SFT +60.5pp（学会走流程，Done 18%→96.5%）；GRPO 相对 SFT **+1.5pp**
（12 改善 / 9 回退，净增 3 题，exact McNemar **p≈0.664，不显著**）；模型规模 +11pp（参考）。
⇒ **在 2B 约束下 RL 不是杠杆点**，优化重心应在数据质量与评测信噪比。

---

## 3. 问题台账

四类。每条给「现象 → 证据 → 处置 → 落点」。

### A. 数据与契约

| # | 现象 | 证据 | 处置 | 落点 |
|---|---|---|---|---|
| A1 | 预算解析覆盖不足（中文数字、口语单位、无前缀区间、`\xa0`） | M1 审计：`grpo_validation` 解析率仅 48% | parser v4：补 6 类句式 + 价格语境门控；评测集解析率 93.5% | `docs/reward-v4.md §1` |
| A2 | 规格匹配被非 ASCII 空白破坏 | 评测集 `task_id=3647` 复现 | NFKC 归一化，用真实语料固化为回归 | `tests/test_reward_v4.py` |
| A3 | Reward 行为已变但版本标识未同步（R11） | v3 代码 + v4 行为并存 | 版本化 + 双层口径（消费端 7 处） | `docs/reward-v4.md §3` |
| A4 | gold 自洽失败 / 死题污染训练池 | M1 审计：`grpo_train` gold 自洽失败 90、死题 97 | 准入契约过滤；`grpo_train` 1000→906 | `audit/admission_verdicts.jsonl` |
| A5 | v4 解析器仍有 4 类已知缺陷 | 全语料重放（23,421 goals） | **不修**：对四个数据集影响为 0，按 P8 不在中途改冻结解析器 | `docs/reward-v4.md §5` |

### B. 训练与算法

| # | 现象 | 证据 | 处置 | 落点 |
|---|---|---|---|---|
| B1 | GRPO 零增益（M6） | 62.0%，与 M5-B 的 123/124 题相同 | 定性为「信号吞吐」问题，不动超参 | `plans/handoff-2026-10-18.md` |
| B2 | 提 lr 无效且自毁（M6-R） | val@25 0.3425 < 起点 0.3599；step 48 `max_consecutive_skipped_updates=10` 熔断 | **lr 杠杆三点封闭**（1e-6 +1.5pt / 5e-6 ±0 / 2e-5 负）；不做 resume | `outputs/archive/m6r-crash48/` |
| B3 | 全池 57.2% 的组零方差 | 全成功 44.4% + 全失败 28.7% | `dynamic_sampling` 本身就是过滤器；静态过滤训练集只省 rollout，不改梯度 | `plans/m6-grpo-runbook.md` |
| B4 | 加 `rollout.n` 看似能扩可用池 | n=8 可用组 45.4%（vs 40.4%），但吞吐 **51.5 vs 97.8** 组/1000 rollout | **不加 n**（亏 1.9×）；留给 M8 的「n=8 + batch=8」组合臂 | `scripts/analyze_group_size_sweep.py` |
| B5 | turn-level 过程奖励看似是缺口 | 步数中位 **8 vs 8**、长尾 0.0% vs 3.9%、三缺陷提升比≈1 | **不做**：可及空间≈0 | `scripts/analyze_eval_failures.py` |
| B6 | `wrong_purchase` 23 题（11.5%）零梯度 | 奖励恒 −0.85，组内 4 条全等 ⇒ `drop_reason="constant_reward"` | M7 靶点：档位内 soft 排序（v5） | `docs/reward-v5.md` |
| B7 | v5 的 cap 取 −0.45 会破坏档位序 | 它高于 `max_steps(−0.50)`、`repeat_loop(−0.65)` | cap 收到 **−0.66**，并写成断言 | `docs/reward-v5.md §1` |
| B8 | v5 的梯度份额可能过小 | `norm_adv_by_std_in_grpo: false` ⇒ `A = r − mean`，带宽即梯度量级 | 新增 **Gate 3**（梯度份额），预登记阈值 10%/5% | `scripts/analyze_v5_gradient_share.py` |

### C. 观测与判读（最容易误判的一类）

| # | 现象 | 证据 | 处置 |
|---|---|---|---|
| C1 | `ppo_kl` 在 `bypass_mode=true` 下是跨引擎噪声地板 | 两轮几乎不动（0.0033→0.0036）而 `clipfrac` +84% | 弃用；改用 `pg_clipfrac`（≥1.1e-3 才算动了） |
| C2 | `clip_ratio=0.28` 等于没有安全带 | clipfrac 只有 1e-3 量级 | 每步仅 56(m6)/108(m6r) 个 token 被裁剪（占 ~100k 的 0.05–0.10%） |
| C3 | 「打开 <2 个商品」被当成缺陷 | 覆盖 **99%** 的题，连 gold 成功题也一样 | 不是缺陷，是行为常态；缺陷必须有区分度 |
| C4 | 用未激活维度算失分 | 未激活维度恒为 0 | 只能用 `evidence.preference_scoring.dimensions` 里 `active=True` 的 |
| C5 | 过程形状用均值判读 | 3 条离群轨迹（15/20/35 步）把均值拉偏 | 改用中位数 + 长尾占比 |
| C6 | 「题间有 14 个取值」被当成 v5 的收益 | GRPO 的 advantage 是**组内**算的 | 题间分辨率不产生梯度；只有 Gate 2（组内方差）算 |
| C7 | 改奖励后 `reward/mean@1` 会机械上移 | 同一策略、同一 checkpoint，失败样本从 −0.85 变 −0.66..−0.85 | 主指标改用 `purchase_success`；用它选点还有偏置（好策略加成更小） |
| C8 | val 指标键前缀是 `val-core/` / `val-aux/`，不是 `val/` | 用 `grep 'val/'` 查不到会误判「验证没跑」 | 记住前缀；`val_before_train` 的初始验证不进 diagnostics |
| C9 | 数步数用 `grep -c '^step:'` 会错 | 跳过行也打 `step:`，且 tqdm 的 `\r` 让 `^` 漏匹配 | 权威口径 `grep -ac optimizer_step <diagnostics.jsonl>` |
| C10 | `timing_s/` 混有 token 长度键 | 未排除时出现 4729% 的荒谬占比 | 用 `NON_DURATION_SUFFIXES` 排除 |
| C11 | 「越训越慢」直接归因超参 | 实测单步 ≈ `90 s + 0.012 s/token` | 先分解：长度上行 vs 重试暴涨（看 `skip_total`） |

### D. 工程与环境

| # | 现象 | 证据 | 处置 |
|---|---|---|---|
| D1 | `calculate_entropy=true` 既无效又必然 OOM | `bypass_mode` 跳过 `_compute_old_log_prob`（熵的唯一来源）；backward 多留 `24576×248320×2 B = 12.47 GiB` | preflight 直接拒绝；启动必须带 `calculate_entropy=false`；chunking 救不了（只切前向） |
| D2 | `data/environment.json` 的 `runtime_files_sha256` 过期 | preflight `SystemExit` | 改 `reward.py` 后**最后一步**重算哈希，且按 LF 归一化算（Windows CRLF 会算错） |
| D3 | `audit/reward-v4-hashes.txt` 是 CRLF 原始字节口径 | 未改动文件也对不上 | v5 那份改用 LF 并在文件头写明差异 |
| D4 | 版本门槛分散在 10 处，漏改静默失败 | 漏 `evaluation/metrics.py` ⇒ `_strict_success` 恒 False | 逐处清单 + `adapter/tools.py` 复用常量消除双份字面量 |
| D5 | 欠费/断连后误以为进程被杀 | `tmux` 会话存活、诊断文件仍在追加 | 先查 `tmux ls` + 文件 mtime + `nvidia-smi`，再决定是否重启 |
| D6 | 训练中途重启 env 服务会打断在飞轨迹 | — | env 健康 ≠ 训练健康，先看 timing 分解 |

---

## 4. 奖励版本演进

| 版本 | 核心变更 | 判定语义 | 文档 |
|---|---|---|---|
| v3 | 基线：hard gate（category + budget，严格 `actual <= upper`）+ 维度加权 | — | `docs/reward-v3.md` |
| v4 | 解析覆盖扩展（parser v4）+ soft 价格接近度 + 过程证据 + 版本同步 | **逐字节不变** | `docs/reward-v4.md` |
| v5 | `wrong_purchase` 由常数 −0.85 → `min(−0.66, −0.85 + 0.20 × ms × prox)` | **逐字节不变**（`reward_type`/hard gate/`purchase_success`/`reward_valid`） | `docs/reward-v5.md` |

**v5 的设计约束**（已写成断言，见 `tests/test_reward_v5.py`）：

- `driver = 0` 时与 v4 完全一致（最坏情形不变）；
- `cap(−0.66) < repeat_loop(−0.65) < max_steps(−0.50) < early_abstain(−0.35)`：档位序保持，
  `wrong_purchase` 仍是最差档（否则等于恢复恐慌性购买）；
- 轴用 `ms × prox` 而非纯 `ms`：23 题里 20 题是超预算失败，且 `ms` 与超支幅度
  Spearman **+0.42**（越像正确答案的商品越贵）。

---

## 5. 负结果清单（同样是资产）

| 尝试 | 预期 | 实测 | 结论 | 为什么值得记 |
|---|---|---|---|---|
| GRPO（作者配方，100 步） | — | +1.5pp，p≈0.664 | 不显著 | 作者自己也几乎没涨，说明 2B + 当前 SFT 接近天花板 |
| M6（我们，50 步，lr 5e-6） | >62% | 62.0%，123/124 题与 M5-B 相同 | 零结果 | 证明「换配方」不是杠杆 |
| M6-R（lr×4） | 更大步长 | val 低于起点 + 熔断 | 负结果 | lr 杠杆三点封闭；熔断是 feature 不是 bug |
| 加 `rollout.n`（4→8） | 更多可用组 | 吞吐 −47% | 否决 | 每步固定只训 `train_batch_size` 组，加 n 不增加独立组 |
| turn-level 过程奖励 | 补过程信号 | 过程形状无区分度 | 否决 | 可及空间≈0；避免下一轮重复探索 |
| 关 `bypass_mode` | 修好 ratio | 定量：梯度方差只膨胀 0.66% | 降级为条件性动作 | 修仪器不修优化器；触发条件：clipfrac ≥1e-2 或长度 >6000 或 skips ≥3 |

**一条容易被忽略的差异**：作者那轮 GRPO 的失败构成是 `repeat_loop 25`（12.5%）、
`wrong_purchase 5`；我们 M6 那轮是 `repeat_loop 1`、`wrong_purchase 23`。
**同一评测集、不同训练配方，失败结构完全不同**——这曾让 M7 靶点一度定在
`repeat_loop` 上，后来被 `m6-step50` 的实测数据纠正。教训：靶点必须用**当前策略**的
失败分布来定，不能用历史轮次的。

---

## 6. 判据体系（下一轮沿用）

| 用途 | 用什么 | 不要用什么 |
|---|---|---|
| 收益判定 | val 曲线（49 题，±1 题 = ±0.0204）+ 最终 Final-200 | 训练侧 score（batch 噪声 ±0.5） |
| 主指标（v5 之后） | `purchase_success` 率（分类量，与奖励数值无关） | `reward/mean@1`（改奖励会机械上移） |
| 步长 | `pg_clipfrac`（≥1.1e-3 才算动了） | `ppo_kl`（噪声地板） |
| 信号吞吐 | 每步 skip 次数、`effective_ratio`、`kept_group_count` | — |
| 发散/熔断 | `consecutive_skips`、`all_equal_ratio` | — |
| 选点 | `scripts/select_grpo_checkpoint.py`（只用 val） | Final-200（那是考卷） |
| 改动安全性 | 数据重放（`scripts/verify_reward_v5.py`）+ preflight | 只靠单测 |
| 值不值得开跑 | Gate 1 指标键 / Gate 2 组内方差 / Gate 3 梯度份额 | 直觉或预期收益 |

---

## 7. 可直接引用的数字（写项目介绍用）

- 商品语料 **23,421** 条；四池任务 **2,442** 条；评测集 **200** 题冻结。
- SFT 数据 **1,252** 行（仓库 1,000 + 外部去重 450 中的可用部分 + M2 补采 100）。
- GRPO 池：train **906** / val **49**（从 1,000 / 50 经准入过滤）。
- **Baseline 0.0% → SFT 62.0%（+62pp）→ GRPO 62.0%（±0）**。
- GRPO 相对 SFT 的历史增益：**+1.5pp，McNemar p≈0.664（不显著）**。
- 失败构成（M6，Final-200）：gold 124 / partial 50 / wrong 23 / repeat_loop 1 / unverifiable 2。
- **零梯度失败占全部失败的 34.2%**（26/76）。
- `wrong_purchase` 内部 `match_score`：0 → 0.812，**14 个取值**，均值 **0.415**；
  其中 20/23 是超预算失败，超支幅度 1.02×–3.18×，13/20 ≤1.35×（近失簇）。
- v5 之后这 23 题从 **1 个取值**变为 **12 个取值**，带内跨度 0.19。

---

## 8. 编号与命名说明

- **M1–M6**：数据审计 / 补采 / Reward v4 / 数据重建 / SFT / GRPO。
- **M6-R**：M6 的 lr 复跑臂（负结果）。
- **M7**：⚠️ 在早期 `plans/task.md` 里指「三模型统一评测 + 配对比较」；自 2026-10-18
  起被复用为「**Reward v5 的 GRPO 臂**」的代号。引用时请注明是哪个含义。
- **M5-A / M5-B**：两种 SFT loss scope（assistant / tool-call-only），B 为 GRPO 起点。
- **Gate 1/2/3**：开跑前的三道准入检查（指标键 / 组内方差 / 梯度份额）。

---

## 9. 维护约定

1. 每完成一个阶段，在 §2 时间线加一行、在 §3 台账补条目；**结论进本文件，细节留在专门文档**。
2. 负结果必须进 §5——它是防止下一轮重复探索的唯一凭据。
3. 改奖励口径必须同时更新 §4 与 `docs/reward-v*.md`，并重算 `audit/reward-v*-hashes.txt`。
4. 本文件入库；`plans/` 与 `.workbuddy-ai/` 不入库，因此**任何有长期价值的结论都要回流到这里**。
