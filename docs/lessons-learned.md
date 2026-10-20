# 经验台账：已关闭的路、判读铁律、硬约束

> **这份文档的用法（强制）**：
> - **动笔改计划前**读 §1（已关闭的路）和 §2（判读铁律）。
> - **定靶点前**读 §2.5（靶点必须用当前策略的失败分布定）。
> - **宣布结果前**读 §4（判据）。
> - **开跑前**读 §3（硬约束）和 §5（工程坑）。
>
> 事实性数字的完整出处见 `docs/project-history.md`；本文只保留**决策所需**的部分。
> 本文的每一条都对应一次真实的返工，不是提醒，是**已经付过学费的结论**。

---

## 0. 一句话

**本项目最大的浪费不是训练失败，是误判。** 训练只烧了约 8 小时显卡；
而「把没教过的算成学不会」这一个错误，让方向在 M7→M8→M9 上连错了三轮，
每一轮都基于一个**看起来有数字支撑、实则口径错**的结论。

到 2026-10-20 为止，同类误判已发生 **6 次**。§2 的六条铁律就是从这 6 次里提炼的。

---

## 1. 已关闭的路（不要再试，除非满足"重开条件"）

| 路 | 判定 | 关键证据 | 重开的唯一条件 |
|---|---|---|---|
| **GRPO 调 lr** | **关闭（三点封闭）** | 1e-6/100 步 → +1.5pp（p≈0.66，不显著）；5e-6/50 步 → **±0**（123/124 题与 SFT-B 完全相同）；2e-5/50 步 → **step 48 熔断崩溃** | 无。三个点已覆盖一个数量级 |
| **GRPO 加倍步数**（5e-6 × 100 步 / 跑满 epoch） | **挂起，不是关闭** | 我们只跑了 906 题池子的 **22%**（200 组），作者跑 100 步也只 +1.5pp | SFT 数据重构（M10）完成后，若新基线仍有 GRPO 空间再考虑。**当前主线不是它** |
| 加 `rollout.n`（4→8） | 关闭 | 吞吐 **−47%**；每步固定只训 `train_batch_size` 组，加 n 不增加独立组 | 无 |
| turn-level 过程奖励 | 关闭 | 过程形状无区分度，可及空间 ≈0 | 无 |
| **Reward v5 救"全错组"** | 关闭 | 4 条采样全错的组只有 **28 个（3.1%）**；v5 三个候选的梯度份额 **0.3% / 0.1% / 0.1%**，门槛 5% ⇒ **差 50 倍** | 无。**注意**：曾误读为"v5 救不了 61%"，那个 61% 指的是「组内至少 2 条 wrong」，不是「4 条全错」，且这些组本来就有梯度 |
| **补「多开商品」示范** | 关闭 | 池里已有 112 条多开轨迹（8.9%），模型执行率仅 **0.5%**；M9 强制 k=3 反而 **7/9 → 0/9** | 无。评测**不考「看了几个」**，只考最终买了什么；且环境只允许买当前打开的那个 ⇒ 多看 = 买最后一个 |
| **改查询捞回"搜不到"的商品** | 关闭（靶点已转移） | A 桶 20 题里 **18 题的 gold 就在 BM25 150 召回内**（中位 42 名、最小 23、最大 114），观察只渲染前 20 名 ⇒ 是**不翻页**，不是搜不到 | 无。正确的靶点是**翻页**（见 §2.4） |
| 改 `SYSTEM_PROMPT` | **禁止** | SFT 轨迹是带着当前 prompt 采的，改了之后 Final-200 分数**不可比**，62.0% 这个基线作废 | 无。且第 3 条早已写明"缩短查询、更换关键词**或翻页**" ⇒ 规则齐备、缺的是示范 |
| 调 `max_steps` | **禁止** | `configs/environment.json` 的 `termination.max_steps` **硬冻结 35**；`SHOP_MAX_STEPS` 与之不同会抛 `ValueError` | 无。`evaluate.sh` 里的 `MAX_STEPS` 是死旋钮，已删 |
| 换更大底座 | 挂起 | — | 数据路线（M10）走完仍不够时 |

---

## 2. 判读铁律（六条，每条对应一次返工）

### 2.1 先问「有没有机会」，再问「会不会」

**规则**：任何形如「模型没做 X」的统计，**必须先证明它当时有机会做 X**。

- **犯过的错**：曾统计出「417 条从没选规格」并定为唯一失分维度。实际上那 417 条里
  模型买的是**另一个商品**，金标的规格轴根本不在页面上 ⇒ 必然"没选"。指标是循环的。
- **另一例**：A 桶 20 题被判定为"搜不到"，实际 18 题的 gold 就在召回内，只是排在第 2–6 页。
- **强制动作**：统计前先写一句「模型在这个状态下能不能做这个动作」，不能就换口径。

### 2.2 「池子里有 N 条同类」只能证明**类型存在**，不能证明**数量够**

- **犯过的错**：用「543/1,252 = 43.4% 的行含规格近似变体」推出"示范量够，补数据无用"。
  43.4% 只能排除「池子里完全没有这种模式」，**完全排除不了「例子太少、2B 学不牢」**。
- **本项目从未测过「数据量 → 分数」曲线**。唯一一次 SFT 对照（59.0% vs 62.0%）
  差的是 **loss 口径**，不是数据量。
- **强制动作**：说"够"之前，必须能回答「加一倍的同类数据，我预期涨几个点，为什么」。

### 2.3 推理期外挂规则的结果，不能外推到「训练也学不会」

- **犯过的错**：M9 在推理期加一条守卫「买前必须开过 3 个候选」，同批 9 题 **7/9 → 0/9**，
  据此宣布"多候选比较这条路不通"。但这测的是**推理期强加的规则**，
  模型没有"回到旧候选再下单"的能力（池里仅 4.6% 有），所以规则把它推向了"买最后一个"。
- **强制动作**：区分三种干预——**改环境规则 / 改 prompt / 改训练数据**。
  前两种的失败不能证明第三种无效。

### 2.4 每个口径都要过「反事实检验」

**规则**：问一句「如果模型当时做了相反的选择，这个指标会变吗？」不会变 ⇒ 口径无效。

- **犯过的错**：用「gold 标题的 2-gram 覆盖率」（A 桶 0.09 vs 成功组 0.26）判定"词表鸿沟，
  换查询也捞不到"。这个指标测的是**措辞相似度**，与**可检索性**无关。直接查索引得到相反结论。
- **强制动作**：把关键分类做成**强制三分类**（如「值在页面上：逐字可见 / 归一化后可见 / 缺失」），
  不允许"其他"这个兜底类别（工具 `scripts/analyze_option_selection_failures.py` 是范例）。

### 2.5 靶点必须用**当前策略**的失败分布定，不能用历史轮次

- **犯过的错**：作者那轮 GRPO 的失败构成是 `repeat_loop 25`（12.5%），我们 M6 那轮是
  `repeat_loop 1`、`wrong_purchase 23`。同一评测集、不同配方，失败结构完全不同。
  M7 一度把靶点定在 `repeat_loop` 上，后来被实测纠正。
- **强制动作**：定靶点前重跑一遍 `scripts/analyze_eval_failures.py`，用当前检查点的轨迹。

### 2.6 指标名对不上，就是没读到

- **犯过的错**：在训练日志里 grep `val/` 找不到验证键，误判"验证没跑"。
  实际前缀是 **`val-core/`** 和 **`val-aux/`**。
- **强制动作**：读日志前先 `head -1` 看一眼真实的 key 名，别按记忆 grep。

---

## 3. 硬约束（改了就等于换题 / 换目标函数）

| 约束 | 位置 | 为什么不能动 |
|---|---|---|
| `max_steps = 35` | `configs/environment.json` → `termination.max_steps` | 硬冻结，改则抛 `ValueError`。模型对每个候选走 8 步仪式 ⇒ k 个候选约需 `1+8k+1` 步，k=3 已 26 步 |
| `SYSTEM_PROMPT` | `evaluation/rollout.py` | SFT 轨迹带着它采的，改了 Final-200 不可比 |
| 工具集 | `environment/tools.py` | `next_page` / `back_to_search` / `finish_without_purchase` 全都在，不缺工具，缺示范 |
| 观测只渲染前 20 名 | `--observation-search-top-k` 默认 20 | **不要调大**——那等于替模型翻页，会污染评测口径 |
| 终局奖励离散查表 | `shopsimulator-reward-v4` | gold 1.0 / valid_alt 0.55 / partial ≤0.25 / repeat_loop −0.65 / wrong −0.85…−0.66。**改表 = 目标函数变 = 历史全部作废** |
| 评测集 200 题 | `data/evaluation/tasks.jsonl` | 冻结。它是考卷，不是训练素材 |
| 四池两两零重叠 | Eval / SFT / GRPO train / val | 破了就无法归因 |
| SFT 行 schema 恰 4 键 | `{trajectory_id, task_id, messages, tools}` | 多一个键就会在训练侧静默出错 |
| `calculate_entropy = true` | 训练启动参数 | **禁止**。`bypass_mode=true` 下跳过 `_compute_old_log_prob` ⇒ `actor/entropy` 永不出现；且强制 `inplace_backward=False` ⇒ backward 多留 12.47 GiB，**必然 OOM**（M6-R 实测）。preflight 会拒绝 |

---

## 4. 判据（宣布"涨了"之前必须过这一关）

| 用途 | 用什么 | 不要用什么 |
|---|---|---|
| **收益判定** | Final-200 严格成功率（`purchase_success`，分类量） | `reward/mean@1`（改奖励会机械上移） |
| 显著门槛 | **≥69%（138/200）**，McNemar p<0.05 | 差 3–5 题就宣布"方向对了" |
| 中间态解读 | 66–69% = 方向对但量不够；≤62% 才谈"能力上限" | 用 val（49 题，±1 题 = ±0.0204）当最终结论 |
| 归因 | 必须重跑分桶：A（没进结果页）/ B（进了没打开）/ C（开了没买）/ D（买了但规格错）+ 预算硬门 | 只看总分 |
| 选点（若回 GRPO） | `scripts/select_grpo_checkpoint.py`（只用 val） | Final-200（那是考卷） |
| 改动安全性 | 数据重放 `scripts/verify_reward_v5.py` + preflight | 只靠单测 |

**顺序原则（用户已认可，继续执行）**：**先花 30 分钟做离线/小样本验证，再花 3–5 小时采集或 1.5 小时训练。**
本项目每一次跳过验证直接开跑，都付出了 2–3 小时的返工。

---

## 5. 工程坑（踩过的，别再踩）

**环境**
- 唯一入口 `bash scripts/setup.sh`（需 uv）。Linux `.venv/bin/python`，Windows `.venv/Scripts/python.exe`。
- Windows **不跑全量 pytest**；判回归需要同环境 HEAD 基线。已知非回归失败：
  `test_rollout` 5 / `test_observation_projection` 4 / `test_grpo_dynamic_sampling_config` 4（缺 hydra）/
  `test_search` 1（临时文件锁）/ `test_probe_group_diversity_filter` 3（临时目录权限）。
- Windows autocrlf 会让 `data/**` 变 CRLF 破坏 SHA。**文本哈希按 LF 归一**。
  `audit/reward-v4-hashes.txt` 是 CRLF 原始字节口径，与 manifest 不同源，别拿它校验。
- 同步用 `git fetch && reset --hard origin/feat/optimize`，别 `pull`。

**训练**
- 显存：单卡 RTX 6000D 84G，必须 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
  无 flash-attn（走 sdpa），`use_remove_padding` 不可开。
- checkpoint 每个 **~11 GB**（不是 4.2）。先 `df -h /root/rivermind-data`（`outputs` 是软链到该盘）。
- vLLM 初始化 **20–25 min**、util=0 是正常的，别以为卡死。
- 单步 ~132 s（`update_actor` 108 s 占 82%），50 步 ≈ 2.7 h。
- **训练数据顺序与 parquet 行序无关**（verl `shuffle:True` 默认）。按方差排序前排放置与提 lr 共线，破坏归因，不做。
- 步数的权威口径：`grep -ac optimizer_step training_diagnostics.jsonl`（别数 `step:` 行）。
- `trainer.logger: [console]`，无 wandb。`timing_s/` 下混有 token 数键（`response_length`/`prompt_length`），统计时要排除。
- **训练内 val 只有 4 个键**（`val-aux/num_turns/{max,mean,min}`、`val-core/shopsimulator/reward/mean@1`），
  没有 `purchase_success` / `r_option` ⇒ **判定必须走离线评测**。

**采集与评测**
- 采集和评测共用 `evaluation/rollout.py::collect_for_task`（同步）；GRPO 用 `training/grpo/adapter/`（异步）。
  共用底层 `environment/{tools,actions,observation,projection,client}.py`。
  ⇒ **改环境层两边同时生效**；改采集策略只影响采集。
- `collection/sft.py::acceptance_reasons` 目前**只接受 `gold_purchase`** ⇒ 池子 100% 正例，
  「合理放弃」示范为 **0 条**。这是 M10 必须改的第一处代码。
- 检索是 **BM25，词间 OR**（`" OR ".join`），召回深度 **150**，界面只渲染**前 20 名**。
  ⇒ "搜不到"和"排得靠后"是两件事，必须分开判（`scripts/audit_retrieval_reachability.py`）。
- `buy_now` **不带参数**，只能买当前打开的那个 ⇒ 任何"多开"策略在物理上都是"买最后一个"。

---

## 6. 可复用工具

| 用途 | 脚本 |
|---|---|
| 失败分桶 + 可达性（BM25 召回内/外，纯 CPU 自建索引） | `scripts/audit_retrieval_reachability.py` |
| 规格失败解剖（逐字/归一化/缺失 三分类） | `scripts/analyze_option_selection_failures.py` |
| 失败是"没看到"还是"看到了没买" | `scripts/analyze_selection_vs_retrieval.py` |
| 零方差组是复制品还是殊途同归 | `scripts/analyze_zero_variance_behaviour.py` |
| 采集选题（按失败类型挑未用商品） | `scripts/select_m8_collection_targets.py`（M10 需扩展） |
| 分组采样吞吐（n=8/16） | `scripts/analyze_group_size_sweep.py` |
| 训练诊断（kept 率 / 熔断） | `scripts/analyze_grpo_diagnostics.py <jsonl> --max-gen-batches 3` |
| 评测 | `scripts/evaluate.sh`、`scripts/run_eval_m9.sh` |

---

## 7. 当前唯一基线（不要记错）

| 项 | 值 |
|---|---|
| champion 检查点 | `outputs/models/sft-b-curriculum/stage-c/merged`（**勿删**） |
| Final-200 | **62.0%（124/200）** |
| 失败构成（**互斥**，合计 76） | A 20（gold 没进结果页）/ B 35（进了没打开）/ C 0 / D 18（买对但规格错）/ 无 gold 参考 3。其中流程失败 3 条（`repeat_loop` 2 + `max_steps` 1）**已含在内，不要另加** |
| 预算硬门失败 | 20 条（**与上面各桶交叉计数，不是并列**；13 条超支 ≤1.35×，属"近失"） |
| SFT 池 | `data/sft_pure_v4/all.jsonl`，**1,252 行，100% gold，0 条放弃** |
| GRPO 池 | train 906 / val 49（**当前不在主线上**） |
