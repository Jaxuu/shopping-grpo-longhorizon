# Reward v4 变更说明（M3）

> **后续版本**：v5（M7）把 `wrong_purchase` 从常数 −0.85 改为 band `[−0.85, −0.66]`，
> 判定逻辑与本文全部不变。见 `docs/reward-v5.md`。
>
> 配套：`plans/task.md` §M3、`plans/optimize.md` §M3、`audit/reward-v4-hashes.txt`（冻结哈希）。
> 版本标识：`shopsimulator-reward-v3` → `shopsimulator-reward-v4`（P2）。预算解析器：`shopping-budget-parser-v4`（M1-R 已落地，本轮未改动行为）。

## 0. 一句话总结

**Reward v4 = v3 的全部 hard gate 语义（逐字节不变）+ 解析覆盖扩展（M1-R 修 bug）+ 两个纯信息层（soft 价格接近度、过程证据）+ 版本标识同步（R11 闭合）。**

## 1. 修 bug（可宣传，P4）

| # | 内容 | 落点 | 状态 |
|---|---|---|---|
| 1 | 预算解析覆盖：中文数字（含"两千五"省略末尾单位）、口语单位（块/块钱/圆）、无前缀区间、带单位价格、万缩写、非 ASCII 空白归一化；价格语境门控拒绝 `SPF50+`/`6g`/`60厘米` | `engine/constraints.py::explicit_budget_from_instruction`（`shopping-budget-parser-v4`） | **M1-R 已完成**，本轮零改动 |
| 2 | `\xa0`/全角空格规格匹配 | `engine/reward_features.py::normalize_option_text`（NFKC，全程匹配路径已覆盖） | **M1-R 核实无需改代码**；本轮以 `tests/test_reward_v4.py::EvaluationTask3647WhitespaceTest`（真实评测集 `task_id=3647`）固化为回归 |
| 3 | **R11：Reward 行为已变但版本标识未同步**（M1 遗留未闭合风险） | 见 §3 | **本轮闭合** |

## 2. 改标准（须单独标注，P4）

| # | 内容 | 性质 | 关键约束 |
|---|---|---|---|
| 1 | **soft 价格接近度**：为"左右/上下"类软偏好恢复名义项（`价格控制在450元左右` → 450）并输出 `proximity = max(0, 1 - \|actual-nominal\|/nominal)` | **纯信息层**：只写入 `evidence.soft_price_preference`，**不参与** hard gate、`match_score`、终局奖励与 `reward_type` 判定 | hard budget 仍消费冻结的 v3 +10% 容差（`explicit_budget_from_instruction("价格控制在450元左右") = 495.0`，测试固定）；有/无软偏好时终局奖励逐字节一致（测试固定） |
| 2 | **过程证据指标**：`evidence.process_evidence`（`shopping-process-evidence-v1`），含 hard gate 状态、`match_score`、`evidence_coverage`、四维分数、soft 价格接近度、`target_asin_match` | **纯信息层**：为 M6 turn-level credit / 过程奖励铺路，不改终局奖励 | 只加字段，不改任何既有字段语义 |
| 3 | `build_task_facts` 增加 `process_evidence` 快照（rubric 侧同名约定） | 评测侧消费口 | `TASK_FACTS` 哈希随之变化；Rubric bundle 按次生成，无冻结件受影响 |

> **为何 soft 维度不进终局奖励**：若并入 `DIMENSION_WEIGHTS`，"左右"题的 gold/valid_alternative/partial 分类会变，等价于改判定标准且使 M1 冻结的准入判定与历史轨迹全部失效。M3 的预期收益（间接 +3~8pp）来自解析覆盖，不依赖此项；真正消费 soft 维度是 M6 过程奖励档位的职责。

## 3. 版本化与双层口径（M3-8 / R11 闭合）

**运行时输出**：`engine/reward.py::REWARD_VERSION = "shopsimulator-reward-v4"`，所有新 rollout 的 `reward_detail.reward_version` 为 v4。

**双层口径（接受 v3 与 v4）**——历史 v3 轨迹仍可读，M7-5 依赖：

| 消费端 | 文件 | 改动 |
|---|---|---|
| 环境配置校验 | `engine/config.py::validate_config` | reward version ∈ {v3, v4} |
| Manifest 校验 | `src/shopping_grpo/environment/manifest.py::validate_manifest` | 同上（报错文案同步） |
| 严格成功判定 | `evaluation/metrics.py`（`_strict_success`、`unexpected_reward_version`） | 接受 {v3, v4} |
| 汇总口径 | `evaluation/summary.py` | `gold_purchases`/`reward_valid_tasks`/`_is_strict_success` 接受双层；`reward_contract` 字段改为 `"shopsimulator-reward-v3+shopsimulator-reward-v4"` |
| SFT 采集验收 | `collection/sft.py::acceptance_reasons` | 接受双层；reject 标识符沿用历史字符串 `reward_v3_*`（避免破坏既有聚合） |
| GRPO adapter | `adapter/runtime.py::validate_reward / reward_breakdown`、`adapter/tools.py` | 接受双层；`validate_reward` 回显原始版本号 |

**同步的冻结标识**：`data/environment.json` 与 `environments/.../configs/environment.json` 的 `reward.version` → v4；`data/environment.json` 的 `runtime_files_sha256["reward.py"]` 按新文件重算（`scripts/check_grpo_runtime.py` 的 preflight 依赖）。`data/{grpo,sft_pure_v4}/metadata.json` 新增 `reward_v4` 块（版本号 + 解析器 + reward.py 哈希）。

**未改动的冻结件**：`data/evaluation/`（评测集 + metadata，SHA 冻结）、`src/shopping_grpo/resources/blind_guard.json` 与 `blind_guard.py`（blind 资产契约标识属于 v3 时代冻结资产，属 M7-5 的 v3 一侧）、`audit/admission_verdicts.jsonl` 等全部 M1 审计产物。

## 4. 离线重放证据（M3-T7，23,421 goal 全语料）

对"v3 能解析的预算"做逐条值比对（`tests/test_reward_v4.py::BudgetValueIdentityV3V4Test`）：

| 漂移类别 | 全语料 | 四数据集内 | 归因 |
|---|---:|---:|---|
| 软后缀补全（v4 = v3×1.1） | 1,381 | 评测集 2 题（21643: 10→11、22547: 15→16.5） | v3 因口语单位（块/上下）漏掉左右后缀，v4 按冻结 +10% 容差补全 |
| 向上重解析（v4 > v3） | 33 | grpo_train 3 + sft 1 | v4 覆盖 v3 截断的中文数字（如"四千"=4000） |
| **向下重解析（v4 < v3）** | 18 | **0** | v4 已知解析缺陷（见 §5），不触及四个数据集 |

**结论**：冻结评测集的漂移全部是"新解析出的预算"且方向为冻结容差规则；训练池无上限压低。hard gate 判定语义（`actual <= upper`、`upper is None` 默认放行、comparator `variant_price_budget_v1`）与 v3 一致，已用单测固定。

## 5. 遗留：v4 解析器缺陷清单（M4 输入，本轮不修）

以下缺陷经全语料重放确认，**均不触及四个数据集的预算上限压低**（对评测集影响为零），按 P8 不在 M3 中途修改冻结解析器（会使 M1 准入判定与池重建失效）；M4 统一重跑准入时一并处理：

1. **万缩写 + 左右漏容差**：`预算1万5左右` → v4 = 15000，v3 = 16500（v3 对万缩写应用了 ×1.1）。全语料 3 例，0 例在数据集内。
2. **软匹配优先级**：`_AROUND`（左右）先于硬上限（不超过/以内）匹配，混合句式会抓错跨度（如"每克250元左右…总价不超过500元" → v4 = 275，v3 = 500）。0 例在数据集内造成压低。
3. **混合数字**：`3百元以内` → v4 = 100（孤字"百"被当成 100），应为 300；`1千2百多元` 类混合数位同样不支持。0 例在数据集内。
4. grpo_train/sft 内 4 例向上重解析中的 2 例（如 sft 8053：`130块左右的鱼肉段，预算在50元左右` → v4 = 143）属"软匹配抓错跨度"，方向放宽，不影响 gold 准入的安全性，但建议 M4 复核。

## 6. 测试与验收对照

| 测试项 | 结果 |
|---|---|
| M3-T1~T4、T6、T8（解析覆盖/歧义/解析率） | ✅ `tests/test_budget_parser_v4.py`（全池 93.5% ≥90%，评测集 ≥93.5%，零覆盖回退） |
| M3-T5（`\xa0` 规格匹配） | ✅ `tests/test_reward_v4.py::EvaluationTask3647WhitespaceTest`（真实语料） |
| M3-T7（hard gate 重放） | ✅ `HardGateSemanticsTest` + `BudgetValueIdentityV3V4Test`（§4 分类不变式） |
| M3-T9（回归） | ✅ `test_evaluation_badcase` / `test_evaluation_dataset` / `test_benchmark` / `test_budget_parser_v4` / `test_admission` / `test_environment_manifest` / `test_sft_collection` / `test_data_contracts` / `test_audit_solvability` + `test_reward_v4` 共 **84 passed** |
| M3-T10（哈希冻结） | ✅ `audit/reward-v4-hashes.txt`；两处 `environment.json` = v4；两个池 metadata 含 `reward_v4` 块 |

> 注：`tests/test_shopping_reward.py` 预存导入错误（§0.4），按计划不使用、不修复（超出 M3 边界）。
