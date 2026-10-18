# Reward v5 变更说明（M7）

> 配套：`plans/m7-decision.md`（定臂）、`plans/m7-implementation.md`（实现方案）、
> `audit/reward-v5-hashes.txt`（冻结哈希）。
> 版本标识：`shopsimulator-reward-v4` → `shopsimulator-reward-v5`。

## 0. 一句话总结

**v5 = v4 的全部判定逻辑（逐字节不变）+ 一个数值分支的连续化：`wrong_purchase`
从常数 −0.85 改为 band `[−0.85, −0.66]`，由 `match_score × 预算接近度` 驱动。**

动机：M6/M6-R 两轮 GRPO 的失败解剖显示，失败的 34.2% 是「零梯度失败」，
其中 23 题（11.5%）是 `wrong_purchase`——终局奖励恒为 −0.85，组内 4 条采样
reward 全等 ⇒ 被 `dynamic_sampling` 以 `drop_reason="constant_reward"` 丢弃，不产生任何 advantage。

## 1. 精确规格

```
prox   = min(1, required / actual)              # 预算门的 required/actual
driver = match_score × prox                     # match_score == weighted_score
wrong  = min(cap, base + scale × driver)
         base = -0.85, scale = 0.20, cap = -0.66
```

| 常量 | 值 | 作用 |
|---|---:|---|
| `wrong_purchase` | −0.85 | 保留为 `driver = 0` 的退化值（与 v4 一致），文档值与地板 |
| `wrong_purchase_base` | −0.85 | band 地板 |
| `wrong_purchase_scale` | 0.20 | 斜率；`driver = 1` 时恰好触及 cap |
| `wrong_purchase_cap` | −0.66 | band 带顶，**必须 < `repeat_loop`(−0.65)** |

**为什么 `cap` 不是原方案的 −0.45**：−0.45 高于 `max_steps`(−0.50) 与 `repeat_loop`(−0.65)，
会让「买错但匹配度高」好于「步数耗尽 / 重复循环」，等于恢复 v3 设计要抑制的
**恐慌性购买**（`shop_env/tests/test_reward.py::test_reward_order_prevents_panic_buying`
正是守这条）。−0.66 把带内跨度上限锁在 0.19，代价是梯度量级减半，但保住了档位序。

**为什么轴是 `ms × prox` 而不是纯 `ms`**：23 题里 20 题是**超预算**失败，超支幅度
1.02×–3.18×，且 `ms` 与超支幅度 Spearman **+0.42**（越像正确答案的商品往往越贵）。
纯 `ms` 会把「几乎全对但超支 3.2 倍」排到接近带顶，等于教模型无视预算。

## 2. 改了什么 / 没改什么

**改（1 个分支）**：`evaluate_purchase` 的 `hard_failed` 分支。

**没改（一律逐字节不变）**：

| 项 | 状态 |
|---|---|
| `reward_type` 分类（9 个类型） | 不变 |
| hard gate 语义（`category` + `budget`，严格 `actual <= upper`） | 不变 |
| `purchase_success` | 不变（由 `reward_type` 决定） |
| `reward_valid` / `sampling_invalid` | 不变 |
| gold 1.0 / valid_alt 0.55 / partial 三常量 / graceful −0.15 / early −0.35 / max_steps −0.50 / repeat_loop −0.65 / unverifiable 0.0 | 不变 |
| `evidence` 结构（`preference_scoring` / `process_evidence` / `soft_price_preference`） | 不变（不加字段，避免动 `TASK_FACTS`） |
| 准入判定 `evaluate_candidate_eligibility` | 不变（不产生终局奖励） |

## 3. 三层版本口径

`SUPPORTED_REWARD_VERSIONS` 由「v3+v4」扩为「v3+v4+v5」，**只加不删**：历史 v3/v4
轨迹必须仍可读。消费端共 10 处，漏改的后果见 `plans/m7-implementation.md §4.2`；
最危险的是 `evaluation/metrics.py`（漏改则 `_strict_success` 恒 False）。

## 4. 验证

| 层 | 内容 |
|---|---|
| 单元 | `tests/test_reward_v5.py`：band 边界、`driver=0 → −0.85`、单调性、`prox` 退化、档位序、v4/v5 差分只差 `wrong_purchase` |
| 数据重放 | `scripts/verify_reward_v5.py`：对已有轨迹逐条重算，断言非 `wrong_purchase` 行**逐位不变**、`reward_type`/`purchase_success`/strict success 不变、变化行全部落在 band 内 |
| 契约 | `tests/test_environment_manifest.py`（哈希）、`scripts/check_grpo_runtime.py`（preflight 必须打印 `reward_v5`） |

`driver = 0` 的行**本来就不变**，所以「变化行数」只能 ≤ `wrong_purchase` 行数，
不能要求相等（`scripts/verify_reward_v5.py` 的口径）。

## 5. 不可比性与测量口径

**不可比**：`mean_final_reward`、`val-core/.../reward/mean@1` 与历史不同标尺
（同一策略、同一 checkpoint，失败样本从 −0.85 变成 −0.66..−0.85，均值机械上移）。
**可比**：`purchase_success` 率、`gold_purchase` 率、strict success——均为分类量。

⚠️ **主指标必须用 `purchase_success`**。用 `reward/mean@1` 选点还有一层隐藏偏置：
策略越好 `wrong_purchase` 越少 ⇒ 机械加成越小 ⇒ 会系统性惩罚更好的 checkpoint。

## 6. 冻结标识

| 文件 | 改动 |
|---|---|
| `data/environment.json` | `reward.version` → v5；补三个常量；`runtime_files_sha256["reward.py"]` 重算 |
| `environments/ShopSimulator/shop_env/configs/environment.json` | 同上（值必须与 `DEFAULT_REWARDS` 逐值相等，`config.py::validate_config` 强制） |
| `data/{grpo,sft_pure_v4}/metadata.json` | 新增 `reward_v5` 块；**保留 `reward_v4` 块**作为建池溯源 |
| `audit/reward-v5-hashes.txt` | 新建（LF 归一化口径）；**不覆盖** `reward-v4-hashes.txt` |

**哈希重算口径**（改 `reward.py` 后必做，否则 `check_grpo_runtime.py` 直接退出）：

```bash
.venv/bin/python - <<'PY'
import hashlib, pathlib
p = pathlib.Path("environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py")
print(hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest())
PY
```

必须在 Linux / LF 检出上算。Windows（`core.autocrlf=true`）的原始字节哈希与
manifest 口径不同，会把 preflight 弄红。
