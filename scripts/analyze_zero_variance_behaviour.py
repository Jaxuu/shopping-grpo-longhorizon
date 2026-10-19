#!/usr/bin/env python3
"""M8-0：零方差组里，4 条轨迹是「复制品」还是「殊途同归」？

为什么要有这个
--------------
M7 结案的核心证据是：116 个含 `wrong_purchase` 的多采样组里，**61.2% 在
`match_score × proximity` 轴上方差恰为 0**。方差恰为 0 意味着 4 条轨迹的
「买到的那个商品」完全相同 ⇒ 任何以最终商品属性为自变量的奖励函数在组内只能
输出同一个数 ⇒ 换 scale / 换 cap / 换轴都无效（Gate 3 实测梯度份额 0.1%，
门槛 5%）。

但「买的东西一样」**不等于**「过程一样」。两条路走到同一个错误终点，和两条
完全相同的路走到同一个错误终点，对下一步的含义截然相反：

* **过程也一样（复制品）** ⇒ 失败是完全确定性的，模型卡死在一个坑里。
  改奖励函数永远无效，只能动**采样多样性**（temperature / top_p）或**数据**。
* **过程不一样（殊途同归）** ⇒ 奖励函数仍有可用信号，只是轴选错了（选的是
  「终点」而不是「路径」）。此时**过程/行为级奖励**才有靶点。

本脚本用现成探针产物回答这个问题，**纯 CPU、零 GPU、零训练**。

关键陷阱：`think` 步是自由文本推理，几乎必然不同
------------------------------------------------
若把 `think` 的 `note` 计入动作签名，**每组都会显得「不一样」**，结论会被
彻底带偏。因此本脚本算两套签名：

* `act`  —— 排除 `think`，只看**真正作用于环境**的工具（搜索/翻页/开商品/
  选规格/看详情/购买/结束）；
* `full` —— 包含 `think`。

两套的差值本身就是信息：若 `replica_act` 高而 `replica_full` 低，说明
**「想的不一样，做的完全一样」**——这是 2B 模型很典型的形态，也是「过程奖励
该盯哪一层」的直接答案。

用法::

    # 主用法（探针产物：同题 4 采样）
    .venv/bin/python scripts/analyze_zero_variance_behaviour.py \
        outputs/probe/m6r-full/trajectories.jsonl \
        --out outputs/probe/m6r-full/zero_variance_behaviour.md

    # 机器可读
    .venv/bin/python scripts/analyze_zero_variance_behaviour.py <file.jsonl> --json

    # 调整判据（默认 0.70 / 0.30，见下）
    .venv/bin/python scripts/analyze_zero_variance_behaviour.py <file.jsonl> \
        --replica-high 0.70 --replica-low 0.30
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# ── 判据（写死并写进报告，避免事后调整口径）──────────────────────────────
# 零方差组里「动作序列完全相同」的占比：
#   ≥ HIGH  ⇒ 失败完全确定性，奖励路线终止，转采样多样性 / 数据
#   ≤ LOW   ⇒ 行为分歧充足，过程奖励有靶点
#   中间     ⇒ 混合，需按分歧位置再判
REPLICA_SHARE_HIGH = 0.70
REPLICA_SHARE_LOW = 0.30

WRONG_TYPE = "wrong_purchase"
# 排除 think 后剩下的「作用于环境」的工具；不在白名单里的也算 act（向前兼容新工具）。
NON_ACT_TOOLS = frozenset({"think"})
PURCHASE_TOOL = "buy_now"
OPEN_TOOL = "open_product"
SEARCH_TOOL = "search_products"
OPTION_TOOL = "select_option"

# 轴值比较容差：与 `analyze_wrong_purchase_variance.py` 的 Gate 2 口径一致。
AXIS_TOLERANCE = 1e-9


# ── 轨迹解析 ────────────────────────────────────────────────────────────
def _detail(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """`reward_detail` 可能在顶层，也可能挂在 `terminal_result` 下。"""
    terminal = row.get("terminal_result")
    if isinstance(terminal, Mapping):
        detail = terminal.get("reward_detail")
        if isinstance(detail, Mapping) and detail:
            return detail
    detail = row.get("reward_detail")
    if isinstance(detail, Mapping):
        return detail
    return terminal if isinstance(terminal, Mapping) else {}


def _steps(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    steps = row.get("steps")
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, Mapping)]


def _tool_name(step: Mapping[str, Any]) -> str:
    return str(step.get("tool_name") or step.get("tool") or "")


def _params(step: Mapping[str, Any]) -> Mapping[str, Any]:
    params = step.get("parameters")
    return params if isinstance(params, Mapping) else {}


def _step_signature(step: Mapping[str, Any]) -> str:
    name = _tool_name(step)
    params = _params(step)
    try:
        payload = json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = str(params)
    return f"{name}|{payload}"


def _finite(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def budget_proximity(required: object, actual: object) -> float:
    """`min(1, required/actual)`；不可算时返回 1.0（中性）。与 v5 实现同口径。"""
    if not isinstance(required, (int, float)) or not isinstance(actual, (int, float)):
        return 1.0
    if not math.isfinite(required) or not math.isfinite(actual) or actual <= 0:
        return 1.0
    return min(1.0, float(required) / float(actual))


def axis_value(row: Mapping[str, Any]) -> float:
    """`ms × prox`，与 Gate 2/3 的 `product` 轴完全一致。"""
    detail = _detail(row)
    gates = detail.get("hard_gates")
    gates = gates if isinstance(gates, Mapping) else {}
    budget = gates.get("budget")
    budget = budget if isinstance(budget, Mapping) else {}
    return _finite(detail.get("weighted_score")) * budget_proximity(
        budget.get("required"), budget.get("actual")
    )


def behaviour(row: Mapping[str, Any]) -> dict[str, Any]:
    """把一条轨迹压成行为指纹。"""
    steps = _steps(row)
    act: list[str] = []
    full: list[str] = []
    queries: list[str] = []
    opened: list[str] = []
    options: list[str] = []
    purchased: str | None = None
    for step in steps:
        name = _tool_name(step)
        signature = _step_signature(step)
        full.append(signature)
        if name not in NON_ACT_TOOLS:
            act.append(signature)
        if name == SEARCH_TOOL:
            query = _params(step).get("query")
            if isinstance(query, str):
                queries.append(query.strip().lower())
        elif name == OPEN_TOOL:
            asin = _params(step).get("asin")
            if isinstance(asin, str) and asin:
                opened.append(asin)
        elif name == OPTION_TOOL:
            value = _params(step).get("value")
            if isinstance(value, str):
                options.append(value)
        elif name == PURCHASE_TOOL and purchased is None and opened:
            purchased = opened[-1]
    return {
        "task_id": str(row.get("task_id")),
        "reward_type": str(_detail(row).get("reward_type", "unknown")),
        "axis": axis_value(row),
        "act": act,
        "full": full,
        "steps": len(steps),
        "queries": queries,
        "opened": opened,
        "options": options,
        "purchased_asin": purchased,
    }


# ── 组内比较 ────────────────────────────────────────────────────────────
def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _pairwise_jaccard(items: Sequence[Sequence[str]]) -> float | None:
    if len(items) < 2:
        return None
    scores = [
        _jaccard(items[i], items[j])
        for i in range(len(items))
        for j in range(i + 1, len(items))
    ]
    return statistics.fmean(scores) if scores else None


def _first_divergence(sequences: Sequence[Sequence[str]]) -> int | None:
    """相对第一条轨迹，最早在第几步出现差异；完全一致返回 None。"""
    if len(sequences) < 2:
        return None
    base = sequences[0]
    earliest: int | None = None
    for other in sequences[1:]:
        if list(other) == list(base):
            continue
        index = 0
        limit = min(len(base), len(other))
        while index < limit and base[index] == other[index]:
            index += 1
        earliest = index if earliest is None else min(earliest, index)
    return earliest


def classify_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """对一个同题组做分桶与分歧度量。"""
    prints = [behaviour(row) for row in rows]
    act_sequences = [tuple(item["act"]) for item in prints]
    full_sequences = [tuple(item["full"]) for item in prints]
    purchased = [item["purchased_asin"] for item in prints]

    replica_act = len(set(act_sequences)) == 1
    replica_full = len(set(full_sequences)) == 1
    if replica_act:
        bucket = "replica"
    elif len(set(purchased)) == 1 and purchased[0] is not None:
        bucket = "same_destination"
    else:
        bucket = "coincidence"

    step_counts = [item["steps"] for item in prints]
    return {
        "task_id": prints[0]["task_id"],
        "n": len(prints),
        "bucket": bucket,
        "replica_act": replica_act,
        "replica_full": replica_full,
        "axis_spread": (
            max(item["axis"] for item in prints) - min(item["axis"] for item in prints)
        ),
        "purchased_unique": len(set(purchased)),
        "distinct_act_sequences": len(set(act_sequences)),
        "distinct_full_sequences": len(set(full_sequences)),
        "first_divergence_act": _first_divergence(act_sequences),
        "first_divergence_full": _first_divergence(full_sequences),
        "query_jaccard": _pairwise_jaccard([item["queries"] for item in prints]),
        "open_jaccard": _pairwise_jaccard([item["opened"] for item in prints]),
        "option_jaccard": _pairwise_jaccard([item["options"] for item in prints]),
        "step_spread": max(step_counts) - min(step_counts),
        "steps_median": statistics.median(step_counts) if step_counts else 0,
    }


def build_groups(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        buckets[str(row.get("task_id"))].append(row)
    return dict(buckets)


def select_zero_variance_groups(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    min_group_size: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """切出三组：全 wrong 且轴方差为 0 / 全 wrong 但有方差 / 混档（不看）。"""
    zero: list[dict[str, Any]] = []
    nonzero: list[dict[str, Any]] = []
    mixed: list[dict[str, Any]] = []
    for rows in groups.values():
        if len(rows) < min_group_size:
            continue
        types = {behaviour(row)["reward_type"] for row in rows}
        stats = classify_group(rows)
        if types != {WRONG_TYPE}:
            mixed.append(stats)
        elif stats["axis_spread"] <= AXIS_TOLERANCE:
            zero.append(stats)
        else:
            nonzero.append(stats)
    key = lambda item: item["task_id"]
    return sorted(zero, key=key), sorted(nonzero, key=key), sorted(mixed, key=key)


# ── 汇总与裁定 ──────────────────────────────────────────────────────────
def _share(count: int, total: int) -> str:
    return f"{count}/{total}（{count / total:.1%}）" if total else f"{count}/0"


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def aggregate(
    zero: Sequence[Mapping[str, Any]], nonzero: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    buckets = Counter(item["bucket"] for item in zero)
    total = len(zero)
    replica = buckets.get("replica", 0)
    replica_share = replica / total if total else None
    divergence = [
        item["first_divergence_act"]
        for item in zero
        if item["first_divergence_act"] is not None
    ]
    return {
        "zero_groups": total,
        "nonzero_groups": len(nonzero),
        "buckets": dict(buckets),
        "replica": replica,
        "replica_share": replica_share,
        "replica_act": sum(1 for item in zero if item["replica_act"]),
        "replica_full": sum(1 for item in zero if item["replica_full"]),
        "replica_share_full": (
            sum(1 for item in zero if item["replica_full"]) / total if total else None
        ),
        "same_destination": buckets.get("same_destination", 0),
        "coincidence": buckets.get("coincidence", 0),
        "first_divergence_median": statistics.median(divergence) if divergence else None,
        "first_divergence_p25": (
            statistics.quantiles(divergence, n=4)[0] if len(divergence) >= 4 else None
        ),
        "step_spread_median": (
            statistics.median([item["step_spread"] for item in zero]) if total else None
        ),
        "query_jaccard_median": _median_of(zero, "query_jaccard"),
        "open_jaccard_median": _median_of(zero, "open_jaccard"),
        "option_jaccard_median": _median_of(zero, "option_jaccard"),
        "steps_median": _median_of(zero, "steps_median"),
    }


def _median_of(items: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [item[key] for item in items if item.get(key) is not None]
    return statistics.median(values) if values else None


def verdict(summary: Mapping[str, Any], *, high: float, low: float) -> dict[str, Any]:
    total = int(summary["zero_groups"])
    if not total:
        return {
            "status": "not_measured",
            "reason": "没有「全 wrong_purchase 且轴方差为 0」的组，本脚本不适用",
            "next": "用探针产物（同题 ≥2 采样）重跑",
        }
    share = float(summary["replica_share"])
    if share >= high:
        return {
            "status": "deterministic",
            "reason": f"动作序列完全相同的组占 {share:.1%} ≥ {high:.0%}："
            "4 次采样做的是同一件事，失败是确定性的",
            "next": "奖励函数路线到此终止；转「采样多样性（temperature/top_p）」或「补边界数据」",
        }
    if share <= low:
        return {
            "status": "behavioural_headroom",
            "reason": f"动作序列完全相同的组只占 {share:.1%} ≤ {low:.0%}："
            "4 次采样走了不同的路，只是终点一样",
            "next": "过程/行为级奖励有充足靶点；下一步设计「路径分」并先用本脚本量可及空间",
        }
    return {
        "status": "mixed",
        "reason": f"复制品占比 {share:.1%} 落在 [{low:.0%}, {high:.0%}]："
        "一部分组确定性失败，一部分组殊途同归",
        "next": "先看分歧首次出现在第几步（报告 §3）：分歧晚 ⇒ 前期路径已被固定，"
        "只能奖励「后半段的纠正」；分歧早 ⇒ 搜索策略本身有分歧，可奖励探索质量",
    }


# ── 报告 ────────────────────────────────────────────────────────────────
def render_report(
    summary: Mapping[str, Any],
    judgement: Mapping[str, Any],
    zero: Sequence[Mapping[str, Any]],
    nonzero: Sequence[Mapping[str, Any]],
    *,
    source: str,
    rows: int,
    examples: int = 15,
    empty_step_rows: int = 0,
) -> str:
    lines: list[str] = []
    lines.append("# 零方差组的行为分歧度（M8-0）")
    lines.append("")
    lines.append(f"> 输入：`{source}`，{rows} 条轨迹。")
    if empty_step_rows:
        lines.append(
            f"> 其中 **{empty_step_rows}** 条没有任何工具步（基础设施失败），已排除——"
            "否则它们会因为「动作序列都是空」被误判成 `replica`。"
        )
    lines.append(
        "> 口径：同题多采样组内**全部**为 `wrong_purchase` 且 `ms × prox` 方差为 0 ⇒ "
        "该组对任何「终点属性」奖励都是常数。"
    )
    lines.append(
        "> `act` 签名排除 `think`（自由文本推理），`full` 签名包含 `think`。"
    )
    lines.append("")

    lines.append("## 1. 分桶")
    lines.append("")
    lines.append("| 桶 | 含义 | 组数 | 占比 |")
    lines.append("|---|---|---:|---:|")
    total = int(summary["zero_groups"])
    bucket_text = {
        "replica": ("`replica`", "4 条动作序列**完全相同**（做的是同一件事）"),
        "same_destination": (
            "`same_destination`",
            "买了同一个 ASIN，但**路径不同**（殊途同归）",
        ),
        "coincidence": (
            "`coincidence`",
            "买的 ASIN **不同**却撞出同一个 `ms×prox`（打分分辨率问题）",
        ),
    }
    for key in ("replica", "same_destination", "coincidence"):
        count = int(summary["buckets"].get(key, 0))
        label, meaning = bucket_text[key]
        lines.append(f"| {label} | {meaning} | {count} | {_share(count, total)} |")
    lines.append("")
    lines.append(
        f"对照：全 wrong 但**有**方差的组 {summary['nonzero_groups']} 个（v5 能救的那批，"
        "不参与本节判定）。"
    )
    lines.append("")

    lines.append("## 2. 「想的不一样，做的完全一样」")
    lines.append("")
    lines.append("| 签名口径 | 完全一致的组数 | 占比 |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| `act`（排除 `think`） | {summary['replica_act']} | "
        f"{_share(int(summary['replica_act']), total)} |"
    )
    lines.append(
        f"| `full`（包含 `think`） | {summary['replica_full']} | "
        f"{_share(int(summary['replica_full']), total)} |"
    )
    lines.append("")
    act_share = summary["replica_share"] or 0.0
    full_share = summary["replica_share_full"] or 0.0
    if act_share - full_share >= 0.20:
        lines.append(
            f"**差值 {act_share - full_share:.1%}**：模型的**推理文本**在变，"
            "但**实际动作**几乎不变。⇒ 过程奖励若只奖励「想得对」而没有「做得不同」，"
            "仍可能拿不到梯度；要奖励就必须绑到可观测动作上。"
        )
    else:
        lines.append(
            f"**差值 {act_share - full_share:.1%}**：`think` 没有掩盖动作差异，"
            "两套口径结论一致。"
        )
    lines.append("")

    lines.append("## 3. 分歧发生在第几步（`act` 口径）")
    lines.append("")
    lines.append(
        f"- 首次分歧步数中位数：**{_fmt(summary['first_divergence_median'], 1)}**"
        f"（25 分位 {_fmt(summary['first_divergence_p25'], 1)}）"
    )
    lines.append(f"- 组内步数极差中位数：**{_fmt(summary['step_spread_median'], 1)}**")
    lines.append(f"- 组内步数中位数：**{_fmt(summary['steps_median'], 1)}**")
    lines.append("")
    lines.append(
        "读法：首次分歧在第 1~2 步 ⇒ 连搜索策略都不同；在 5 步以后 ⇒ 前期路径已被固定，"
        "分歧只在收尾。"
    )
    lines.append("")

    lines.append("## 4. 分歧的「内容」有多不同（两两 Jaccard，1.0 = 完全相同）")
    lines.append("")
    lines.append("| 维度 | 中位 Jaccard | 读法 |")
    lines.append("|---|---:|---|")
    lines.append(
        f"| 搜索词集合 | {_fmt(summary['query_jaccard_median'])} | "
        "低 ⇒ 连搜什么都不同；高 ⇒ 搜索策略一致 |"
    )
    lines.append(
        f"| 打开过的商品集合 | {_fmt(summary['open_jaccard_median'])} | "
        "低 ⇒ 看的是不同商品；高 ⇒ 看的商品一致，只差顺序 |"
    )
    lines.append(
        f"| 选过的规格集合 | {_fmt(summary['option_jaccard_median'])} | "
        "低 ⇒ 规格选择不同；高 ⇒ 规格一致 |"
    )
    lines.append("")

    if zero:
        lines.append(f"## 5. 零方差组明细（前 {min(examples, len(zero))} 个）")
        lines.append("")
        lines.append(
            "| task_id | 桶 | 步数 | 步数极差 | 买的 ASIN 数 | act 序列数 | "
            "首次分歧步 | 搜索 Jaccard | 打开 Jaccard |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for item in zero[:examples]:
            lines.append(
                f"| {item['task_id']} | `{item['bucket']}` | {item['steps_median']:.0f} | "
                f"{item['step_spread']} | {item['purchased_unique']} | "
                f"{item['distinct_act_sequences']} | "
                f"{_fmt(item['first_divergence_act'], 0)} | "
                f"{_fmt(item['query_jaccard'])} | {_fmt(item['open_jaccard'])} |"
            )
        lines.append("")

    lines.append("## 6. 裁定")
    lines.append("")
    lines.append(f"**{judgement['status']}**：{judgement['reason']}")
    lines.append("")
    lines.append(f"下一步：{judgement['next']}")
    lines.append("")
    return "\n".join(lines)


# ── 入口 ────────────────────────────────────────────────────────────────
def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def build_report(
    path: Path, *, min_group_size: int = 2, high: float = REPLICA_SHARE_HIGH,
    low: float = REPLICA_SHARE_LOW,
) -> dict[str, Any]:
    rows = load_rows(path)
    # 无工具步的轨迹是基础设施失败：动作序列都是空 ⇒ 会被误判成 replica，必须先剔掉。
    usable = [row for row in rows if _steps(row)]
    empty_step_rows = len(rows) - len(usable)
    groups = build_groups(usable)
    zero, nonzero, mixed = select_zero_variance_groups(
        groups, min_group_size=min_group_size
    )
    summary = aggregate(zero, nonzero)
    judgement = verdict(summary, high=high, low=low)
    return {
        "source": str(path),
        "rows": len(rows),
        "empty_step_rows": empty_step_rows,
        "groups": len(groups),
        "mixed_groups": len(mixed),
        "summary": summary,
        "verdict": judgement,
        "zero_groups": zero,
        "nonzero_groups": nonzero,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("trajectories", type=Path, help="探针轨迹 jsonl（同题多采样）")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而非 markdown")
    parser.add_argument("--out", type=Path, default=None, help="把报告写到该路径")
    parser.add_argument(
        "--min-group-size", type=int, default=2, help="至少几条同题轨迹才算一组（默认 2）"
    )
    parser.add_argument(
        "--replica-high", type=float, default=REPLICA_SHARE_HIGH,
        help=f"判「确定性失败」的复制品占比下限（默认 {REPLICA_SHARE_HIGH}）",
    )
    parser.add_argument(
        "--replica-low", type=float, default=REPLICA_SHARE_LOW,
        help=f"判「行为分歧充足」的复制品占比上限（默认 {REPLICA_SHARE_LOW}）",
    )
    parser.add_argument("--examples", type=int, default=15, help="明细表最多列几个组")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.trajectories.exists():
        print(f"找不到轨迹文件：{args.trajectories}", file=sys.stderr)
        return 2
    report = build_report(
        args.trajectories,
        min_group_size=args.min_group_size,
        high=args.replica_high,
        low=args.replica_low,
    )
    if args.json:
        payload = json.dumps(
            {
                "source": report["source"],
                "rows": report["rows"],
                "empty_step_rows": report["empty_step_rows"],
                "groups": report["groups"],
                "mixed_groups": report["mixed_groups"],
                "summary": report["summary"],
                "verdict": report["verdict"],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    else:
        payload = render_report(
            report["summary"],
            report["verdict"],
            report["zero_groups"],
            report["nonzero_groups"],
            source=report["source"],
            rows=report["rows"],
            examples=args.examples,
            empty_step_rows=report["empty_step_rows"],
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
        print(f"已写入 {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
