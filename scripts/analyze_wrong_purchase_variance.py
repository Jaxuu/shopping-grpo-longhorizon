#!/usr/bin/env python3
"""`wrong_purchase` 档位内的排序信息：Gate 2 判据 + v5 候选公式对比。

为什么要有这个
--------------
终局奖励 v4 把 `wrong_purchase` 压成常数 −0.85。M6/M6-R 两轮 GRPO 都死在
「全成功 / 零方差组把可训练组吃空 → `dynamic_sampling` 熔断」，而失败解剖显示
**零梯度失败 26/76 题（34.2%）里 23 题就是 `wrong_purchase`**。M7 的唯一变量因此
定为「把这一档按内部匹配度连续化」（v5）。

但「23 题之间有 12 个取值」只证明**题与题之间**有差异，还差最后一环：
**同一题、同一策略的 4 条采样之间**是否有差异。若 4 条都买了同一个商品，
`match_score` 相同，v5 只是把常数换成另一个常数 —— 零梯度照旧。

本脚本回答两件事：

1. **Gate 2（生死判据）**：同题多采样组的组内标准差中位数是否 ≥ 0.05。
2. **公式对比**：三个候选 v5 公式在这批题上分别产生多少分辨率、
   是否破坏既有档位序（`wrong_purchase` 必须仍是**最差**的非 unverifiable 档位）。

两个候选轴
----------
- `match_score`（`ms`）：`weighted_score`，v4 已计算并落盘，衡量「买对了多少」。
- `budget_proximity`（`prox`）：`min(1, required/actual)`，衡量「超预算多少」。

⚠️ **为什么可能需要第二根轴**：在 M5-B 的 23 题上 `ms` 与超支幅度
（`actual/required`）的 Spearman 相关是 **+0.42**（正相关，不是正交）——
「越像正确答案的商品往往越贵」。只用 `ms` 会把「买了完全正确的商品但超预算
3.2 倍」排到接近带顶，等于教模型无视预算。`ms × prox` 把这两件事分开。

用法::

    # Gate 2（探针产物：同题 4 采样）
    .venv/bin/python scripts/analyze_wrong_purchase_variance.py \
        outputs/probe/m6r-full/trajectories.jsonl --out outputs/probe/m6r-full/wp_variance.md

    # 单采样评测集：只看公式对比（组内方差自动跳过）
    .venv/bin/python scripts/analyze_wrong_purchase_variance.py \
        outputs/evaluation/sft-b-curriculum/trajectories.jsonl

    # 机器可读
    .venv/bin/python scripts/analyze_wrong_purchase_variance.py <file.jsonl> --json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

# ── 判据（写死并写进报告，避免事后调整口径）──────────────────────────────
# Gate 2：全 wrong 组的轴值组内标准差中位数。
GROUP_STD_PASS = 0.05
GROUP_STD_MARGINAL = 0.02
# 边际区间的补充条件：能被 v5 从「丢弃」转为「可训练」的组占比。
RECOVERABLE_SHARE = 0.05

WRONG_TYPE = "wrong_purchase"

# v4 的既有档位阶梯（除 wrong_purchase 外全部冻结，v5 不许越界）。
FROZEN_LADDER = {
    "gold_purchase": 1.0,
    "valid_alternative_purchase": 0.55,
    "partial_alternative_purchase_cap": 0.25,
    "graceful_stop": -0.15,
    "partial_alternative_purchase_floor": -0.30,
    "early_abstain": -0.35,
    "max_steps": -0.50,
    "repeat_loop": -0.65,
}
WRONG_FLOOR = -0.85  # v4 常数，v5 在 ms=0 时必须与之一致


# ── 候选公式（与 partial 同形：min(cap, base + scale × axis)）──────────────
def wrong_purchase_reward(
    match_score: float,
    proximity: float,
    *,
    scale: float,
    cap: float,
    axis: str = "product",
) -> float:
    """按候选公式算 `wrong_purchase` 的奖励。

    ``axis``：

    * ``"match"``   —— 只用 ``match_score``（m7-decision.md §4 的原方案轴）；
    * ``"product"`` —— 用 ``match_score × proximity``，把「买对了多少」与
      「超预算多少」分开计分。

    ``cap`` 是带顶（最大值），``WRONG_FLOOR`` 是带底（``axis=0`` 时的退化值）。
    """
    if axis == "match":
        driver = float(match_score)
    elif axis == "product":
        driver = float(match_score) * float(proximity)
    else:
        raise ValueError(f"unknown axis: {axis!r}")
    return min(float(cap), WRONG_FLOOR + float(scale) * driver)


CANDIDATES: dict[str, dict[str, Any]] = {
    # m7-decision.md §4 的现方案：带顶 −0.45 高于 repeat_loop(−0.65) 与 max_steps(−0.50)。
    "A_ms_only_wide": {"scale": 0.55, "cap": -0.45, "axis": "match"},
    # 保序、单轴：跨度被压在 0.12 附近，不越界。
    "B_ms_only_safe": {"scale": 0.15, "cap": -0.66, "axis": "match"},
    # 保序、双轴：21 个不同取值，且把「离谱超支」沉到中下游。
    "C_ms_x_prox": {"scale": 0.20, "cap": -0.66, "axis": "product"},
}


def _candidate_value(name: str, match_score: float, proximity: float) -> float:
    spec = CANDIDATES[name]
    return wrong_purchase_reward(
        match_score,
        proximity,
        scale=spec["scale"],
        cap=spec["cap"],
        axis=spec["axis"],
    )


# ── 数据提取 ────────────────────────────────────────────────────────────
def budget_proximity(required: object, actual: object) -> float:
    """``min(1, required/actual)``；不可算时返回 1.0（中性，不惩罚）。

    预算门 PASS 时 ``required/actual > 1``，被截到 1.0 ⇒ 该项不产生任何影响；
    门 FAIL 时它才是真正的「超支折扣」。未声明预算（``required is None``）
    同样返回 1.0，此时 ``axis="product"`` 退化成 ``axis="match"``。
    """
    if not isinstance(required, (int, float)) or not isinstance(actual, (int, float)):
        return 1.0
    if not math.isfinite(required) or not math.isfinite(actual) or actual <= 0:
        return 1.0
    return min(1.0, float(required) / float(actual))


def _reward_detail(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """轨迹里的 ``reward_detail`` 可能在顶层，也可能挂在 ``terminal_result`` 下。"""
    detail = row.get("reward_detail")
    if isinstance(detail, Mapping):
        return detail
    terminal = row.get("terminal_result")
    if isinstance(terminal, Mapping):
        nested = terminal.get("reward_detail")
        if isinstance(nested, Mapping):
            return nested
    return {}


def _task_id(row: Mapping[str, Any], detail: Mapping[str, Any]) -> str:
    for source in (row, detail):
        value = source.get("task_id")
        if value is not None:
            return str(value)
    return "unknown"


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def extract_wrong_purchases(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """抽出所有 ``reward_type == wrong_purchase`` 的记录（含 ms / prox 两轴）。"""
    records: list[dict[str, Any]] = []
    for row in rows:
        detail = _reward_detail(row)
        if detail.get("reward_type") != WRONG_TYPE:
            continue
        gates = detail.get("hard_gates")
        gates = gates if isinstance(gates, Mapping) else {}
        budget = gates.get("budget")
        budget = budget if isinstance(budget, Mapping) else {}
        category = gates.get("category")
        category = category if isinstance(category, Mapping) else {}
        match_score = _finite_float(detail.get("weighted_score"))
        records.append(
            {
                "task_id": _task_id(row, detail),
                "match_score": match_score,
                "proximity": budget_proximity(
                    budget.get("required"), budget.get("actual")
                ),
                "category_status": category.get("status"),
                "budget_status": budget.get("status"),
                "target_asin_match": bool(detail.get("target_asin_match")),
                "reward_v4": _finite_float(
                    detail.get("terminal_utility", detail.get("reward")), WRONG_FLOOR
                ),
            }
        )
    return records


def reward_type_counts(rows: Iterable[Mapping[str, Any]]) -> Counter:
    return Counter(
        str(_reward_detail(row).get("reward_type", "unknown")) for row in rows
    )


# ── Gate 2：组内方差 ─────────────────────────────────────────────────────
def group_spread(
    records: Sequence[Mapping[str, Any]], axis: str
) -> dict[str, Any]:
    """按 ``task_id`` 分组，算组内总体标准差；只统计组大小 ≥ 2 的组。

    ``axis="match"`` 取 ``match_score``，``axis="product"`` 取 ``ms × prox``。
    单采样评测集（每组 1 条）返回 ``groups=0``，调用方据此跳过 Gate 2。
    """
    buckets: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = float(record["match_score"])
        if axis == "product":
            value *= float(record["proximity"])
        elif axis != "match":
            raise ValueError(f"unknown axis: {axis!r}")
        buckets[str(record["task_id"])].append(value)
    spreads = [
        statistics.pstdev(values) for values in buckets.values() if len(values) >= 2
    ]
    spreads.sort()
    return {
        "axis": axis,
        "groups": len(spreads),
        "median": statistics.median(spreads) if spreads else None,
        "mean": statistics.fmean(spreads) if spreads else None,
        "zero_share": (
            sum(1 for value in spreads if value <= 0.0) / len(spreads)
            if spreads
            else None
        ),
        "spreads": spreads,
    }


def gate2_verdict(spread: Mapping[str, Any]) -> dict[str, Any]:
    """把组内标准差中位数翻成 pass / marginal / fail。"""
    if not spread.get("groups"):
        return {
            "status": "not_measured",
            "reason": "没有同题多采样组（单采样评测集），Gate 2 需用探针产物",
        }
    median = float(spread["median"])
    if median >= GROUP_STD_PASS:
        status, reason = "pass", f"中位标准差 {median:.4f} ≥ {GROUP_STD_PASS}"
    elif median >= GROUP_STD_MARGINAL:
        status, reason = (
            "marginal",
            f"中位标准差 {median:.4f} ∈ [{GROUP_STD_MARGINAL}, {GROUP_STD_PASS})，"
            f"需「可回收组占比」≥ {RECOVERABLE_SHARE:.0%} 才开跑",
        )
    else:
        status, reason = (
            "fail",
            f"中位标准差 {median:.4f} < {GROUP_STD_MARGINAL}：该轴组内无差异，"
            "v5 换公式也造不出梯度",
        )
    return {"status": status, "reason": reason}


# ── 公式对比 ────────────────────────────────────────────────────────────
def formula_summary(
    records: Sequence[Mapping[str, Any]], name: str
) -> dict[str, Any]:
    """某候选公式在这批题上的分辨率与越界情况。"""
    values = [
        _candidate_value(name, float(r["match_score"]), float(r["proximity"]))
        for r in records
    ]
    if not values:
        return {"name": name, "values": [], "distinct": 0, "top": None, "bottom": None,
                "span": 0.0, "violations": []}
    top, bottom = max(values), min(values)
    violations = sorted(k for k, v in FROZEN_LADDER.items() if top > v)
    return {
        "name": name,
        "values": values,
        "distinct": len({round(value, 6) for value in values}),
        "top": top,
        "bottom": bottom,
        "span": top - bottom,
        "violations": violations,
        "degenerates_at_zero": math.isclose(bottom, WRONG_FLOOR, abs_tol=1e-9),
    }


# ── 报告 ────────────────────────────────────────────────────────────────
def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _share(count: int, total: int) -> str:
    return f"{count}/{total}（{count / total:.1%}）" if total else f"{count}/0"


def render_report(
    counts: Counter,
    records: Sequence[Mapping[str, Any]],
    spreads: Mapping[str, Mapping[str, Any]],
    verdicts: Mapping[str, Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    *,
    source: str,
) -> str:
    lines: list[str] = []
    total = sum(counts.values())
    wrong_total = counts.get(WRONG_TYPE, 0)

    lines.append("# `wrong_purchase` 内部排序信息（Gate 2 + v5 公式对比）")
    lines.append("")
    lines.append(f"> 输入：`{source}`，{total} 条轨迹。")
    lines.append("")
    lines.append("## 1. 总览")
    lines.append("")
    lines.append("| reward_type | 条数 | 占比 |")
    lines.append("|---|---:|---:|")
    for name, count in counts.most_common():
        lines.append(f"| `{name}` | {count} | {count / total:.1%} |")
    lines.append("")
    lines.append(
        f"`wrong_purchase` 共 **{wrong_total}** 条，v4 全部为常数 "
        f"{WRONG_FLOOR}（零梯度）。"
    )
    lines.append("")

    if records:
        zero = sum(1 for r in records if r["match_score"] <= 0.0)
        full = sum(1 for r in records if math.isclose(r["proximity"], 1.0))
        lines.append("## 2. 内部解剖")
        lines.append("")
        lines.append("| 观察 | 数值 |")
        lines.append("|---|---|")
        lines.append(
            f"| `match_score` 跨度 | {_fmt(min(r['match_score'] for r in records), 3)} "
            f"→ {_fmt(max(r['match_score'] for r in records), 3)}，"
            f"均值 {_fmt(statistics.fmean(r['match_score'] for r in records), 3)} |"
        )
        lines.append(f"| `match_score = 0` 的题 | {_share(zero, len(records))} |")
        lines.append(
            f"| 预算未超（`proximity = 1.0`）的题 | {_share(full, len(records))} "
            "（即失败原因不是超预算） |"
        )
        gate_pairs = Counter(
            (str(r["category_status"]), str(r["budget_status"])) for r in records
        )
        lines.append(
            "| hard gate 组合 | "
            + "、".join(f"category={c}/budget={b}：{n}" for (c, b), n in gate_pairs.most_common())
            + " |"
        )
        lines.append(
            f"| `target_asin_match = True` | "
            f"{_share(sum(1 for r in records if r['target_asin_match']), len(records))}"
            "（买的就是 gold ASIN，只是没通过硬门） |"
        )
        lines.append("")

    lines.append("## 3. Gate 2：组内方差")
    lines.append("")
    lines.append("判据：同题多采样组的轴值组内标准差**中位数**。")
    lines.append("")
    lines.append("| 轴 | 可测组数 | 中位标准差 | 均值 | 零方差组占比 | 裁定 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for axis, spread in spreads.items():
        verdict = verdicts[axis]
        zero_share = spread["zero_share"]
        zero_text = "—" if zero_share is None else f"{zero_share:.1%}"
        lines.append(
            f"| `{axis}` | {spread['groups']} | {_fmt(spread['median'])} | "
            f"{_fmt(spread['mean'])} | {zero_text} | "
            f"**{verdict['status']}** |"
        )
    lines.append("")
    for axis, verdict in verdicts.items():
        lines.append(f"- `{axis}`：{verdict['reason']}")
    lines.append("")

    lines.append("## 4. v5 候选公式对比")
    lines.append("")
    lines.append("全部与 `partial_alternative_purchase` 同形：`min(cap, −0.85 + scale × 轴)`。")
    lines.append("")
    lines.append("| 候选 | 轴 | scale | cap | 不同取值数 | 带顶 | 带底 | 跨度 | 越界档位 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for summary in summaries:
        spec = CANDIDATES[summary["name"]]
        violations = summary["violations"]
        lines.append(
            f"| `{summary['name']}` | `{spec['axis']}` | {spec['scale']} | {spec['cap']} | "
            f"{summary['distinct']} | {_fmt(summary['top'])} | {_fmt(summary['bottom'])} | "
            f"{_fmt(summary['span'], 3)} | "
            f"{'无（保序）' if not violations else '、'.join(violations)} |"
        )
    lines.append("")
    lines.append(
        "> 「越界档位」= 该公式的带顶高于哪些冻结档位。非空即说明 `wrong_purchase` "
        "不再是**最差**的非 unverifiable 结果，档位序被改动。"
    )
    lines.append("")

    if records:
        lines.append("### 4.1 逐题对照（按 `ms × prox` 从好到坏）")
        lines.append("")
        header = "| task_id | category | budget | `ms` | `prox` | `ms×prox` |"
        separator = "|---|---|---|---:|---:|---:|"
        for summary in summaries:
            header += f" `{summary['name']}` |"
            separator += "---:|"
        lines.append(header)
        lines.append(separator)
        ordered = sorted(
            records, key=lambda r: -(r["match_score"] * r["proximity"])
        )
        for record in ordered:
            row = (
                f"| {record['task_id']} | {record['category_status']} | "
                f"{record['budget_status']} | {_fmt(record['match_score'], 3)} | "
                f"{_fmt(record['proximity'], 3)} | "
                f"{_fmt(record['match_score'] * record['proximity'], 3)} |"
            )
            for summary in summaries:
                value = _candidate_value(
                    summary["name"], record["match_score"], record["proximity"]
                )
                row += f" {_fmt(value)} |"
            lines.append(row)
        lines.append("")

    lines.append("## 5. 裁定")
    lines.append("")
    product_verdict = verdicts.get("product", {})
    match_verdict = verdicts.get("match", {})
    if product_verdict.get("status") == "not_measured":
        lines.append(
            "**Gate 2 未测**：输入是单采样评测集，组内方差算不出来。"
            "本报告只用于公式对比；Gate 2 必须用同题多采样的探针产物重跑。"
        )
    else:
        best = product_verdict if product_verdict.get("status") != "fail" else match_verdict
        lines.append(f"**Gate 2（`ms×prox` 轴）= {product_verdict.get('status')}**：{product_verdict.get('reason')}")
        lines.append("")
        lines.append(f"**Gate 2（`ms` 轴）= {match_verdict.get('status')}**：{match_verdict.get('reason')}")
        lines.append("")
        lines.append(f"⇒ 采用 `{best.get('status')}` 的轴开跑。")
    lines.append("")
    safe = [s for s in summaries if not s["violations"]]
    if safe:
        widest = max(safe, key=lambda s: s["distinct"])
        lines.append(
            f"公式：保序候选里分辨率最高的是 `{widest['name']}`"
            f"（{widest['distinct']} 个不同取值、跨度 {widest['span']:.3f}）。"
        )
    else:
        lines.append("公式：**没有保序候选**，需重新设计 cap。")
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


def build_report(path: Path) -> dict[str, Any]:
    rows = load_rows(path)
    counts = reward_type_counts(rows)
    records = extract_wrong_purchases(rows)
    spreads = {axis: group_spread(records, axis) for axis in ("match", "product")}
    verdicts = {axis: gate2_verdict(spread) for axis, spread in spreads.items()}
    summaries = [formula_summary(records, name) for name in CANDIDATES]
    return {
        "source": str(path),
        "rows": len(rows),
        "counts": dict(counts),
        "records": records,
        "spreads": spreads,
        "verdicts": verdicts,
        "formulas": summaries,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("trajectories", type=Path, help="轨迹 jsonl（评测或探针产物）")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而非 markdown")
    parser.add_argument("--out", type=Path, default=None, help="把报告写到该路径")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.trajectories.exists():
        print(f"找不到轨迹文件：{args.trajectories}", file=sys.stderr)
        return 2
    report = build_report(args.trajectories)
    if args.json:
        payload = json.dumps(
            {key: value for key, value in report.items() if key != "records"},
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    else:
        payload = render_report(
            Counter(report["counts"]),
            report["records"],
            report["spreads"],
            report["verdicts"],
            report["formulas"],
            source=report["source"],
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
