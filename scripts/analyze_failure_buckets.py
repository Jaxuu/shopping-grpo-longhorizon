#!/usr/bin/env python3
"""把失败的题按「错在哪一环」拆开，回答「下一轮该教什么」。

为什么要有这个
--------------
M10-H 之后 Final-200 停在 63.5%，失败 73 道。翻页分析只回答了「该翻的题翻没翻」，
但失败里最大的一块（47 道）是**第一页就能看到答案却买错**，翻页分析管不到它。

要造数据就必须先知道错在哪一环，因为三种情况的处方完全相反：

* **买了对的商品但没过硬门**（多为超预算）⇒ 教「选完规格后核对实际价格」；
* **打开过对的商品却买了别的** ⇒ 教「在几个候选之间比对约束」；
* **从没打开过对的商品** ⇒ 教「怎么把对的商品找出来 / 什么时候翻页」。

猜错一环，造出来的数据就是在教模型它已经会的事——M10-R v3 就这么白跑过一轮。

和 `analyze_selection_vs_retrieval.py` 的区别
---------------------------------------------
那个脚本是给**同题多采样**的探针轨迹用的，正确商品得从组内成功样本反推。
评测轨迹每题只有 1 条，但它自带 `terminal_result.goal.asin`（题面给的目标商品），
所以这里**直接读真值**，不需要反推，也不需要多采样。

不需要显卡、不需要起环境服务，纯读 jsonl。

用法::

    .venv/bin/python scripts/analyze_failure_buckets.py \
        outputs/evaluation/grpo-m10h/trajectories.jsonl --detail

    # 多个文件一起比（例如 v2 vs GRPO）
    .venv/bin/python scripts/analyze_failure_buckets.py \
        outputs/evaluation/m10r-v2/trajectories.jsonl \
        outputs/evaluation/grpo-m10h/trajectories.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SUCCESS_TYPES = {"gold_purchase", "valid_alternative_purchase"}
PURCHASE_TOOL = "buy_now"
OPEN_TOOL = "open_product"
SEARCH_TOOL = "search_products"
OPTION_TOOL = "select_option"
PAGE_TOOL = "next_page"
GATE_FAIL = "fail"

# 四个互斥的失败桶，顺序即报告里的顺序（从「最接近成功」到「最远」）。
BUCKETS = (
    ("bought_gold_gate_failed", "买对了商品但没过硬门（多为超预算）"),
    ("saw_gold_bought_other", "打开过正确商品，却买了别的"),
    ("saw_gold_no_purchase", "打开过正确商品，但没下单"),
    ("never_saw_gold", "从没打开过正确商品"),
    ("unknown", "无法判定（缺 gold 或购买记录）"),
)


# ── 读数据 ──────────────────────────────────────────────────────────────


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _detail(row: Mapping[str, Any]) -> dict:
    detail = (row.get("terminal_result") or {}).get("reward_detail")
    return detail if isinstance(detail, dict) else {}


def _steps(row: Mapping[str, Any]) -> list[dict]:
    steps = row.get("steps")
    return [s for s in steps if isinstance(s, dict)] if isinstance(steps, list) else []


def _params(step: Mapping[str, Any]) -> dict:
    params = step.get("parameters")
    return params if isinstance(params, dict) else {}


def _tool(step: Mapping[str, Any]) -> str:
    return str(step.get("tool_name") or "")


def _asin(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _gate_status(gate: Any) -> str:
    if isinstance(gate, Mapping):
        return str(gate.get("status") or "unknown")
    return "unknown"


def extract(row: Mapping[str, Any]) -> dict:
    """抽出一条评测轨迹的诊断字段。"""
    terminal = row.get("terminal_result") or {}
    detail = _detail(row)
    gates = detail.get("hard_gates")
    gates = gates if isinstance(gates, Mapping) else {}

    gold = _asin((terminal.get("goal") or {}).get("asin"))
    purchased = _asin((terminal.get("purchase") or {}).get("asin"))

    opened: list[str] = []
    searches = 0
    selects = 0
    pages = 0
    for step in _steps(row):
        name = _tool(step)
        if name == OPEN_TOOL:
            asin = _asin(_params(step).get("asin"))
            if asin:
                opened.append(asin)
        elif name == SEARCH_TOOL:
            searches += 1
        elif name == OPTION_TOOL:
            selects += 1
        elif name == PAGE_TOOL:
            pages += 1
        elif name == PURCHASE_TOOL and purchased is None and opened:
            # 有些版本不落 terminal.purchase，用「下单前最后打开的那个」兜底。
            purchased = opened[-1]

    scores = detail.get("dimension_scores")
    scores = {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))} if isinstance(scores, dict) else {}

    reward_type = str(detail.get("reward_type") or "unknown")
    return {
        "task_id": str(row.get("task_id")),
        "reward_type": reward_type,
        "success": reward_type in SUCCESS_TYPES,
        "gold": gold,
        "purchased": purchased,
        "opened": opened,
        "opened_unique": len(set(opened)),
        "searches": searches,
        "selects": selects,
        "pages": pages,
        "budget_status": _gate_status(gates.get("budget")),
        "category_status": _gate_status(gates.get("category")),
        "target_asin_match": bool(detail.get("target_asin_match")),
        "weighted_score": detail.get("weighted_score"),
        "dimension_scores": scores,
        "steps": len(_steps(row)),
    }


# ── 分桶 ────────────────────────────────────────────────────────────────


def bucket(record: Mapping[str, Any]) -> str:
    """给一条轨迹定桶。成功的返回 "success"。"""
    if record["success"]:
        return "success"
    gold = record["gold"]
    purchased = record["purchased"]
    opened = set(record["opened"])
    if not gold:
        return "unknown"
    if purchased == gold:
        return "bought_gold_gate_failed"
    if gold in opened:
        return "saw_gold_bought_other" if purchased else "saw_gold_no_purchase"
    return "never_saw_gold"


def summarize(records: Sequence[Mapping[str, Any]]) -> dict:
    total = len(records)
    successes = sum(1 for r in records if r["success"])
    buckets = Counter(bucket(r) for r in records)
    failures = [r for r in records if not r["success"]]

    # 失败轨迹的硬门情况：budget / category 各有多少明确 fail。
    gates = Counter()
    for record in failures:
        gates[f"budget:{record['budget_status']}"] += 1
        gates[f"category:{record['category_status']}"] += 1

    # 维度诊断：只看有打分的失败轨迹，统计「这个维度没拿满分」的比例。
    dims: dict[str, list[float]] = defaultdict(list)
    for record in failures:
        for name, score in record["dimension_scores"].items():
            dims[name].append(score)
    dim_rows = [
        {
            "name": name,
            "n": len(values),
            "mean": statistics.fmean(values),
            "below_full": sum(1 for v in values if v < 1.0) / len(values),
        }
        for name, values in dims.items()
    ]
    dim_rows.sort(key=lambda row: (-row["below_full"], row["name"]))

    def med(key: str, group: Iterable[Mapping[str, Any]]) -> float:
        values = [float(r[key]) for r in group]
        return statistics.median(values) if values else 0.0

    return {
        "total": total,
        "successes": successes,
        "rate": successes / total if total else 0.0,
        "buckets": {name: buckets.get(name, 0) for name, _ in BUCKETS},
        "unknown_success": buckets.get("success", 0),
        "gates": dict(gates),
        "dims": dim_rows,
        "fail_opened_median": med("opened_unique", failures),
        "fail_searches_median": med("searches", failures),
        "fail_pages_median": med("pages", failures),
        "ok_opened_median": med("opened_unique", [r for r in records if r["success"]]),
        "ok_searches_median": med("searches", [r for r in records if r["success"]]),
        "ok_pages_median": med("pages", [r for r in records if r["success"]]),
    }


def verdict(summary: Mapping[str, Any]) -> list[str]:
    """按桶的大小给出「下一轮该教什么」的判读，不猜。"""
    buckets = summary["buckets"]
    failures = summary["total"] - summary["successes"]
    if failures == 0:
        return ["- 没有失败样本，无需判读。"]
    ranked = sorted(
        ((name, count) for name, count in buckets.items() if name != "unknown"),
        key=lambda pair: -pair[1],
    )
    top_name, top_count = ranked[0]
    label = dict(BUCKETS)[top_name]
    share = top_count / failures
    lines = [
        f"- 失败 {failures} 道，最大的一块是 **{label}**：{top_count} 道（{share:.1%}）。"
    ]
    lines.append(
        {
            "bought_gold_gate_failed": "- ⇒ 商品找对了，栽在**硬门**（多半是超预算）。数据该教"
            "「选完规格后读实际价格再下单」，不是教检索。",
            "saw_gold_bought_other": "- ⇒ 找对了也看到了，栽在**取舍**。数据该教「逐条比对候选的约束」"
            "（容量/尺寸/价格），不是教搜更准。",
            "saw_gold_no_purchase": "- ⇒ 看到了却没下单，栽在**决策收尾**。数据该教「确认符合就买」，"
            "不必改检索。",
            "never_saw_gold": "- ⇒ 根本没看到对的货，栽在**覆盖**。数据该教「换更精确的词 / 往下翻页」。",
        }[top_name]
    )
    budget_fail = summary["gates"].get(f"budget:{GATE_FAIL}", 0)
    if budget_fail:
        lines.append(
            f"- 硬门明确判 fail 的里，budget 占 {budget_fail} 道 ⇒ 超预算是可量化的一类，"
            "**可以单独造一批「核价」示范**。"
        )
    if summary["dims"]:
        worst = summary["dims"][0]
        lines.append(
            f"- 失败轨迹里最常没拿满分的维度是 **{worst['name']}**"
            f"（{worst['below_full']:.0%} 的失败轨迹在该维度失分，均值 {worst['mean']:.2f}）。"
        )
    return lines


# ── 输出 ────────────────────────────────────────────────────────────────


def format_report(name: str, summary: Mapping[str, Any]) -> str:
    lines = [
        f"== {name}（{summary['total']} 条轨迹）",
        "",
        f"成功 {summary['successes']}/{summary['total']} = **{summary['rate']:.1%}**，"
        f"失败 {summary['total'] - summary['successes']} 道",
        "",
        "| 失败桶 | 道数 | 占失败 | 含义 |",
        "|---|---:|---:|---|",
    ]
    failures = summary["total"] - summary["successes"]
    for key, label in BUCKETS:
        count = summary["buckets"].get(key, 0)
        share = f"{count / failures:.1%}" if failures else "—"
        lines.append(f"| `{key}` | {count} | {share} | {label} |")
    lines += [
        "",
        "| 行为 | 失败轨迹中位数 | 成功轨迹中位数 |",
        "|---|---:|---:|",
        f"| 打开过的不同商品数 | {summary['fail_opened_median']:.0f} | {summary['ok_opened_median']:.0f} |",
        f"| 搜索次数 | {summary['fail_searches_median']:.0f} | {summary['ok_searches_median']:.0f} |",
        f"| 翻页次数 | {summary['fail_pages_median']:.0f} | {summary['ok_pages_median']:.0f} |",
    ]
    if summary["gates"]:
        lines += ["", "| 硬门状态 | 道数（失败轨迹内） |", "|---|---:|"]
        for key in sorted(summary["gates"]):
            lines.append(f"| {key} | {summary['gates'][key]} |")
    if summary["dims"]:
        lines += [
            "",
            "| 偏好维度 | 样本 | 均值 | 没拿满分的比例 |",
            "|---|---:|---:|---:|",
        ]
        for row in summary["dims"]:
            lines.append(
                f"| {row['name']} | {row['n']} | {row['mean']:.2f} | {row['below_full']:.0%} |"
            )
    lines += ["", "**判读**", ""] + verdict(summary)
    return "\n".join(lines)


def format_detail(records: Sequence[Mapping[str, Any]], limit: int) -> str:
    rows = [r for r in records if not r["success"]]
    rows.sort(key=lambda r: (bucket(r), r["task_id"]))
    lines = [
        "| task_id | 桶 | reward_type | 买的是 gold？ | 打开数 | 搜索 | 翻页 | budget |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for record in rows[:limit]:
        lines.append(
            f"| {record['task_id']} | `{bucket(record)}` | {record['reward_type']} | "
            f"{'是' if record['purchased'] == record['gold'] else '否'} | "
            f"{record['opened_unique']} | {record['searches']} | {record['pages']} | "
            f"{record['budget_status']} |"
        )
    if len(rows) > limit:
        lines.append(f"| … | 省略 {len(rows) - limit} 道 | | | | | | |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把失败题按「错在哪一环」拆开")
    parser.add_argument("trajectories", type=Path, nargs="+", help="评测轨迹 jsonl（可多个）")
    parser.add_argument("--detail", action="store_true", help="附逐题明细")
    parser.add_argument("--limit", type=int, default=40, help="明细最多列几道")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    payload = {}
    for path in args.trajectories:
        records = [extract(row) for row in load_rows(path)]
        payload[path.name] = (summarize(records), records)

    if args.json:
        print(json.dumps({k: v[0] for k, v in payload.items()}, ensure_ascii=False, indent=2))
        return 0

    for name, (summary, records) in payload.items():
        print(format_report(name, summary))
        if args.detail:
            print()
            print(format_detail(records, args.limit))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
