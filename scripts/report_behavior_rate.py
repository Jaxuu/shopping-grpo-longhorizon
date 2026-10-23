#!/usr/bin/env python3
"""M10-R 验收：新模型在 Final-200 上到底翻页 / 换词 / 多开了没有。

为什么单独立一个脚本
--------------------
M10-R 的目标不是「分数涨」，而是**把「只开第 1 名」这个动作改掉**。分数只是必要条件：
如果探索率没抬起来，即使分数不掉，也说明这次 SFT 没起作用（只是没伤到原有能力）；
如果探索率抬起来了但分数掉了，说明抬得过头或示范质量差。两种失败要能区分开。

所以判据是三条率，不是一条分数：

* **翻页率** —— 出现过 ``next_page``（``prev_page`` 是「退回列表」的常规导航，不算）
* **换词率** —— 出现过 **≥2 个不同**的 ``search_products`` 查询词（归一化后比较）
* **多开率** —— 打开过 **≥2 个不同** ASIN

外加「模板率」= 单搜单开，用来对照 SFT 池里 57.9% 的占比。

口径
----
与 ``scripts/prepare_sft_curriculum.behavior_profile`` 完全一致（探索 = 翻页 / 换词 /
开≥2；模板 = 搜索 ≤1 次且开商品 ≤1 次），这样池子和评测集的数可以直接对比。

两种输入格式都吃：

* 评测 / 采集轨迹（``steps[].tool_name`` + ``steps[].parameters``）
* SFT 池（``messages[].tool_calls[].function``）

用法::

    .venv/bin/python scripts/report_behavior_rate.py \\
        outputs/evaluation/m10r-final200/trajectories.jsonl \\
        data/sft_m10r/capped/all.jsonl \\
        data/sft_pure_v4/all.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PAGE_TOOL = "next_page"
SEARCH_TOOL = "search_products"
OPEN_TOOL = "open_product"
NON_ACTION_TOOLS = frozenset({"think"})


def normalize_query(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def _parse_arguments(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def iter_actions(row: dict):
    """产出 ``(tool_name, arguments)``，自动适配两种行格式。"""
    steps = row.get("steps")
    if isinstance(steps, list) and steps:
        for step in steps:
            name = str(step.get("tool_name") or "")
            if name:
                yield name, _parse_arguments(step.get("parameters") or {})
        return
    for message in row.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            if name:
                yield name, _parse_arguments(function.get("arguments"))


def profile_row(row: dict) -> dict:
    names: list[str] = []
    queries: list[str] = []
    opened: list[str] = []
    for name, arguments in iter_actions(row):
        if name in NON_ACTION_TOOLS:
            continue
        names.append(name)
        if name == SEARCH_TOOL and arguments.get("query"):
            queries.append(normalize_query(arguments["query"]))
        elif name == OPEN_TOOL and arguments.get("asin"):
            opened.append(str(arguments["asin"]).strip())

    distinct_queries = {query for query in queries if query}
    distinct_opened = {asin for asin in opened if asin}
    paging = PAGE_TOOL in names
    searched = names.count(SEARCH_TOOL)
    opened_calls = names.count(OPEN_TOOL)
    explored = paging or len(distinct_queries) >= 2 or len(distinct_opened) >= 2
    if explored:
        bucket = "exploratory"
    elif searched <= 1 and opened_calls <= 1:
        bucket = "template"
    else:
        bucket = "other"
    return {
        "tool_calls": len(names),
        "paging": paging,
        "requery": len(distinct_queries) >= 2,
        "multi_open": len(distinct_opened) >= 2,
        "distinct_queries": len(distinct_queries),
        "distinct_opened": len(distinct_opened),
        "bucket": bucket,
    }


def summarize(rows: list[dict]) -> dict:
    profiles = [profile_row(row) for row in rows]
    total = len(profiles)

    def count(key: str) -> int:
        return sum(1 for profile in profiles if profile[key])

    buckets = {"exploratory": 0, "template": 0, "other": 0}
    for profile in profiles:
        buckets[profile["bucket"]] += 1
    steps = [profile["tool_calls"] for profile in profiles]
    return {
        "trajectories": total,
        "paging": count("paging"),
        "requery": count("requery"),
        "multi_open": count("multi_open"),
        "exploratory": buckets["exploratory"],
        "template": buckets["template"],
        "other": buckets["other"],
        "mean_tool_calls": (sum(steps) / total) if total else 0.0,
        "max_tool_calls": max(steps) if steps else 0,
    }


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    skipped = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    if skipped:
        print(f"[warn] {path} 有 {skipped} 行不是合法 JSON，已跳过", file=sys.stderr)
    return rows


def format_report(name: str, stats: dict) -> str:
    total = stats["trajectories"]
    lines = [f"== {name}（{total} 条轨迹）"]
    lines.append("| 指标 | 条数 | 占比 |")
    lines.append("|---|---:|---:|")
    for label, key in (
        ("翻页（next_page ≥1）", "paging"),
        ("换词（≥2 个不同查询词）", "requery"),
        ("多开（≥2 个不同 ASIN）", "multi_open"),
        ("探索（三者任一）", "exploratory"),
        ("模板（单搜单开）", "template"),
        ("其他", "other"),
    ):
        value = stats[key]
        lines.append(f"| {label} | {value} | {rate(value, total) * 100:.1f}% |")
    lines.append(
        f"| 平均工具步 | {stats['mean_tool_calls']:.2f} | 最多 {stats['max_tool_calls']} 步 |"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+", help="轨迹 jsonl（评测或 SFT 池）")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--expect",
        type=int,
        default=0,
        help="若给出，检查第一个文件的轨迹条数是否等于它（评测集应为 200）",
    )
    args = parser.parse_args(argv)

    results: dict[str, dict] = {}
    for path in args.paths:
        if not path.exists():
            print(f"找不到 {path}", file=sys.stderr)
            return 1
        rows = load_jsonl(path)
        if not rows:
            print(f"{path} 是空的", file=sys.stderr)
            return 1
        results[str(path)] = summarize(rows)

    if args.as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    first = results[str(args.paths[0])]
    for index, (name, stats) in enumerate(results.items()):
        if index:
            print()
        print(format_report(name, stats))

    if args.expect:
        actual = first["trajectories"]
        if actual != args.expect:
            print(
                f"\n[FAIL] 轨迹数 {actual} ≠ 期望 {args.expect}，这份评测不完整，别拿去比分数。",
                file=sys.stderr,
            )
            return 1
        print(f"\n[ok] 轨迹数 {actual} = 期望 {args.expect}")

    if len(results) > 1:
        print("\n== 对比（第一个文件 vs 其余）")
        keys = ("paging", "requery", "multi_open", "exploratory", "template")
        header = "| 指标 | " + " | ".join(Path(name).parent.name or name for name in results) + " |"
        print(header)
        print("|---|" + "---:|" * len(results))
        for key in keys:
            cells = []
            for stats in results.values():
                cells.append(f"{rate(stats[key], stats['trajectories']) * 100:.1f}%")
            print(f"| {key} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
