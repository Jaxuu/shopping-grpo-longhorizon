#!/usr/bin/env python3
"""A 段的 43.5% 到底亏在哪：翻页本身有害，还是翻页顺手弄丢了别的动作？

背景
----
M10-R 的 A 段（只训探索桶）单独评测：翻页率 31.0%，但 strict 只有 43.5%
（终版 63.0%）。两种解释完全不同，修法相反：

* **翻页有害**：翻了页就买错 ⇒ 探索数据的示范有问题，得回去改采集
* **翻页无辜**：分数是被别的东西拖的（比如丢了「选规格」这条链）⇒ 改训练配方

本脚本只做一件事：按「这条轨迹翻没翻页」分组，比成功率、比动作覆盖。
如果翻页组的成功率和没翻页组差不多，那翻页就不是病灶——真正丢的是别的动作。

为什么还要看动作覆盖
--------------------
A 段产物的平均工具步只有 4.79（池子 10.10），partial_alternative_purchase
从 48 条涨到 63 条。`partial` 的典型成因是**没选对规格**，而选规格靠
`select_option`。如果 A 段产物大面积不调用 `select_option`，那 43.5% 就是
「翻页 + 丢规格链」两个因素叠加，不能把账全算在翻页头上。

用法::

    .venv/bin/python scripts/analyze_paging_payoff.py \\
        outputs/evaluation/m10r-stageA/trajectories.jsonl \\
        outputs/evaluation/m10r-final200/trajectories.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:  # 以模块方式导入（pytest / python -m）
    from scripts.report_behavior_rate import action_names, load_jsonl
except ModuleNotFoundError:  # 直接 `python scripts/xxx.py` 时 sys.path[0] 是 scripts/
    from report_behavior_rate import action_names, load_jsonl  # type: ignore[no-redef]

GOLD = "gold_purchase"
PARTIAL = "partial_alternative_purchase"
WRONG = "wrong_purchase"

#: 只在「成功路径」上才有意义的动作。缺了它们分数就会掉，但翻页率看不出来。
CHAIN_ACTIONS = ("select_option", "open_product", "view_features", "view_description", "buy_now")


def reward_type(row: dict) -> str:
    terminal = row.get("terminal_result") or {}
    detail = terminal.get("reward_detail") or {}
    return str(detail.get("reward_type") or "unknown")


def summarize(rows: list[dict]) -> dict:
    groups = {"翻页": [], "没翻页": []}
    for row in rows:
        names = action_names(row)
        key = "翻页" if "next_page" in names else "没翻页"
        groups[key].append((row, names))

    result = {}
    for key, items in groups.items():
        if not items:
            continue
        total = len(items)
        types = [reward_type(row) for row, _ in items]
        result[key] = {
            "tasks": total,
            "gold": types.count(GOLD),
            "partial": types.count(PARTIAL),
            "wrong": types.count(WRONG),
            "mean_steps": sum(len(names) for _, names in items) / total,
            "chain": {
                action: sum(1 for _, names in items if action in names) for action in CHAIN_ACTIONS
            },
        }
    return result


def format_report(name: str, stats: dict) -> str:
    lines = [f"== {name}"]
    lines.append("| 分组 | 题数 | 买对(gold) | 部分满足 | 买错 | 平均工具步 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key in ("翻页", "没翻页"):
        if key not in stats:
            continue
        item = stats[key]
        total = item["tasks"]
        lines.append(
            f"| {key} | {total} | {item['gold']}（{item['gold'] / total * 100:.1f}%） | "
            f"{item['partial']}（{item['partial'] / total * 100:.1f}%） | "
            f"{item['wrong']}（{item['wrong'] / total * 100:.1f}%） | {item['mean_steps']:.2f} |"
        )
    lines.append("")
    lines.append("| 动作覆盖（至少调用一次的题数） | " + " | ".join(k for k in ("翻页", "没翻页") if k in stats) + " |")
    lines.append("|---|" + "---:|" * sum(1 for k in ("翻页", "没翻页") if k in stats))
    for action in CHAIN_ACTIONS:
        cells = []
        for key in ("翻页", "没翻页"):
            if key not in stats:
                continue
            item = stats[key]
            count = item["chain"][action]
            cells.append(f"{count}/{item['tasks']}（{count / item['tasks'] * 100:.0f}%）")
        lines.append(f"| {action} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+", help="评测 trajectories.jsonl")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    for path in args.paths:
        if not path.exists():
            print(f"找不到 {path}", file=sys.stderr)
            return 1
        rows = load_jsonl(path)
        stats = summarize(rows)
        if args.as_json:
            print(json.dumps({str(path): stats}, ensure_ascii=False, indent=2))
        else:
            print(format_report(str(path), stats))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
