#!/usr/bin/env python3
"""M10-R：给「老师到底能不能被逼出目标动作」抽一批探针题。

为什么需要它（2026-10-21）
--------------------------
M10 Step 2 的结论是「老师不翻页 / 不回头买 / 不排除超预算」，三条 <20% 就地关闭。
但那次采集是**每题 1 次、temperature 0.0**（`scripts/run_m10_step2.sh`），
量到的是老师的**默认行为**，不是**能力上限**。这两件事的差别决定了后面怎么走：

* 默认不做、但逼一下能做 ⇒ 提高温度 + 多次采样就能从真实采集里拿回来（首选）。
* 怎么逼都不做 ⇒ 只能脚本合成（我们规定动作、环境给真实观测）。

用户 2026-10-21 的要求正是先回答第一个问题：**开训前把能从数据里拿回来的能力拿回来**。

本脚本只做选题：从三批已建好的目标清单里各抽 N 题，合成一份探针清单。
纯 CPU、不调 API、不碰环境。

三个场景与「目标动作」的对应（判定在 `analyze_teacher_action_rate.py`）
--------------------------------------------------------------------
* ``paging``  → ``next_page``（翻页）
* ``requery`` → 两次及以上**不同**的 ``search_products`` 查询词（换词重搜）
* ``compare`` → 打开两个及以上**不同** ASIN（对比候选）

翻页场景只收名次 21–60
----------------------
gold 排 61 名以后的题，环境在第 3 次 ``next_page`` 时判 ``repeat_loop`` 终止
（`docs/lessons-learned.md` 硬约束表：翻页最多 2 次）。拿这些题去探「老师会不会翻页」，
会把「翻不到」记成「不翻页」——这正是 M10 早期误判的同一类错误。

用法::

    # 默认每场景 40 题，写到 data/collection/m10_teacher_probe_targets.jsonl
    .venv/bin/python scripts/build_m10_teacher_probe.py

    # 每场景 30 题、换种子
    .venv/bin/python scripts/build_m10_teacher_probe.py --per-scenario 30 --seed 7

    # 只看会选出什么，不写文件
    .venv/bin/python scripts/build_m10_teacher_probe.py --dry-run
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 场景 → 已建好的目标清单。三批都与评测集、现有 SFT 池零重叠。
SCENARIO_TARGETS = {
    "paging": ROOT / "data" / "collection" / "m10_paging_targets.jsonl",
    "requery": ROOT / "data" / "collection" / "m10_requery_targets.jsonl",
    "compare": ROOT / "data" / "collection" / "m10_compare_targets.jsonl",
}

#: 翻页场景的名次上限。见模块 docstring「翻页场景只收名次 21–60」。
PAGING_MAX_RANK = 60

#: 探针清单默认落盘位置。
DEFAULT_OUTPUT = ROOT / "data" / "collection" / "m10_teacher_probe_targets.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def eligible(scenario: str, rows: list[dict]) -> list[dict]:
    """按场景口径过滤。

    翻页场景要砍掉名次 >60 的题（翻不到）；其余两批的建单口径已经是「必须换词 /
    必须对比」，原样保留。
    """
    if scenario == "paging":
        kept = []
        for row in rows:
            rank = row.get("rank")
            if rank is None:
                continue
            if int(rank) <= PAGING_MAX_RANK:
                kept.append(row)
        return kept
    return list(rows)


def sample(rows: list[dict], count: int, *, seed: int) -> list[dict]:
    """确定性抽样：先按 task_id 排序再抽，避免文件顺序变动影响结果。"""
    ordered = sorted(rows, key=lambda row: int(row["task_id"]))
    if count >= len(ordered):
        return ordered
    rng = random.Random(seed)
    picked = rng.sample(ordered, count)
    return sorted(picked, key=lambda row: int(row["task_id"]))


def build(
    *,
    per_scenario: int,
    seed: int,
    targets: dict[str, Path] | None = None,
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    """返回 (探针清单, 每场景的供给/选中统计)。

    同一 task_id 只保留第一次出现的场景——三批建单时已互斥，这里再兜一层，
    因为采集端按 ``(task_id, attempt_index)`` 去重，重复会让计数虚高。
    """
    targets = targets or SCENARIO_TARGETS
    chosen: list[dict] = []
    seen: set[int] = set()
    report: dict[str, dict[str, int]] = {}

    for scenario, path in targets.items():
        rows = load_jsonl(path) if Path(path).exists() else []
        pool = eligible(scenario, rows)
        picked = sample(pool, per_scenario, seed=seed)
        kept = []
        for row in picked:
            task_id = int(row["task_id"])
            if task_id in seen:
                continue
            seen.add(task_id)
            kept.append({**row, "scenario": scenario})
        chosen.extend(kept)
        report[scenario] = {
            "available": len(rows),
            "eligible": len(pool),
            "selected": len(kept),
        }

    chosen.sort(key=lambda row: (row["scenario"], int(row["task_id"])))
    return chosen, report


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_report(report: dict[str, dict[str, int]]) -> str:
    lines = ["| 场景 | 清单题数 | 口径内可抽 | 本次选中 |", "|---|---|---|---|"]
    for scenario in ("paging", "requery", "compare"):
        if scenario not in report:
            continue
        row = report[scenario]
        lines.append(
            f"| {scenario} | {row['available']} | {row['eligible']} | {row['selected']} |"
        )
    total = sum(row["selected"] for row in report.values())
    lines.append(f"| **合计** | | | **{total}** |")
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-scenario", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="只打印统计，不写文件")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.per_scenario < 1:
        print("--per-scenario 必须 >= 1", file=sys.stderr)
        return 2

    chosen, report = build(per_scenario=args.per_scenario, seed=args.seed)
    print(render_report(report))

    counts = collections.Counter(row["scenario"] for row in chosen)
    print()
    print("选中分布：" + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    if not chosen:
        print("没有选出任何题：先确认三批目标清单存在。", file=sys.stderr)
        return 1
    if args.dry_run:
        print("（--dry-run：不写文件）")
        return 0

    write_jsonl(args.output, chosen)
    print(f"已写 {len(chosen)} 行 → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
