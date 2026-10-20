#!/usr/bin/env python3
"""体检 `generate_m10_paging.py` 产出的轨迹：四条硬检查，逐条给结论。

为什么单独做一个脚本
--------------------
冒烟跑完 `summary.json` 只会说「3 条全过」。但脚本自己判「过」不等于轨迹
真的对 —— 它可能翻页了却没翻到位、可能名次标签和线上检索不一致、可能
消息结构拼错了。这些必须看原文，而看原文的临时片段一旦靠手工粘贴就会
丢缩进。所以固化成脚本，一条命令跑完。

四条检查
--------
1. **第 1 页不能有 gold**：有的话说明「翻页」是假的，会教出无意义的动作。
2. **gold 出现在第几页**：对照清单里的预测名次，判断线上检索与离线索引是否同源。
3. **终局必须是 gold_purchase，且买的 ASIN 等于 goal ASIN**。
4. **messages 必须是 system → user → (assistant → tool) 严格交替**。

用法::

    .venv/bin/python scripts/inspect_m10_paging_output.py outputs/m10-paging-smoke/raw.jsonl
    .venv/bin/python scripts/inspect_m10_paging_output.py outputs/m10-paging/raw.jsonl --summary-only
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

#: 与 `environment.actions.product_ids` 同源，避免「以为看见」与「能点」分歧。
SEARCH_ROW = re.compile(r"^(?P<rank>\d+)\|(?P<asin>\d{8,12})\|", re.MULTILINE)
SEARCH_PAGE = "page_type: search_results"
FIRST_PAGE_END = 20


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def search_pages(steps: list[dict], gold: str) -> list[dict]:
    """按出现顺序返回每个搜索页：渲染行数、gold 在本页的名次（没有则 None）。"""
    pages: list[dict] = []
    for step in steps:
        observation = step.get("observation") or ""
        if SEARCH_PAGE not in observation:
            continue
        rows = [
            (int(match.group("rank")), match.group("asin"))
            for match in SEARCH_ROW.finditer(observation)
        ]
        ranks = [rank for rank, asin in rows if asin == gold]
        pages.append(
            {
                "tool": step.get("tool_name"),
                "rows": len(rows),
                "gold_rank": ranks[0] if ranks else None,
            }
        )
    return pages


def role_sequence(messages: list[dict]) -> list[str]:
    return [str(message.get("role")) for message in messages]


def check_roles(messages: list[dict]) -> str | None:
    """返回问题描述；None 表示结构正确。"""
    roles = role_sequence(messages)
    if roles[:2] != ["system", "user"]:
        return f"开头应为 system→user，实际 {roles[:2]}"
    body = roles[2:]
    if len(body) % 2:
        return f"system/user 之后的轮次不成对：{len(body)} 条"
    for index in range(0, len(body), 2):
        if body[index] != "assistant" or body[index + 1] != "tool":
            return f"第 {index // 2 + 1} 轮应为 assistant→tool，实际 {body[index:index + 2]}"
    return None


def rank_band(rank: int) -> str:
    """把名次折成可读区间：1–10、11–20、21–30 …（直接 //10 会得到 0 号段）。"""
    start = (int(rank) - 1) // 10 * 10 + 1
    return f"{start}-{start + 9}"


def inspect_row(row: dict) -> dict:
    steps = row.get("steps") or []
    terminal = row.get("terminal_result") or {}
    detail = terminal.get("reward_detail") or {}
    goal = terminal.get("goal") or {}
    goal_asin = str(goal.get("asin") or "")
    purchased = str((terminal.get("purchase") or {}).get("asin") or "")
    pages = search_pages(steps, goal_asin)

    problems: list[str] = []
    first_page_gold = pages[0]["gold_rank"] if pages else None
    if not pages:
        problems.append("没有任何搜索页观测")
    elif first_page_gold is not None:
        problems.append(f"第 1 页就出现 gold（名次 {first_page_gold}），翻页示范无效")
    if detail.get("reward_type") != "gold_purchase":
        problems.append(f"reward_type={detail.get('reward_type')!r}")
    if purchased != goal_asin:
        problems.append(f"买的 {purchased!r} ≠ goal {goal_asin!r}")
    role_problem = check_roles(row.get("messages") or [])
    if role_problem:
        problems.append(role_problem)

    # gold 第一次出现是在第几个搜索页（1 基），用于和清单预测名次对照
    found_on_page = next(
        (index for index, page in enumerate(pages, start=1) if page["gold_rank"] is not None),
        None,
    )
    return {
        "task_id": row.get("task_id"),
        "steps": len(steps),
        "tools": [step.get("tool_name") for step in steps],
        "reward_type": detail.get("reward_type"),
        "reward": detail.get("reward"),
        "purchase_success": detail.get("purchase_success"),
        "goal_asin": goal_asin,
        "purchased_asin": purchased,
        "pages": pages,
        "found_on_page": found_on_page,
        "gold_rank": next(
            (page["gold_rank"] for page in pages if page["gold_rank"] is not None), None
        ),
        "problems": problems,
    }


def print_row(report: dict) -> None:
    print(f"task {report['task_id']} | 步数 {report['steps']} | 动作 {report['tools']}")
    print(
        f"   reward_type {report['reward_type']} | reward {report['reward']}"
        f" | purchase_success {report['purchase_success']}"
    )
    print(
        f"   goal asin {report['goal_asin']} | 实际购买 {report['purchased_asin']}"
    )
    for index, page in enumerate(report["pages"], start=1):
        rank = page["gold_rank"]
        label = f"名次 {rank}" if rank is not None else "不在本页"
        print(f"   搜索页#{index}（{page['tool']}）: 渲染 {page['rows']} 行, gold {label}")
    if report["problems"]:
        for problem in report["problems"]:
            print(f"   [FAIL] {problem}")
    else:
        print("   [ok] 四条检查全过")
    print()


def summarize(reports: list[dict]) -> dict:
    failed = [report for report in reports if report["problems"]]
    page_histogram = collections.Counter(report["found_on_page"] for report in reports)
    rank_histogram = collections.Counter(
        rank_band(report["gold_rank"]) for report in reports if report["gold_rank"]
    )
    return {
        "total": len(reports),
        "failed": len(failed),
        "failed_ids": [report["task_id"] for report in failed],
        "found_on_page": dict(sorted(page_histogram.items(), key=lambda kv: (kv[0] is None, kv[0]))),
        "gold_rank_band": dict(sorted(rank_histogram.items())),
        "steps_mean": round(
            sum(report["steps"] for report in reports) / len(reports), 2
        ) if reports else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("raw", type=Path, help="generate_m10_paging.py 产出的 raw.jsonl")
    parser.add_argument("--summary-only", action="store_true", help="只看汇总，不逐条打印")
    args = parser.parse_args(argv)

    if not args.raw.is_file():
        raise SystemExit(f"没有 {args.raw}")
    rows = read_jsonl(args.raw)
    reports = [inspect_row(row) for row in rows]

    if not args.summary_only:
        for report in reports:
            print_row(report)

    summary = summarize(reports)
    print("=" * 60)
    print(f"共 {summary['total']} 条，四条检查全过 {summary['total'] - summary['failed']} 条，"
          f"有问题 {summary['failed']} 条")
    if summary["failed"]:
        print(f"  有问题的 task_id：{summary['failed_ids'][:20]}"
              + (" …" if summary["failed"] > 20 else ""))
    print(f"gold 首次出现的搜索页次分布：{summary['found_on_page']}（1 = 第 1 页就有）")
    print(f"gold 线上名次分布（按 10 名一段）：{summary['gold_rank_band']}")
    print(f"平均步数 {summary['steps_mean']}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
