#!/usr/bin/env python3
"""M9：62% 的天花板到底卡在哪一环——模型有没有在"比较候选"？

为什么要有这个
--------------
M6/M6-R 用掉了两轮训练（~5 h 显卡）却没有可测收益。在决定第三轮之前必须先回答一个
比"奖励函数对不对"更基础的问题：**模型实际执行的是什么策略**。

本脚本只做一件事：把每条轨迹的动作序列摊开，量出「它到底看了几个候选」。

关键发现（2026-10-19，Final-200 / M5-B）
----------------------------------------
200 条轨迹里 **198 条只调用 1 次 `open_product`**，且都紧跟在唯一一次 `search_products`
之后；每题固定 8 步（149 条）或 9 步（37 条）。也就是：

    search → open(第一个候选) → select → features → prev → description → prev → buy

所谓"核验"发生在选择已经确定之后，不参与决策。SFT 池 1,252 条里 1,117 条（89.2%）
同样是"只开 1 个就下单"，最常见序列与上面逐字相同 ⇒ 学生忠实复制了老师的单候选策略。

因此成功率几乎由**一次搜索的第一个结果**决定：第 1 名是正确商品时 86.8%（106 题），
不是时 34.0%（94 题）。**62.0% 是"单候选策略"的天花板，不是规格能力的上限。**

五个桶（互斥，按"走到哪一步"排序）
----------------------------------
1. `bought_gold_incomplete` —— 买的就是正确商品，但某个维度没满分（多为规格）
2. `opened_gold_bought_other` —— 打开过正确商品，最后买了别的（判断问题）
3. `opened_gold_no_purchase` —— 打开过正确商品，但没下单（循环/放弃）
4. `saw_gold_not_opened` —— 正确商品在搜索结果里，却没打开（**本脚本的主要靶点**）
5. `never_saw_gold` —— 正确商品从没出现在任何搜索结果里（检索问题）

只有第 4 桶能靠"多看几个候选"直接救回来，所以额外给一张「打开前 k 个能覆盖多少」的
累积表——那是探索策略的**上界**，不是保证。

用法::

    .venv/bin/python scripts/analyze_candidate_exploration.py \
        outputs/evaluation/sft-b-curriculum/trajectories.jsonl \
        --out outputs/evaluation/sft-b-curriculum/candidate_exploration.md

    # 机器可读
    .venv/bin/python scripts/analyze_candidate_exploration.py <file.jsonl> --json

不需要外部数据、不需要起环境服务、不需要显卡。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

SUCCESS_TYPES = frozenset({"gold_purchase", "valid_alternative_purchase"})
SEARCH_ROW = re.compile(r"^(?P<rank>\d+)\|(?P<asin>[^|\n]+)\|", re.MULTILINE)
PRODUCT_ASIN = re.compile(r"(?m)^asin:\s*(?P<asin>\S+)\s*$")

BUCKET_ORDER = (
    "bought_gold_incomplete",
    "opened_gold_bought_other",
    "opened_gold_no_purchase",
    "saw_gold_not_opened",
    "never_saw_gold",
    "no_gold_reference",
)
BUCKET_LABELS = {
    "bought_gold_incomplete": "买对了商品，但某个维度没满分",
    "opened_gold_bought_other": "打开过正确商品，却买了别的",
    "opened_gold_no_purchase": "打开过正确商品，但没下单",
    "saw_gold_not_opened": "正确商品在结果里，却没打开",
    "never_saw_gold": "正确商品从未出现在搜索结果里",
    "no_gold_reference": "轨迹里没有正确商品可对照（无法归因）",
}


def _observation(step: Mapping[str, Any]) -> str:
    """优先用未被投影截断的原始观测。"""
    raw = step.get("raw_observation")
    if isinstance(raw, str) and raw:
        return raw
    return str(step.get("observation") or "")


def search_result_asins(observation: str) -> list[str]:
    """按出现顺序返回搜索结果页里的 ASIN；非搜索结果页返回空表。"""
    if "page_type: search_results" not in observation:
        return []
    return [match.group("asin").strip() for match in SEARCH_ROW.finditer(observation)]


def product_page_asin(observation: str) -> str | None:
    """返回商品详情页的 ASIN；非商品页返回 ``None``。"""
    if "page_type: product_detail" not in observation:
        return None
    match = PRODUCT_ASIN.search(observation)
    return match.group("asin").strip() if match else None


def diagnose(row: Mapping[str, Any]) -> dict[str, Any]:
    """把一条轨迹摊成"看了几个候选、卡在哪一步"。

    缺少 ``goal.asin`` 时仍然统计打开/搜索次数，只是无法归因到具体环节，
    归入 ``no_gold_reference``。
    """
    terminal = row.get("terminal_result") or {}
    detail = terminal.get("reward_detail") or {}
    goal = terminal.get("goal") or {}
    gold = str(goal.get("asin") or "").strip()

    reward_type = str(detail.get("reward_type") or "")
    success = reward_type in SUCCESS_TYPES
    bought_gold = bool(detail.get("target_asin_match"))

    opens = 0
    searches = 0
    opened_asins: list[str] = []
    opened_gold = False
    gold_ranks: list[int] = []
    first_search_rank1: str | None = None
    current_results: list[str] = []

    for step in row.get("steps") or []:
        observation = _observation(step)
        if step.get("tool_name") == "search_products":
            searches += 1
        results = search_result_asins(observation)
        if results:
            current_results = results
            if first_search_rank1 is None:
                first_search_rank1 = results[0]
            if gold in results:
                gold_ranks.append(results.index(gold) + 1)
        page_asin = product_page_asin(observation)
        if page_asin and page_asin == gold:
            opened_gold = True
        if step.get("tool_name") == "open_product":
            opens += 1
            asin = str((step.get("parameters") or {}).get("asin") or "")
            opened_asins.append(asin)
            if gold and asin == gold:
                opened_gold = True

    purchased = bool(terminal.get("purchase"))

    if success:
        bucket = "success"
    elif bought_gold:
        bucket = "bought_gold_incomplete"
    elif not gold:
        bucket = "no_gold_reference"
    elif opened_gold:
        bucket = "opened_gold_bought_other" if purchased else "opened_gold_no_purchase"
    elif gold_ranks:
        bucket = "saw_gold_not_opened"
    else:
        bucket = "never_saw_gold"

    return {
        "task_id": row.get("task_id"),
        "reward_type": reward_type,
        "success": success,
        "bucket": bucket,
        "opens": opens,
        "searches": searches,
        "steps": len(row.get("steps") or []),
        "gold_best_rank": min(gold_ranks) if gold_ranks else None,
        "first_search_rank1_is_gold": first_search_rank1 == gold if first_search_rank1 else None,
        "opened_asins": opened_asins,
        "result_page_size": len(current_results),
    }


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def aggregate(diagnosed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """汇总成功率、桶分布、打开次数分布与"前 k 名覆盖"上界。"""
    total = len(diagnosed)
    successes = sum(1 for item in diagnosed if item["success"])
    failures = total - successes
    buckets = Counter(item["bucket"] for item in diagnosed if not item["success"])

    open_counts = Counter(int(item["opens"]) for item in diagnosed)
    search_counts = Counter(int(item["searches"]) for item in diagnosed)
    step_counts = Counter(int(item["steps"]) for item in diagnosed)

    rank1_hit = [item for item in diagnosed if item["first_search_rank1_is_gold"] is True]
    rank1_miss = [item for item in diagnosed if item["first_search_rank1_is_gold"] is False]

    # 只在"正确商品确实出现在结果里"的失败题上算覆盖：打开前 k 名能碰到它的比例。
    saw_ranks = sorted(
        int(item["gold_best_rank"])
        for item in diagnosed
        if not item["success"] and item["gold_best_rank"]
    )
    coverage = {
        k: sum(1 for rank in saw_ranks if rank <= k) for k in (1, 3, 5, 10, 20)
    }
    reachable = buckets.get("saw_gold_not_opened", 0)

    return {
        "total": total,
        "successes": successes,
        "success_rate": _rate(successes, total),
        "failures": failures,
        "buckets": dict(buckets),
        "single_open_tasks": open_counts.get(1, 0),
        "single_open_share": _rate(open_counts.get(1, 0), total),
        "open_counts": dict(sorted(open_counts.items())),
        "search_counts": dict(sorted(search_counts.items())),
        "step_counts": dict(sorted(step_counts.items())),
        "rank1_hit": {
            "tasks": len(rank1_hit),
            "successes": sum(1 for item in rank1_hit if item["success"]),
        },
        "rank1_miss": {
            "tasks": len(rank1_miss),
            "successes": sum(1 for item in rank1_miss if item["success"]),
        },
        "reachable_by_exploration": reachable,
        "reachable_rate_ceiling": _rate(successes + reachable, total),
        "coverage_by_topk": coverage,
        "coverage_base": len(saw_ranks),
    }


def render_report(
    summary: Mapping[str, Any],
    diagnosed: Sequence[Mapping[str, Any]],
    source: str,
    examples: int,
) -> str:
    lines = [
        "# 候选探索解剖（M9 诊断）",
        "",
        f"- 数据源：`{source}`",
        f"- 轨迹：**{summary['total']}** 条；严格成功 **{summary['successes']}** "
        f"（{summary['success_rate'] * 100:.1f}%）；失败 **{summary['failures']}** 条",
        (
            f"- 其中 **{summary['missing_gold']}** 条缺少 `goal.asin`，无法定位正确商品，"
            f"归入「无法归因」"
            if summary.get("missing_gold")
            else ""
        ),
        "",
        "## 1. 它到底看了几个候选",
        "",
        f"**每题 `open_product` 次数分布**：`{summary['open_counts']}`",
        "",
        f"**只开 1 个商品就下单：{summary['single_open_tasks']}/{summary['total']} "
        f"= {summary['single_open_share'] * 100:.1f}%**",
        "",
        f"每题 `search_products` 次数：`{summary['search_counts']}`",
        "",
        f"每题步数：`{summary['step_counts']}`",
        "",
        "## 2. 成功率几乎由首条搜索结果决定",
        "",
        "| 第一次搜索的第 1 名 | 题数 | 成功 | 成功率 |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (("rank1_hit", "就是正确商品"), ("rank1_miss", "不是正确商品")):
        block = summary[key]
        lines.append(
            f"| {label} | {block['tasks']} | {block['successes']} | "
            f"{_rate(block['successes'], block['tasks']) * 100:.1f}% |"
        )

    lines += [
        "",
        "## 3. 失败卡在哪一环",
        "",
        "| 环节 | 条数 | 占失败 |",
        "|---|---:|---:|",
    ]
    for bucket in BUCKET_ORDER:
        count = summary["buckets"].get(bucket, 0)
        lines.append(
            f"| {BUCKET_LABELS[bucket]} | {count} | {_rate(count, summary['failures']) * 100:.0f}% |"
        )

    lines += [
        "",
        "## 4. 探索能救回多少（上界）",
        "",
        "只在「正确商品确实出现在结果里、却没打开」的题上算：",
        "",
        "| 打开前 k 名 | 覆盖到正确商品的题数 |",
        "|---|---:|",
    ]
    for k, count in summary["coverage_by_topk"].items():
        lines.append(f"| 前 {k} 名 | {count} |")
    lines += [
        "",
        f"覆盖基数 {summary['coverage_base']} 条；其中 **{summary['reachable_by_exploration']}** 条"
        f"属于「出现过但没打开」。",
        "",
        f"⇒ 若这些题全部转化为成功，成功率上界 **{summary['reachable_rate_ceiling'] * 100:.1f}%**"
        f"（现 {summary['success_rate'] * 100:.1f}%）。**这是上界，不是预期收益**："
        "打开之后仍要靠规格与预算判断才能满分。",
        "",
    ]

    if examples:
        lines += ["## 5. 逐题样例（出现过但没打开）", ""]
        shown = 0
        for item in diagnosed:
            if item["bucket"] != "saw_gold_not_opened" or shown >= examples:
                continue
            shown += 1
            lines.append(
                f"- task `{item['task_id']}`：正确商品在结果第 **{item['gold_best_rank']}** 名，"
                f"实际打开 `{item['opened_asins'] or '（无）'}`，"
                f"共 {item['opens']} 次打开 / {item['steps']} 步"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="解剖候选探索：模型到底看了几个商品")
    parser.add_argument("trajectories", type=Path, help="轨迹 jsonl 路径")
    parser.add_argument("--out", type=Path, default=None, help="报告输出路径（.md）")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    parser.add_argument("--examples", type=int, default=10, help="逐题样例条数")
    args = parser.parse_args(argv)

    if not args.trajectories.exists():
        print(f"轨迹文件不存在：{args.trajectories}", file=sys.stderr)
        return 2

    rows = []
    with args.trajectories.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    diagnosed = [diagnose(row) for row in rows]
    if not diagnosed:
        print("没有可用轨迹", file=sys.stderr)
        return 1

    summary = aggregate(diagnosed)
    summary["missing_gold"] = summary["buckets"].get("no_gold_reference", 0)
    if args.json:
        payload = json.dumps(
            {
                "source": str(args.trajectories),
                "rows": len(rows),
                "missing_gold": summary["missing_gold"],
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    else:
        payload = render_report(summary, diagnosed, str(args.trajectories), args.examples)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"已写入 {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
