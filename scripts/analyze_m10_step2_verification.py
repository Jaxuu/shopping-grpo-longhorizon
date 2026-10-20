#!/usr/bin/env python3
"""M10 Step 2：验证老师在被选中的题上**真的做出了目标动作**。

为什么需要这一步
----------------
M8/M9 的两次返工都源于同一个假设：「选题保证了难度，老师就会做目标动作」。
事实是老师会退化成「搜一次、开第一个、买它」。所以采集之前必须先量一次。

三个场景各自的「目标动作」（互斥口径，不重叠计数）
--------------------------------------------------
* **S1 翻页**：轨迹里出现 `next_page`。这是唯一证据——不能看步数多就猜它翻页了。
* **S2 候选筛选 + 回头买**：`open_product` ≥2 次，且**最终买的不是最后一次打开的那个**。
  M9 的教训：「开了几个」是错的度量，环境只允许买当前打开的那个，所以「开得多」等于
  「买最后一个」。真正要学的是**比较之后回头买对的**。
* **S3 超预算排除**：`open_product` ≥2 次，且存在「打开了 A 却没买 A、后面买了 B」。
  这是「排除」动作的充分证据。

用法::

    .venv/Scripts/python.exe scripts/analyze_m10_step2_verification.py \
        outputs/sft-collection-m10/trajectories.jsonl \
        --targets data/collection/m10_step2_targets.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_tiers(path: Path | None) -> dict[int, str]:
    if path is None or not path.is_file():
        return {}
    tiers: dict[int, str] = {}
    for row in load_jsonl(path):
        try:
            tiers[int(row["task_id"])] = str(row.get("tier") or "")
        except (KeyError, TypeError, ValueError):
            continue
    return tiers


def actions(row: dict) -> dict:
    """把一条轨迹压成判定所需的最小结构。"""
    opened: list[str] = []
    tools: list[str] = []
    for step in row.get("steps") or []:
        name = step.get("tool_name") or ""
        tools.append(name)
        if name == "open_product":
            asin = (step.get("parameters") or {}).get("asin")
            if asin:
                opened.append(str(asin))
    terminal = row.get("terminal_result") or {}
    detail = terminal.get("reward_detail") or {}
    goal = terminal.get("goal") or {}
    purchased = (terminal.get("purchase") or {}).get("asin")
    return {
        "task_id": row.get("task_id"),
        "reward_type": detail.get("reward_type"),
        "gold": str(goal.get("asin")) if goal.get("asin") else None,
        "purchased": str(purchased) if purchased else None,
        "opened": opened,
        "tools": tools,
        "success": detail.get("reward_type") in {"gold_purchase", "valid_alternative_purchase"},
        "used_next_page": "next_page" in tools,
        "search_count": tools.count("search_products"),
        "open_count": len(opened),
        # 「回头买」：买了前面打开过的，而不是最后一次打开的那个
        "bought_earlier_candidate": bool(
            purchased and len(opened) >= 2 and str(purchased) != opened[-1]
            and str(purchased) in opened[:-1]
        ),
        # 「排除」：打开了某个商品，最终买的不是它
        "excluded_a_candidate": bool(
            purchased and opened and str(purchased) not in opened
        ) or bool(
            purchased and len(opened) >= 2 and str(purchased) != opened[-1]
        ),
    }


def summarize(rows: list[dict], tiers: dict[int, str]) -> dict:
    by_tier: dict[str, list[dict]] = collections.defaultdict(list)
    overall = []
    for row in rows:
        record = actions(row)
        overall.append(record)
        try:
            task_id = int(record["task_id"])
        except (TypeError, ValueError):
            continue
        by_tier[tiers.get(task_id, "unknown")].append(record)

    def rate(records: list[dict], key: str) -> tuple[int, int, float]:
        hit = sum(1 for r in records if r[key])
        total = len(records)
        return hit, total, (hit / total if total else 0.0)

    report = {"overall": {}, "by_tier": {}}
    for key in ("used_next_page", "bought_earlier_candidate", "excluded_a_candidate", "success"):
        hit, total, ratio = rate(overall, key)
        report["overall"][key] = {"hit": hit, "total": total, "rate": ratio}
    for tier, records in sorted(by_tier.items()):
        entry = {"count": len(records)}
        for key in ("used_next_page", "bought_earlier_candidate", "excluded_a_candidate", "success"):
            hit, total, ratio = rate(records, key)
            entry[key] = {"hit": hit, "total": total, "rate": ratio}
        report["by_tier"][tier] = entry
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("--targets", type=Path, default=None, help="带 tier 的选题清单")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--min-rate", type=float, default=0.5, help="闸门 G2 的门槛")
    args = parser.parse_args(argv)

    rows = load_jsonl(args.trajectories)
    tiers = load_tiers(args.targets)
    report = summarize(rows, tiers)
    report["gate"] = {"threshold": args.min_rate}

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print(f"轨迹 {len(rows)} 条；有 tier 标注的题 {len(tiers)} 个")
    print("\n整体动作率：")
    labels = {
        "used_next_page": "用过 next_page（S1 目标动作）",
        "bought_earlier_candidate": "回头买前面打开过的候选（S2 目标动作）",
        "excluded_a_candidate": "排除了某个候选（S3 目标动作）",
        "success": "最终买对（gold_purchase）",
    }
    for key, label in labels.items():
        entry = report["overall"][key]
        print(f"  {label}: {entry['hit']}/{entry['total']} = {entry['rate']:.1%}")

    if report["by_tier"]:
        print("\n按场景：")
        print("| 场景 | 轨迹数 | 翻页率 | 回头买率 | 排除率 | 成功率 |")
        print("|---|---:|---:|---:|---:|---:|")
        for tier, entry in report["by_tier"].items():
            print(
                f"| {tier} | {entry['count']} | "
                f"{entry['used_next_page']['rate']:.1%} | "
                f"{entry['bought_earlier_candidate']['rate']:.1%} | "
                f"{entry['excluded_a_candidate']['rate']:.1%} | "
                f"{entry['success']['rate']:.1%} |"
            )

    print("\n闸门 G2 判定（目标动作率 ≥{:.0%} 才进入批量采集）：".format(args.min_rate))
    verdict_map = {
        "S1_paging": "used_next_page",
        "S2_compare": "bought_earlier_candidate",
        "S3_budget_exclude": "excluded_a_candidate",
    }
    for tier, key in verdict_map.items():
        entry = report["by_tier"].get(tier)
        if not entry:
            print(f"  {tier}: 本次没有采样")
            continue
        value = entry[key]["rate"]
        if value >= args.min_rate:
            verdict = "**通过**"
        elif value >= 0.2:
            verdict = "部分有效（提 temperature 多次采样再筛）"
        else:
            verdict = "**不通过**（改选题策略，不要批量采）"
        print(f"  {tier}: {value:.1%} → {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
