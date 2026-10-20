#!/usr/bin/env python3
"""M10-R：从采集到的 raw.jsonl 里筛出「正确答案不在第 1 名」的成功轨迹。

筛什么
------
1. 先过 `acceptance_reasons`（与现有池子同一套验收：`gold_purchase` + 有效 + 已结束）；
2. 再要求 **gold 在老师自己搜到的结果里从来没排过第 1 名**（取所有搜索/翻页里最好的名次）。

第 2 条是本轮唯一新增的口径，理由是实测：

* SFT 池 1,246 条里，打开的 ASIN 有 **65.9%** 就是第 1 名；
* 评测里 **第 1 名对 → 86.8%，不对 → 34.0%**；
* ⇒ 模型学成「开第一个」。要让「开第一个」亏本，训练集里「答案不在第 1 名」的样本得占多数。

为什么读 raw.jsonl
------------------
`accepted.jsonl` 只剩成功轨迹，无法回算名次；而且它由 `build_collection_artifacts`
生成，口径固定为 `gold_purchase`，本脚本要再加一条名次条件。

用法::

    .venv/Scripts/python.exe scripts/filter_m10_rank2.py \\
        outputs/m10-rank2/raw.jsonl \\
        --out outputs/m10-rank2/kept.jsonl

    ... --min-rank 3          # 更严：只要第 3 名以后
    ... --stats-only          # 只看命中率，不写文件
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 搜索结果行首的 `rank|asin|`。投影截断只影响后半段字段，不影响这两个。
SEARCH_ROW = re.compile(r"^(?P<rank>\d+)\|(?P<asin>[^|\n]+)\|", re.MULTILINE)

#: 只有 `next_page` 算「翻页示范」。
#: ⚠️ `prev_page` 名字里也有 page，但实测 2,032 次几乎全是「从详情页/信息子页返回列表」，
#: 不是翻搜索结果页；把它算进来会把翻页示范从 14 条虚报成 1,089 条。
PAGING_TOOLS = frozenset({"next_page"})
FIRST_PAGE_END = 20


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def gold_rank(trajectory: dict, gold: str) -> int | None:
    """gold 在老师**自己看到过**的所有结果里最好的名次；从没出现返回 None。"""
    best: int | None = None
    for step in trajectory.get("steps") or []:
        if step.get("tool_name") != "search_products":
            continue
        text = str(step.get("observation") or "")
        for match in SEARCH_ROW.finditer(text):
            if match.group("asin") == gold:
                rank = int(match.group("rank"))
                best = rank if best is None else min(best, rank)
    return best


def opened_asins(trajectory: dict) -> list[str]:
    return [
        str((step.get("parameters") or {}).get("asin"))
        for step in trajectory.get("steps") or []
        if step.get("tool_name") == "open_product"
        and (step.get("parameters") or {}).get("asin")
    ]


def uses_paging(trajectory: dict) -> bool:
    """老师有没有真的翻页（`next_page` / `prev_page` 都是无参数工具）。"""
    return any(
        step.get("tool_name") in PAGING_TOOLS for step in trajectory.get("steps") or []
    )


def is_kept(rank: int | None, min_rank: int, max_rank: int = 0) -> bool:
    """名次条件：必须出现过（不是 None），且 ≥min_rank；max_rank=0 表示不限。"""
    if rank is None or rank < min_rank:
        return False
    if max_rank and rank > max_rank:
        return False
    return True


def summarize(rows: list[dict], accepted: list[dict], kept: list[dict]) -> dict:
    ranks = collections.Counter()
    for row in accepted:
        gold = ((row.get("terminal_result") or {}).get("goal") or {}).get("asin")
        rank = gold_rank(row, str(gold)) if gold else None
        ranks["未出现" if rank is None else rank] += 1

    # 留下的样本里，「必须翻页才看得到」的有多少 —— 这是 A 桶（gold 在第 21 名以后）
    # 唯一能靠数据补上的那一块，必须单独数出来，不能混在 383 里当"已覆盖"。
    kept_beyond_first_page = 0
    kept_paging = 0
    for row in kept:
        gold = ((row.get("terminal_result") or {}).get("goal") or {}).get("asin")
        rank = gold_rank(row, str(gold)) if gold else None
        if rank is not None and rank > FIRST_PAGE_END:
            kept_beyond_first_page += 1
        if uses_paging(row):
            kept_paging += 1

    opened_counts = collections.Counter(len(opened_asins(row)) for row in kept)
    steps = [len(row.get("steps") or []) for row in kept]
    return {
        "trajectories": len(rows),
        "accepted": len(accepted),
        "kept": len(kept),
        "hit_rate": (len(kept) / len(rows)) if rows else 0.0,
        "kept_share_of_accepted": (len(kept) / len(accepted)) if accepted else 0.0,
        "rank_distribution_of_accepted": {
            str(key): value
            for key, value in sorted(
                ranks.items(),
                # 数字名次按数值排在前，字符串（"未出现"）排最后
                key=lambda kv: (1, 0) if isinstance(kv[0], str) else (0, kv[0]),
            )
        },
        "kept_single_open": opened_counts.get(1, 0),
        "kept_multi_open": sum(v for k, v in opened_counts.items() if k >= 2),
        "kept_beyond_first_page": kept_beyond_first_page,
        "kept_with_paging": kept_paging,
        "kept_mean_steps": (sum(steps) / len(steps)) if steps else 0.0,
        "kept_tasks": len({str(row.get("task_id")) for row in kept}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("raw", type=Path, help="采集产出的 raw.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="筛出的 SFT 行写到这里")
    parser.add_argument("--min-rank", type=int, default=2, help="gold 名次至少多少（默认 2）")
    parser.add_argument("--max-rank", type=int, default=0, help="0 表示不限")
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not args.raw.is_file():
        raise SystemExit(f"轨迹文件不存在: {args.raw}")

    sys.path.insert(0, str(ROOT / "src"))
    from shopping_grpo.collection.sft import (  # noqa: PLC0415
        acceptance_reasons,
        build_sft_row,
    )

    rows = load_jsonl(args.raw)
    accepted: list[dict] = []
    kept: list[dict] = []
    kept_ids: set[str] = set()
    for row in rows:
        ok, _ = acceptance_reasons(row)
        if not ok:
            continue
        accepted.append(row)
        gold = ((row.get("terminal_result") or {}).get("goal") or {}).get("asin")
        if not gold:
            continue
        rank = gold_rank(row, str(gold))
        if not is_kept(rank, args.min_rank, args.max_rank):
            continue
        task_key = str(row.get("task_id"))
        if task_key in kept_ids:  # 同一题只留第一条
            continue
        kept_ids.add(task_key)
        kept.append(row)

    stats = summarize(rows, accepted, kept)
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        print(f"轨迹 {stats['trajectories']} 条 → 过验收 {stats['accepted']} 条 → 留下 {stats['kept']} 条")
        print(f"  命中率（占全部轨迹）: {stats['hit_rate']:.1%}")
        print(f"  占已成功轨迹: {stats['kept_share_of_accepted']:.1%}")
        print(f"  留下题数: {stats['kept_tasks']}；平均步数 {stats['kept_mean_steps']:.1f}")
        print(f"  只开 1 个: {stats['kept_single_open']}；开 ≥2 个: {stats['kept_multi_open']}")
        print(
            f"  gold 在第 21 名以后（必须翻页才看得到）: {stats['kept_beyond_first_page']}；"
            f"真的翻了页: {stats['kept_with_paging']}"
        )
        print("  已成功轨迹的 gold 名次分布（前 10）:")
        for rank, count in list(stats["rank_distribution_of_accepted"].items())[:10]:
            print(f"    第 {rank} 名: {count}")

    if args.stats_only or args.out is None:
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(build_sft_row(row), ensure_ascii=False) + "\n")
    print(f"\n写入 {args.out}：{len(kept)} 行（schema 与现有 SFT 池一致）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
