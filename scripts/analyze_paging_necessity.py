#!/usr/bin/env python3
"""Was paging actually needed, and did the model page far enough?

M10-R v2 把翻页从 1.0% 抬到 6.0%，Final-200 却只从 126 动到 127（噪声内）。
"翻页有没有用"这句话在这里是含糊的，它混了两件事：

* 翻的题对不对 —— gold 本来就在第一页，翻了是白翻（v1 的病：31% 到处翻）
* 翻得够不够深 —— gold 在第 41 名，只翻 1 页仍然看不到

两个问题要用不同的尺子量，修法也相反。本脚本把每条轨迹按
「gold 在 BM25 召回里的名次」算出**需要翻几页**，再和**实际翻了几页**、
**有没有买对**交叉，分成六组：

| 组 | 含义 | 期望 |
|---|---|---|
| not_needed | 第一页就能看到 gold，没翻 | 正常 |
| not_needed_paged | 第一页就能看到，却翻了 | 白翻，可能伤害 |
| needed_reached | 需要翻页且翻到了 | 翻页的有效样本 |
| needed_paged_short | 翻了但页数不够 | 差一点，缺示范 |
| needed_not_paged | 该翻没翻 | 漏掉的机会 |
| rank_unknown | 模型没发搜索 / 名次不可知 | 单独看 |

名次用模型**自己发的查询**重查 BM25 得到（不是读观察里的行号），
所以它回答的是"沿着这条轨迹的查询，再往下翻能不能看到 gold"。

纯 CPU、只读：重建本地冻结索引，不需要运行中的商店服务。
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
from pathlib import Path

try:
    from scripts.audit_retrieval_reachability import (
        CORPUS,
        DEFAULT_INDEX,
        SEARCH_ROW,
        load_index,
        load_rows,
        recall_rank,
    )
except ModuleNotFoundError:  # 直接以脚本方式运行时 scripts/ 不在包路径上
    from audit_retrieval_reachability import (  # type: ignore[no-redef]
        CORPUS,
        DEFAULT_INDEX,
        SEARCH_ROW,
        load_index,
        load_rows,
        recall_rank,
    )

GOLD = "gold_purchase"
GROUP_LABELS = collections.OrderedDict(
    [
        ("not_needed", "第一页可见 · 没翻"),
        ("not_needed_paged", "第一页可见 · 却翻了"),
        ("needed_reached", "该翻 · 翻到了"),
        ("needed_paged_short", "该翻 · 页数不够"),
        ("needed_not_paged", "该翻 · 没翻"),
        ("rank_unknown", "名次不可知"),
    ]
)


def pages_needed(rank: int | None, page_size: int) -> int | None:
    """翻几页才能让 ``rank`` 落在已渲染的窗口内。``None`` 表示不可知。"""

    if rank is None:
        return None
    if rank <= page_size:
        return 0
    return math.ceil(rank / page_size) - 1


def paged_pages(row: dict) -> int:
    return sum(1 for step in row.get("steps") or [] if step.get("tool_name") == "next_page")


def first_page_asins(row: dict, page_size: int) -> list[str]:
    """第一次搜索返回的列表里被渲染出来的那些 ASIN（默认前 20 名）。"""

    for step in row.get("steps") or []:
        if step.get("tool_name") == "search_products":
            text = step.get("observation") or ""
            return [match.group("asin") for match in SEARCH_ROW.finditer(text)][:page_size]
    return []


def load_domains(corpus: Path) -> dict[str, str]:
    """asin → 品类。只用来回答「第一页里有没有同类商品」，不参与判定。"""

    with gzip.open(corpus, "rt", encoding="utf-8") as handle:
        products = json.load(handle)
    return {
        item.get("asin"): (item.get("domain_zh") or item.get("domain_en_short") or "")
        for item in products
        if item.get("asin")
    }


def bought_gold(row: dict) -> bool:
    detail = (row.get("terminal_result") or {}).get("reward_detail") or {}
    return detail.get("reward_type") == GOLD


def classify(row: dict, rank: int | None, page_size: int) -> dict:
    need = pages_needed(rank, page_size)
    pages = paged_pages(row)
    if need is None:
        group = "rank_unknown"
    elif need == 0:
        group = "not_needed_paged" if pages else "not_needed"
    elif pages >= need:
        group = "needed_reached"
    elif pages:
        group = "needed_paged_short"
    else:
        group = "needed_not_paged"
    terminal = row.get("terminal_result") or {}
    return {
        "task_id": row.get("task_id"),
        "gold": (terminal.get("goal") or {}).get("asin"),
        "purchased": (terminal.get("purchase") or {}).get("asin"),
        "first_page": first_page_asins(row, page_size),
        "rank": rank,
        "pages_needed": need,
        "pages": pages,
        "paged": pages > 0,
        "success": bought_gold(row),
        "group": group,
    }


def rank_band(rank: int | None, page_size: int) -> str:
    if rank is None:
        return "unknown"
    if rank <= page_size:
        return f"1-{page_size}"
    # 档位按「需要翻几页」切，互不重叠：1 页 / 2 页 / 3-4 页 / 5-7 页 / 8+。
    need = pages_needed(rank, page_size)
    if need == 1:
        return "1 页"
    if need == 2:
        return "2 页"
    if need <= 4:
        return "3-4 页"
    if need <= 7:
        return "5-7 页"
    return "8 页以上"


def summarize(items: list[dict], page_size: int) -> dict:
    groups = collections.OrderedDict()
    for key in GROUP_LABELS:
        groups[key] = {"count": 0, "success": 0}
    bands = collections.OrderedDict()
    for item in items:
        bucket = groups[item["group"]]
        bucket["count"] += 1
        bucket["success"] += int(item["success"])
        band = rank_band(item["rank"], page_size)
        row = bands.setdefault(band, {"count": 0, "paged": 0, "success": 0})
        row["count"] += 1
        row["paged"] += int(item["paged"])
        row["success"] += int(item["success"])
    return {"total": len(items), "groups": groups, "bands": bands}


def format_report(name: str, stats: dict) -> str:
    lines = [f"== {name}（{stats['total']} 条轨迹）"]
    lines.append("| 分组 | 题数 | 买对 | 买对率 |")
    lines.append("|---|---:|---:|---:|")
    for key, label in GROUP_LABELS.items():
        bucket = stats["groups"][key]
        count = bucket["count"]
        if not count:
            continue
        rate = bucket["success"] / count
        lines.append(f"| {label} | {count} | {bucket['success']} | {rate * 100:.1f}% |")

    lines.append("")
    lines.append("| gold 名次（需翻页数） | 题数 | 翻了 | 买对 | 买对率 |")
    lines.append("|---|---:|---:|---:|---:|")
    for band, row in stats["bands"].items():
        rate = row["success"] / row["count"] if row["count"] else 0.0
        lines.append(
            f"| {band} | {row['count']} | {row['paged']} | {row['success']} | {rate * 100:.1f}% |"
        )

    groups = stats["groups"]
    needed = groups["needed_reached"]["count"] + groups["needed_paged_short"]["count"] + groups["needed_not_paged"]["count"]
    reached = groups["needed_reached"]
    lines.append("")
    if needed:
        lines.append(
            f"该翻页的题 {needed} 道：翻到 {reached['count']}、页数不够 "
            f"{groups['needed_paged_short']['count']}、完全没翻 {groups['needed_not_paged']['count']}"
        )
        if reached["count"]:
            lines.append(
                f"翻到的那些买对 {reached['success']}/{reached['count']} = "
                f"{reached['success'] / reached['count'] * 100:.1f}%"
            )
    wasted = groups["not_needed_paged"]["count"]
    if wasted:
        lines.append(
            f"第一页就能看到的题里翻了 {wasted} 道（买对 "
            f"{groups['not_needed_paged']['success']}/{wasted}）—— 这部分是白翻"
        )
    return "\n".join(lines)


def format_detail(items: list[dict], domains: dict[str, str]) -> str:
    """逐题摊开「该翻却没看到 gold」的题，并数第一页里有几个同类商品。

    同类商品数是关键：如果第一页摆着一堆「差不多但不完全对」的诱饵，
    模型很可能买了诱饵而不是继续翻；如果第一页全是无关商品，那它不翻
    就只是没学会「列表为空该翻」这条规则。两者修法不同。
    """

    targets = [i for i in items if i["group"] in ("needed_not_paged", "needed_paged_short")]
    lines = [f"== 逐题明细：该翻却没看到 gold 的 {len(targets)} 道"]
    lines.append("| task_id | gold 名次 | 需翻 | 实翻 | 买到 gold | 第一页同类 | 第一页条数 |")
    lines.append("|---|---:|---:|---:|---|---:|---:|")
    decoys = 0
    for item in targets:
        gold_domain = domains.get(item["gold"], "")
        same = sum(1 for asin in item["first_page"] if domains.get(asin) == gold_domain)
        decoys += int(same > 0)
        bought = "是" if item["purchased"] == item["gold"] else "否"
        lines.append(
            f"| {item['task_id']} | {item['rank']} | {item['pages_needed']} | "
            f"{item['pages']} | {bought} | {same} | {len(item['first_page'])} |"
        )
    if targets:
        lines.append("")
        lines.append(
            f"其中 {decoys}/{len(targets)} 道的第一页里有同类商品"
            f"（「差不多但不完全对」的诱饵）"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trajectories", type=Path, nargs="+")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--recall-depth", type=int, default=150)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--detail", action="store_true", help="逐题打印该翻却没看到 gold 的题")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    searcher = load_index(args.index)
    payload = {}
    for path in args.trajectories:
        items = []
        for row in load_rows(path):
            terminal = row.get("terminal_result") or {}
            gold = (terminal.get("goal") or {}).get("asin")
            queries = [
                (step.get("parameters") or {}).get("query")
                for step in row.get("steps") or []
                if step.get("tool_name") == "search_products"
            ]
            rank = recall_rank(searcher, [q for q in queries if q], gold, args.recall_depth) if gold else None
            items.append(classify(row, rank, args.page_size))
        payload[path.name] = (summarize(items, args.page_size), items)

    if args.json:
        print(json.dumps({name: stats for name, (stats, _) in payload.items()}, ensure_ascii=False, indent=2))
        return 0
    domains = load_domains(CORPUS) if args.detail else {}
    for name, (stats, items) in payload.items():
        print(format_report(name, stats))
        if args.detail:
            print()
            print(format_detail(items, domains))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
