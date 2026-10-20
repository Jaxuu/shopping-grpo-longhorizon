#!/usr/bin/env python3
"""Audit whether a failure was reachable: is the gold product inside BM25 recall?

The earlier M9 conclusion "the gold title shares almost no vocabulary with the
instruction, so no faithful query can retrieve it" only proved that the *first
page* misses.  Environment v2 search is multi-field BM25 with OR semantics over
Chinese bigrams and a recall depth of 150, while the observation renderer shows
only the top ``--page-size`` rows.  A product at rank 23-114 is therefore
*retrievable but never displayed* — a behaviour gap (nobody calls ``next_page``),
not a lexical gap.

This script separates the two claims:

* ``in_recall`` — the ASIN appears in the top-k BM25 hits of a query the model
  itself issued, i.e. one ``next_page`` chain would have surfaced it.
* ``not_in_recall`` — absent even from the full recall list, so only a
  different query can help.

It is read-only and CPU-only: it rebuilds the frozen index from the local
product corpus, so it needs no running environment service.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = REPO_ROOT / "environments" / "ShopSimulator" / "shop_env"

SEARCH_ROW = re.compile(r"^(?P<rank>\d+)\|(?P<asin>[^|\n]+)\|", re.MULTILINE)
SUCCESS_TYPES = {"gold_purchase", "valid_alternative_purchase"}
CORPUS = SHOP_ENV / "data" / "fine_items_eval_train_all.json.gz"
DEFAULT_INDEX = REPO_ROOT / ".workbuddy-ai" / "cache" / "products_local.sqlite3"


def load_index(path: Path):
    """Import the frozen searcher lazily.

    ``sys.path`` is only touched here on purpose: mutating it at module import
    time leaks a sibling ``scripts`` directory into the search path and breaks
    unrelated test modules that do ``import scripts.<name>``.
    """

    shop_env = str(SHOP_ENV)
    if shop_env not in sys.path:
        sys.path.append(shop_env)
    from web_agent_site.engine.search import (
        MultiFieldBM25Searcher,
        build_index,
    )

    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(CORPUS, "rt", encoding="utf-8") as handle:
            products = json.load(handle)
        build_index(iter(products), path, product_data_sha256="local-audit")
    return MultiFieldBM25Searcher(path)


def load_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bucket_row(row: dict) -> tuple[str, dict]:
    """Classify one finished trajectory into the mutually exclusive A/B/C/D set."""

    terminal = row.get("terminal_result") or {}
    detail = terminal.get("reward_detail") or {}
    goal = terminal.get("goal") or {}
    gold = goal.get("asin")
    purchased = (terminal.get("purchase") or {}).get("asin")
    ranks: dict[str, int] = {}
    for step in row.get("steps") or []:
        if step.get("tool_name") != "search_products":
            continue
        for match in SEARCH_ROW.finditer(step.get("observation") or ""):
            ranks.setdefault(match.group("asin"), int(match.group("rank")))
    opened = {
        (step.get("parameters") or {}).get("asin")
        for step in row.get("steps") or []
        if step.get("tool_name") == "open_product"
    }
    queries = [
        (step.get("parameters") or {}).get("query")
        for step in row.get("steps") or []
        if step.get("tool_name") == "search_products"
    ]
    info = {"gold": gold, "queries": [q for q in queries if q], "ranks": ranks, "opened": opened}
    if detail.get("reward_type") in SUCCESS_TYPES:
        return "success", info
    if not gold:
        return "no_gold_reference", info
    if purchased == gold:
        return "D_bought_gold_incomplete", info
    if gold in opened:
        return "C_opened_gold_bought_other", info
    if gold in ranks:
        return "B_seen_not_opened", info
    return "A_never_seen", info


def recall_rank(searcher, queries, gold, depth):
    """Best (smallest) BM25 rank the model's own queries achieve for ``gold``."""

    best = None
    for query in queries:
        if not query:
            continue
        for hit in searcher.search(query, k=depth):
            if hit.asin == gold:
                if best is None or hit.rank < best:
                    best = hit.rank
                break
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--recall-depth", type=int, default=150)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    searcher = load_index(args.index)
    rows = load_rows(args.trajectories)
    buckets = collections.Counter()
    a_ranks: list[int] = []
    a_missing = 0
    for row in rows:
        bucket, info = bucket_row(row)
        buckets[bucket] += 1
        if bucket != "A_never_seen":
            continue
        rank = recall_rank(searcher, info["queries"], info["gold"], args.recall_depth)
        if rank is None:
            a_missing += 1
        else:
            a_ranks.append(rank)

    pages = [0, 1, 2, 3, 4, 5]
    reachable = {
        page: sum(1 for rank in a_ranks if rank <= args.page_size * (page + 1))
        for page in pages
    }
    total = sum(buckets.values())
    summary = {
        "rows": total,
        "buckets": dict(buckets),
        "a_bucket": {
            "total": buckets.get("A_never_seen", 0),
            "in_recall": len(a_ranks),
            "not_in_recall": a_missing,
            "median_rank": sorted(a_ranks)[len(a_ranks) // 2] if a_ranks else None,
            "min_rank": min(a_ranks) if a_ranks else None,
            "max_rank": max(a_ranks) if a_ranks else None,
            "reachable_by_pages": {f"next_page_x{page}": n for page, n in reachable.items()},
        },
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    print(f"轨迹 {total} 条")
    for name, count in sorted(buckets.items()):
        print(f"  {name}: {count}")
    a = summary["a_bucket"]
    print(
        f"\nA 桶 {a['total']} 题：{a['in_recall']} 题的 gold 在 BM25 召回"
        f"（depth={args.recall_depth}）内，{a['not_in_recall']} 题不在。"
    )
    if a_ranks:
        print(f"  名次 中位 {a['median_rank']}  最小 {a['min_rank']}  最大 {a['max_rank']}")
        print(f"  观察只显示前 {args.page_size} 名 ⇒ 这些题「可检索但不可见」")
        for page, count in reachable.items():
            print(f"    翻 {page} 页（看到前 {args.page_size * (page + 1)} 名）：{count} 题")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
