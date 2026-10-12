#!/usr/bin/env python3
"""Long-tail structural probe.

Answers one question: **does any task structurally require >= 21 tool calls?**
See ``plans/longtail-probe-plan.md`` for the definitions and decision rule.

Deterministic, CPU-only, reuses the M1 BM25 index cache.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments" / "ShopSimulator" / "shop_env"
if str(SHOP_ENV) not in sys.path:
    sys.path.append(str(SHOP_ENV))

from web_agent_site.engine.search import (  # noqa: E402
    MultiFieldBM25Searcher,
    build_index,
)

_SPEC = importlib.util.spec_from_file_location(
    "audit_solvability", ROOT / "scripts" / "audit_solvability.py"
)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)

LONG_TAIL_THRESHOLD = 21
AMBIGUITY_RATIO = 0.9
AMBIGUITY_CAP = 20
PAGE_SIZE = 20


def _index_path() -> Path:
    return ROOT / ".workbuddy-ai" / "cache" / "audit_products.sqlite3"


def _ensure_index(products) -> Path:
    path = _index_path()
    product_sha = audit.sha256_bytes(
        json.dumps([p.get("asin") for p in products], ensure_ascii=False).encode("utf-8")
    )
    if path.is_file():
        searcher = MultiFieldBM25Searcher(path)
        current = searcher.manifest.get("product_data_sha256") == product_sha
        searcher.close()
        if current:
            return path
    path.parent.mkdir(parents=True, exist_ok=True)
    build_index(products, path, product_data_sha256=product_sha)
    return path


def probe_dataset(searcher, goals, ids, dataset):
    rows = []
    for task_id in ids:
        goal = goals[task_id]
        n_axes = len(goal.get("required_options_by_key") or {}) + len(
            goal.get("unresolved_option_requirements") or []
        )
        floor = n_axes + 3
        query = audit.retrieval_queries(goal)["Q4_category_plus_attributes"]
        hits = searcher.search(query, k=150)
        gold = str(goal["asin"])
        rank = next((h.rank for h in hits if str(h.asin) == gold), None)
        ambiguity = 0
        if rank is not None:
            gold_score = next(h.score for h in hits if str(h.asin) == gold)
            ambiguity = sum(
                1
                for h in hits
                if str(h.asin) != gold and h.score >= AMBIGUITY_RATIO * gold_score
            )
        search_pages = math.ceil(rank / PAGE_SIZE) if rank else 1
        # search_pages (searches) + 1 (open gold) + ambiguity (extra opens)
        # + n_axes (selects) + 1 (buy)
        upper = search_pages + 1 + min(ambiguity, AMBIGUITY_CAP) + n_axes + 1
        rows.append(
            {
                "dataset": dataset,
                "task_id": int(task_id),
                "n_axes": n_axes,
                "L_floor": floor,
                "gold_rank_Q4": rank,
                "search_pages": search_pages,
                "ambiguity": ambiguity,
                "L_upper": upper,
            }
        )
    return rows


def _bucket(value):
    if value <= 5:
        return "1-5"
    if value <= 10:
        return "6-10"
    if value <= 20:
        return "11-20"
    return "21-35"


def summarize(rows):
    floors = sorted(r["L_floor"] for r in rows)
    uppers = sorted(r["L_upper"] for r in rows)
    ambiguities = sorted(r["ambiguity"] for r in rows)
    ranks = sorted(r["gold_rank_Q4"] for r in rows if r["gold_rank_Q4"])
    long_tail = sorted(
        (r for r in rows if r["L_upper"] >= LONG_TAIL_THRESHOLD),
        key=lambda r: (-r["L_upper"], r["task_id"]),
    )
    return {
        "tasks": len(rows),
        "L_floor": {
            "min": floors[0],
            "median": floors[len(floors) // 2],
            "max": floors[-1],
            "buckets": dict(sorted(Counter(_bucket(v) for v in floors).items())),
        },
        "L_upper": {
            "min": uppers[0],
            "median": uppers[len(uppers) // 2],
            "max": uppers[-1],
            "buckets": dict(sorted(Counter(_bucket(v) for v in uppers).items())),
        },
        "ambiguity": {
            "min": ambiguities[0],
            "median": ambiguities[len(ambiguities) // 2],
            "p90": ambiguities[int(len(ambiguities) * 0.9)],
            "max": ambiguities[-1],
            "count_ge_cap": sum(1 for v in ambiguities if v >= AMBIGUITY_CAP),
        },
        "gold_rank_Q4": {
            "median": ranks[len(ranks) // 2] if ranks else None,
            "p90": ranks[int(len(ranks) * 0.9)] if ranks else None,
            "max": ranks[-1] if ranks else None,
        },
        "tasks_with_L_upper_ge_threshold": len(long_tail),
        "candidate_long_tail_task_ids": [r["task_id"] for r in long_tail],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "audit" / "longtail_probe.json")
    args = parser.parse_args()

    products = audit.load_products(audit.DEFAULT_PRODUCTS)
    goals = audit.build_goals(products)
    index_path = _ensure_index(products)

    dataset_ids = {
        name: audit.task_ids(path)
        for name, path in audit.DEFAULT_DATASETS.items()
        if path.exists()
    }

    searcher = MultiFieldBM25Searcher(index_path)
    try:
        all_rows = []
        per_dataset = {}
        for name, ids in dataset_ids.items():
            rows = probe_dataset(searcher, goals, ids, name)
            all_rows.extend(rows)
            per_dataset[name] = summarize(rows)
    finally:
        searcher.close()

    payload = {
        "schema_version": "shopping-longtail-probe-v1",
        "definitions": {
            "L_floor": "n_axes + 3 (oracle path: search + open + select*axes + buy)",
            "L_upper": "search_pages + min(ambiguity, 20) + n_axes + 1",
            "ambiguity_ratio": AMBIGUITY_RATIO,
            "ambiguity_cap": AMBIGUITY_CAP,
            "page_size": PAGE_SIZE,
            "long_tail_threshold": LONG_TAIL_THRESHOLD,
        },
        "caveat": (
            "Structural proxy, not a rollout. The 27B per-task trajectories are not "
            "in the repo, so the actual 21-35 step tasks cannot be identified."
        ),
        "datasets": per_dataset,
        "rows": sorted(all_rows, key=lambda r: (r["dataset"], r["task_id"])),
    }
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    for name, summary in per_dataset.items():
        print(f"--- {name}")
        print(f"  L_floor  min/median/max = {summary['L_floor']['min']}/"
              f"{summary['L_floor']['median']}/{summary['L_floor']['max']}  "
              f"buckets={summary['L_floor']['buckets']}")
        print(f"  L_upper  min/median/max = {summary['L_upper']['min']}/"
              f"{summary['L_upper']['median']}/{summary['L_upper']['max']}  "
              f"buckets={summary['L_upper']['buckets']}")
        print(f"  L_upper >= {LONG_TAIL_THRESHOLD}: "
              f"{summary['tasks_with_L_upper_ge_threshold']} tasks "
              f"{summary['candidate_long_tail_task_ids'][:20]}")
    print("output:", args.out.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
