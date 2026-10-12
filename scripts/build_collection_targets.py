#!/usr/bin/env python3
"""Build the M2 collection target list and held-out set.

Scans the whole goal corpus (23,421 tasks), keeps the **high-ambiguity** tasks
(>=20 near-identical candidates in the list page) that are **not already used**
by any pool, verifies them against the admission contract, and writes:

* ``data/collection/collect_targets.jsonl`` -- ``task_id`` rows, ranked by ambiguity;
* ``data/collection/held_out_task_ids.json`` -- every task id that must be excluded;
* ``plans/collection-targets-report.md``     -- selection audit (plans/ is git-ignored).

Deterministic. Reuses the M1 BM25 index cache.
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

from web_agent_site.engine.admission import admit_task  # noqa: E402
from web_agent_site.engine.search import MultiFieldBM25Searcher  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "audit_solvability", ROOT / "scripts" / "audit_solvability.py"
)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)

AMBIGUITY_RATIO = 0.9
MIN_AMBIGUITY = 20
DEFAULT_TARGETS = 242


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=int, default=DEFAULT_TARGETS)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "collection")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "plans" / "collection-targets-report.md",
        help="narrative report destination (plans/ is git-ignored)",
    )
    parser.add_argument("--index", type=Path, default=ROOT / ".workbuddy-ai" / "cache" / "audit_products.sqlite3")
    args = parser.parse_args()

    products = audit.load_products(audit.DEFAULT_PRODUCTS)
    goals = audit.build_goals(products)
    product_by_asin = {str(p["asin"]): p for p in products}

    used: dict[str, set[int]] = {}
    for name, path in audit.DEFAULT_DATASETS.items():
        if path.exists():
            used[name] = set(audit.task_ids(path))
    held_out = set().union(*used.values())
    print(f"已用 task_id 合计 {len(held_out)}（{ {k: len(v) for k, v in used.items()} }）")

    searcher = MultiFieldBM25Searcher(args.index)
    candidates: list[dict] = []
    scanned = 0
    try:
        for task_id in range(len(goals)):
            if task_id in held_out:
                continue
            scanned += 1
            goal = goals[task_id]
            query = audit.retrieval_queries(goal)["Q4_category_plus_attributes"]
            hits = searcher.search(query, k=150)
            gold = str(goal["asin"])
            gold_hit = next((h for h in hits if str(h.asin) == gold), None)
            if gold_hit is None:
                continue
            ambiguity = sum(
                1
                for h in hits
                if str(h.asin) != gold and h.score >= AMBIGUITY_RATIO * gold_hit.score
            )
            if ambiguity < MIN_AMBIGUITY:
                continue
            candidates.append(
                {
                    "task_id": task_id,
                    "ambiguity": ambiguity,
                    "gold_rank_Q4": gold_hit.rank,
                    "instruction": goal["instruction_text"],
                }
            )
    finally:
        searcher.close()
    print(f"扫描 {scanned} 题，候选（ambiguity ≥ {MIN_AMBIGUITY}）{len(candidates)} 题")

    # 按 ambiguity 降序，逐题过准入契约，取前 N 个通过的
    candidates.sort(key=lambda r: (-r["ambiguity"], r["task_id"]))
    selected: list[dict] = []
    reason_counts: Counter = Counter()
    for row in candidates:
        if len(selected) >= args.targets:
            break
        task_id = row["task_id"]
        goal = goals[task_id]
        verdict = admit_task(
            task_id,
            goal,
            product_by_asin[str(goal["asin"])],
            held_out_task_ids=held_out,
        )
        if not verdict["admitted"]:
            reason_counts.update(verdict["reasons"])
            continue
        row["admission"] = "pass"
        selected.append(row)
    print(f"通过准入契约并入选 {len(selected)} 题")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "collect_targets.jsonl").write_text(
        "".join(
            json.dumps({"task_id": r["task_id"]}, ensure_ascii=False) + "\n"
            for r in selected
        ),
        encoding="utf-8",
        newline="\n",
    )
    (out / "held_out_task_ids.json").write_text(
        json.dumps(sorted(held_out), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    # ``--held-out-tasks`` reads JSONL rows carrying ``task_id`` (see
    # ``task_ids_from_jsonl``), so ship a JSONL twin next to the JSON list.
    (out / "held_out_task_ids.jsonl").write_text(
        "".join(
            json.dumps({"task_id": task_id}, ensure_ascii=False) + "\n"
            for task_id in sorted(held_out)
        ),
        encoding="utf-8",
        newline="\n",
    )
    (out / "collect_targets_detail.jsonl").write_text(
        "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in selected
        ),
        encoding="utf-8",
        newline="\n",
    )

    amb = sorted(r["ambiguity"] for r in selected)
    lines = [
        "# M2 采集目标清单（selection audit）",
        "",
        f"- 扫描未使用任务：**{scanned}** 题",
        f"- 高歧义候选（`ambiguity ≥ {MIN_AMBIGUITY}`）：**{len(candidates)}** 题",
        f"- 通过准入契约并入选：**{len(selected)}** 题（目标 {args.targets}）",
        f"- 排除集（评测集 ∪ SFT ∪ GRPO）：**{len(held_out)}** 个 task_id",
        "",
        "## 入选清单的 ambiguity 分布",
        "",
        f"- min **{amb[0]}** / median **{amb[len(amb)//2]}** / max **{amb[-1]}**",
        f"- `gold_rank_Q4` 中位 {sorted(r['gold_rank_Q4'] for r in selected)[len(selected)//2]}",
        "",
        "## 被准入契约拒绝的候选（原因分布）",
        "",
        "| 原因 | 次数 |",
        "|---|---:|",
    ]
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| `{reason}` | {count} |")
    lines += [
        "",
        "## 文件",
        "",
        "```text",
        "data/collection/collect_targets.jsonl         采集任务清单（task_id，按 ambiguity 降序）",
        "data/collection/collect_targets_detail.jsonl  含 ambiguity / gold_rank / instruction",
        "data/collection/held_out_task_ids.json        排除集（必须传给 --held-out-tasks）",
        "plans/collection-targets-report.md            本文件",
        "```",
        "",
    ]
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    print("output:", out.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
