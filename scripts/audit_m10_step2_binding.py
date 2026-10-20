#!/usr/bin/env python3
"""M10 Step 2 复核：选题在**老师自己的 query** 下，是否真的构成了「必须做目标动作」的处境。

为什么需要这一步
----------------
Step 2 采了 90 题，三个场景的目标动作率全部 <20%（6.7% / 6.7% / 3.6%）。
但「动作率低」有两种完全不同的解释：

* **A. 老师不会做** → 行为缺口，补示范有意义（但三条都关，等于补不了）。
* **B. 题目在老师眼里根本不需要做** → 选题失效，那个 <20% 什么都不能证明。

Step 1 的选题用的是「三个忠实查询取**最好**名次」，那是**我们猜老师会怎么写**。
本脚本改用**老师实际打进去的 query**（就存在 Step 2 的 `raw.jsonl` 里）重算一遍，
把 A / B 分开。这一步纯 CPU，不调 API。

判据（每个场景的「处境成立」怎么算）
------------------------------------
* **S1 翻页**：gold 在老师 query 下名次 **>20 或不在 150 召回内** ⇒ 真的必须翻页。
* **S2 比较**：gold 名次 **2–20**（第 1 页可见但不是第一）⇒ 真的存在「选哪个」的问题。
* **S3 排除**：老师第一次搜索的**第 1 名最低价 > 预算** ⇒ 真的必须先排除一个太贵的。

关键数字不是「绑定率」，而是**绑定题里的动作率**：
如果处境成立时老师有 50%+ 会做，说明它会做，是选题把结果稀释了（解释 B）。

用法::

    .venv/bin/python scripts/audit_m10_step2_binding.py \\
        outputs/m10-step2/raw.jsonl \\
        --targets data/collection/m10_step2_targets.jsonl

    ... --detail      # 逐题明细
    ... --json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments" / "ShopSimulator" / "shop_env"
SCRIPT_DIR = Path(__file__).resolve().parent

#: 服务器上有环境索引（`setup.sh` 建的）；本机只有审计用的缓存索引。
INDEX_CANDIDATES = (
    SHOP_ENV / "search_engine" / "products.sqlite3",
    ROOT / ".workbuddy-ai" / "cache" / "products_local.sqlite3",
)

#: 场景 → 判定用的动作键（与 analyze_m10_step2_verification 对齐）。
ACTION_KEY = {
    "S1_paging": "used_next_page",
    "S2_compare": "bought_earlier_candidate",
    "S3_budget_exclude": "excluded_over_budget",
}


def _siblings():
    """惰性导入同目录脚本，避免顶层污染 `sys.path`。"""
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import analyze_m10_step2_verification as step2  # noqa: PLC0415
    import audit_retrieval_reachability as retrieval  # noqa: PLC0415

    return step2, retrieval


def default_index() -> Path:
    for candidate in INDEX_CANDIDATES:
        if candidate.is_file():
            return candidate
    return INDEX_CANDIDATES[0]


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


def teacher_queries(row: dict) -> list[str]:
    queries = []
    for step in row.get("steps") or []:
        if step.get("tool_name") != "search_products":
            continue
        query = (step.get("parameters") or {}).get("query")
        if query:
            queries.append(str(query))
    return queries


def first_search_rows(row: dict, step2) -> list[dict]:
    for step in row.get("steps") or []:
        if step.get("tool_name") == "search_products":
            return step2.search_rank_low_prices(step2.observation_text(step))
    return []


def binding_of(tier: str, rank_gold, search_rows, budget) -> tuple[bool, str]:
    """返回 (处境是否成立, 一句解释)。纯函数，方便测试。"""
    if tier == "S1_paging":
        if rank_gold is None:
            return True, "gold 不在 150 召回内 → 必须换词/翻页"
        if rank_gold > 20:
            return True, f"gold 第 {rank_gold} 名 → 必须翻页"
        return False, f"gold 第 {rank_gold} 名 → 第 1 页就看得见，不用翻页"

    if tier == "S2_compare":
        if rank_gold is None:
            return False, "gold 不在召回内 → 不是「比较」问题"
        if 2 <= rank_gold <= 20:
            return True, f"gold 第 {rank_gold} 名 → 第 1 页可见但不是第一"
        if rank_gold == 1:
            return False, "gold 就是第 1 名 → 没有可比较的对手"
        return False, f"gold 第 {rank_gold} 名 → 要翻页，不是「比较」问题"

    if tier == "S3_budget_exclude":
        if budget is None:
            return False, "指令里没有明确预算 → 无从排除"
        if not search_rows:
            return False, "老师没有搜索记录"
        top = search_rows[0]
        if top["low"] is None:
            return False, "第 1 名价格不可读"
        if top["low"] > budget:
            return True, f"第 1 名最低价 {top['low']:.0f} > 预算 {budget:.0f} → 必须排除"
        return False, f"第 1 名最低价 {top['low']:.0f} ≤ 预算 {budget:.0f} → 没有要排除的"

    return False, "未知场景"


def summarize(rows, tiers, searcher, recall_depth, budget_parser, step2, retrieval) -> dict:
    by_tier: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        record = step2.analyze(row, budget_parser=budget_parser)
        try:
            task_id = int(record["task_id"])
        except (TypeError, ValueError):
            continue
        tier = tiers.get(task_id, "unknown")
        queries = teacher_queries(row)
        rank_gold = None
        if searcher is not None and record["gold"] and queries:
            rank_gold = retrieval.recall_rank(searcher, queries, record["gold"], recall_depth)
        search_rows = first_search_rows(row, step2)
        binding, reason = binding_of(tier, rank_gold, search_rows, record["budget"])
        action_key = ACTION_KEY.get(tier)
        by_tier[tier].append(
            {
                "task_id": task_id,
                "tier": tier,
                "queries": len(queries),
                "rank_gold": rank_gold,
                "budget": record["budget"],
                "top1_price": (search_rows[0]["low"] if search_rows else None),
                "binding": binding,
                "reason": reason,
                "action": bool(record[action_key]) if action_key else False,
                "action_key": action_key,
                "success": record["success"],
                "distinct_opened": record["distinct_count"],
                "step_count": record["step_count"],
            }
        )

    report: dict = {"by_tier": {}, "totals": {}}
    for tier, items in sorted(by_tier.items()):
        bound = [item for item in items if item["binding"]]
        unbound = [item for item in items if not item["binding"]]
        report["by_tier"][tier] = {
            "count": len(items),
            "bound": len(bound),
            "binding_rate": (len(bound) / len(items)) if items else 0.0,
            "action_rate_all": (
                sum(1 for item in items if item["action"]) / len(items) if items else 0.0
            ),
            "action_rate_bound": (
                sum(1 for item in bound if item["action"]) / len(bound) if bound else None
            ),
            "action_rate_unbound": (
                sum(1 for item in unbound if item["action"]) / len(unbound) if unbound else None
            ),
            "mean_steps": (sum(i["step_count"] for i in items) / len(items)) if items else 0.0,
            "items": items,
        }
    report["totals"] = {
        "trajectories": len(rows),
        "tasks": len({item["task_id"] for items in by_tier.values() for item in items}),
    }
    return report


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.0%}"


def render(report: dict) -> None:
    print(f"复核 {report['totals']['tasks']} 道题（{report['totals']['trajectories']} 条轨迹）")
    print("\n| 场景 | 题数 | 处境成立 | 绑定率 | 绑定题里动作率 | 未绑定题里动作率 | 动作率（全部） |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for tier, entry in report["by_tier"].items():
        print(
            f"| {tier} | {entry['count']} | {entry['bound']} | "
            f"{entry['binding_rate']:.0%} | {_fmt(entry['action_rate_bound'])} | "
            f"{_fmt(entry['action_rate_unbound'])} | {entry['action_rate_all']:.0%} |"
        )

    print("\n判读：")
    for tier, entry in report["by_tier"].items():
        if entry["count"] == 0:
            continue
        if entry["binding_rate"] < 0.5:
            print(
                f"  {tier}: **选题失效** —— 只有 {entry['bound']}/{entry['count']} 题在老师眼里"
                f"构成了「必须做这个动作」的处境，Step 2 的动作率不能作数。"
            )
        elif entry["action_rate_bound"] is not None and entry["action_rate_bound"] >= 0.5:
            print(
                f"  {tier}: **老师会做** —— 处境成立时动作率 "
                f"{entry['action_rate_bound']:.0%}，是选题把结果稀释了。"
            )
        else:
            print(
                f"  {tier}: **行为缺口成立** —— 处境成立（{entry['bound']}/{entry['count']}）"
                f"但动作率只有 {_fmt(entry['action_rate_bound'])}。"
            )


def render_detail(report: dict) -> None:
    print("\n逐题明细：")
    print("| task_id | 场景 | gold名次 | 第1名最低价 | 预算 | 处境成立 | 老师动作 | 买对 |")
    print("|---|---|---:|---:|---:|---|---|---|")
    for tier, entry in report["by_tier"].items():
        for item in entry["items"]:
            rank = "150+" if item["rank_gold"] is None else str(item["rank_gold"])
            top1 = "—" if item["top1_price"] is None else f"{item['top1_price']:.0f}"
            budget = "—" if item["budget"] is None else f"{item['budget']:.0f}"
            print(
                f"| {item['task_id']} | {tier} | {rank} | {top1} | {budget} | "
                f"{'Y' if item['binding'] else '.'} | {'Y' if item['action'] else '.'} | "
                f"{'Y' if item['success'] else '.'} |"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("trajectories", type=Path, help="Step 2 的 raw.jsonl")
    parser.add_argument("--targets", type=Path, required=True, help="带 tier 的选题清单")
    parser.add_argument("--index", type=Path, default=None, help="BM25 索引（默认自动找）")
    parser.add_argument("--recall-depth", type=int, default=150)
    parser.add_argument("--detail", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="不查 BM25（只判 S3 的预算处境，S1/S2 会判成「未绑定」）",
    )
    args = parser.parse_args(argv)

    step2, retrieval = _siblings()
    if not args.trajectories.is_file():
        raise SystemExit(f"轨迹文件不存在: {args.trajectories}")
    if not args.targets.is_file():
        raise SystemExit(f"选题清单不存在: {args.targets}")

    rows = load_jsonl(args.trajectories)
    tiers = step2.load_tiers(args.targets)
    searcher = None
    if not args.no_index:
        index_path = args.index or default_index()
        print(f"索引: {index_path}")
        searcher = retrieval.load_index(index_path)

    report = summarize(
        rows,
        tiers,
        searcher,
        args.recall_depth,
        step2._budget_parser(),
        step2,
        retrieval,
    )
    if args.json:
        for entry in report["by_tier"].values():
            entry["items"] = entry["items"]
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0

    render(report)
    if args.detail:
        render_detail(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
