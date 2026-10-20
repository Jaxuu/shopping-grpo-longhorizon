#!/usr/bin/env python3
"""M10 Step 2：验证老师在被选中的题上**真的做出了目标动作**。

为什么需要这一步
----------------
M8/M9 的两次返工都源于同一个假设：「选题保证了难度，老师就会做目标动作」。
事实是老师会退化成「搜一次、开第一个、买它」。所以采集之前必须先量一次。

**必须读 `raw.jsonl`，不能读 `accepted.jsonl`**
------------------------------------------------
`accepted.jsonl` 只保留 `gold_purchase` 的成功轨迹，而「老师有没有做这个动作」
的证据恰恰在失败轨迹里。读错文件等于把分母换成「已经成功的那些」。

三个场景的目标动作（口径见下，改动前请先读 `docs/lessons-learned.md` §2）
-----------------------------------------------------------------------
* **S1 翻页**：轨迹里出现 `next_page`。这是唯一证据——不能看步数多就猜它翻页了。
* **S2 候选筛选 + 回头买**：打开过 ≥2 个不同 ASIN，且**买到的那个在打开序列里
  出现过两次**（形如 A→B→A 才下单）。
* **S3 超预算排除**：打开过某个候选，其**详情页最低价就超过预算**，最终买的是另一个
  **在预算内**的商品。

⚠️ **历史坑（2026-10-20 修，勿回退）**
--------------------------------------
本脚本原先判定「回头买」用 `purchased != opened[-1]`。这个条件在本环境里
**永远为假**：`buy_now` 不带参数（`environment/actions.py`），只能买**当前页面**
那个商品，所以「买到的」必然等于「最后一次 `open_product` 的」。原写法会让
S2/S3 恒为 0%，然后按「<20% 就地关闭」的规则把两条路误关——
这正是 `docs/lessons-learned.md` §0 里那类误判。回归测试见
`tests/test_analyze_m10_step2_verification.py`。

价格口径
--------
* 详情页 `price:` 可能是**区间**（`5.0 to 18.0`），取**最低价**判「超预算」——
  只有「连最便宜的规格都超」才算排除，避免把「能挑到便宜规格」的商品算成超预算。
* 观察文本优先用 `raw_observation`：`observation` 可能被 token 投影截断成
  `5.0 to…8.0`，直接正则解析会读错。
* 预算来自 `initial_result.instruction`，用 Reward v4 的
  `explicit_budget_from_instruction`（与评测硬门同一口径）。

用法::

    .venv/bin/python scripts/analyze_m10_step2_verification.py \\
        outputs/m10-step2/raw.jsonl \\
        --targets data/collection/m10_step2_targets.jsonl

    # 逐题明细（诊断用）
    ... --detail

    # 机器可读
    ... --json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments" / "ShopSimulator" / "shop_env"

#: 详情页的独立价格行。搜索结果表里的价格在同一条 `rank|asin|price|...` 行内，
#: 且常被投影截断，所以只用于宽松口径（S3b）。
_PRICE_LINE = re.compile(r"^price:\s*(?P<value>.+?)\s*$", re.MULTILINE)
_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")

#: 场景 → (指标键, 分母口径)。`judgeable` 表示只在「可判定」的轨迹上算。
GATE_MAP = {
    "S1_paging": ("used_next_page", "all"),
    "S2_compare": ("bought_earlier_candidate", "all"),
    "S3_budget_exclude": ("excluded_over_budget", "judgeable"),
}

METRIC_KEYS = (
    "used_next_page",
    "bought_earlier_candidate",
    "excluded_over_budget",
    "compared_before_buy",
    "bought_in_budget",
    "success",
)


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


def _budget_parser():
    """惰性导入：只在需要时把 ShopSimulator 加进 `sys.path`，避免顶层污染。"""
    if str(SHOP_ENV) not in sys.path:
        sys.path.append(str(SHOP_ENV))
    from web_agent_site.engine.constraints import (  # noqa: PLC0415
        explicit_budget_from_instruction,
    )

    return explicit_budget_from_instruction


def _numbers(text: str) -> list[float]:
    values: list[float] = []
    for raw in _NUMBER.findall(text):
        try:
            values.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return values


def observation_text(step: dict) -> str:
    """优先取未被投影截断的原文。"""
    return str(step.get("raw_observation") or step.get("observation") or "")


def parse_price_range(text: str) -> tuple[float, float] | None:
    """详情页价格 → (最低, 最高)。区间 `5.0 to 18.0` 取两端。"""
    match = _PRICE_LINE.search(text)
    if not match:
        return None
    values = _numbers(match.group("value"))
    if not values:
        return None
    return min(values), max(values)


def search_rank_low_prices(text: str) -> list[dict]:
    """搜索结果表 → [{rank, asin, low}]。只用于宽松口径，容忍截断。"""
    rows: list[dict] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 3 or not parts[0].isdigit() or len(parts[1]) < 8:
            continue
        values = _numbers(parts[2])
        rows.append(
            {
                "rank": int(parts[0]),
                "asin": parts[1],
                "low": min(values) if values else None,
            }
        )
    return rows


def _instruction_text(row: dict) -> str:
    for container in (row.get("initial_result") or {}, row.get("terminal_result") or {}):
        value = container.get("instruction")
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            text = value.get("instruction")
            if isinstance(text, str) and text.strip():
                return text
    return ""


def _purchase_price(purchase: dict) -> float | None:
    value = purchase.get("price")
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    values = _numbers(str(value))
    return values[0] if values else None


def analyze(row: dict, budget_parser=None) -> dict:
    """把一条轨迹压成判定所需的最小结构。"""
    opened_seq: list[str] = []
    opened_prices: dict[str, tuple[float, float]] = {}
    tools: list[str] = []
    first_search_rows: list[dict] | None = None

    for step in row.get("steps") or []:
        name = step.get("tool_name") or ""
        tools.append(name)
        text = observation_text(step)
        if name == "search_products" and first_search_rows is None:
            first_search_rows = search_rank_low_prices(text)
        elif name == "open_product":
            asin = (step.get("parameters") or {}).get("asin")
            if asin:
                asin = str(asin)
                opened_seq.append(asin)
                price = parse_price_range(text)
                if price and asin not in opened_prices:
                    opened_prices[asin] = price

    terminal = row.get("terminal_result") or {}
    detail = terminal.get("reward_detail") or {}
    goal = terminal.get("goal") or {}
    purchase = terminal.get("purchase") or {}
    purchased = purchase.get("asin")
    purchased = str(purchased) if purchased else None
    purchase_price = _purchase_price(purchase)

    instruction = _instruction_text(row)
    budget = None
    if budget_parser is not None and instruction:
        try:
            budget = budget_parser(instruction)
        except Exception:  # 解析器对畸形文本不该让整轮判定失败
            budget = None

    distinct = sorted(set(opened_seq))
    compared = len(distinct) >= 2
    # 「回头买」：买到的那个在打开序列里出现过两次（A→B→A 才下单）。
    # 因为 buy_now 只能买当前页，purchased 必然 == opened_seq[-1]，
    # 所以证据是「它也出现在 opened_seq[:-1] 里」。
    bought_earlier = bool(
        purchased and compared and purchased in opened_seq[:-1]
    )

    over_budget_opened = [
        asin
        for asin in distinct
        if asin in opened_prices
        and budget is not None
        and opened_prices[asin][0] > budget
        and asin != purchased
    ]
    bought_in_budget = bool(
        purchased is not None
        and purchase_price is not None
        and budget is not None
        and purchase_price <= budget
    )
    s3_judgeable = bool(
        budget is not None and purchased is not None and purchase_price is not None
    )
    excluded_over_budget = bool(over_budget_opened and bought_in_budget)

    # 宽松口径：第 1 名就超预算，但最终买了预算内的（哪怕没打开贵的那个）。
    top_over_budget = bool(
        budget is not None
        and first_search_rows
        and first_search_rows[0]["low"] is not None
        and first_search_rows[0]["low"] > budget
    )
    avoided_top_over_budget = bool(top_over_budget and bought_in_budget)

    return {
        "task_id": row.get("task_id"),
        "attempt_index": row.get("attempt_index"),
        "status": row.get("status"),
        "reward_type": detail.get("reward_type"),
        "gold": str(goal.get("asin")) if goal.get("asin") else None,
        "purchased": purchased,
        "bought_gold": bool(purchased and goal.get("asin") and purchased == str(goal.get("asin"))),
        "success": detail.get("reward_type")
        in {"gold_purchase", "valid_alternative_purchase"},
        "budget": budget,
        "purchase_price": purchase_price,
        "opened_seq": opened_seq,
        "opened_distinct": distinct,
        "opened_count": len(opened_seq),
        "distinct_count": len(distinct),
        "used_next_page": "next_page" in tools,
        "search_count": tools.count("search_products"),
        "step_count": len(tools),
        "compared_before_buy": compared,
        "bought_earlier_candidate": bought_earlier,
        "over_budget_opened": over_budget_opened,
        "bought_in_budget": bought_in_budget,
        "excluded_over_budget": excluded_over_budget,
        "s3_judgeable": s3_judgeable,
        "top_over_budget": top_over_budget,
        "avoided_top_over_budget": avoided_top_over_budget,
    }


def _rate(records: list[dict], key: str) -> dict:
    hit = sum(1 for record in records if record[key])
    total = len(records)
    return {"hit": hit, "total": total, "rate": (hit / total if total else 0.0)}


def summarize(rows: list[dict], tiers: dict[int, str], budget_parser=None) -> dict:
    by_tier: dict[str, list[dict]] = collections.defaultdict(list)
    overall: list[dict] = []
    for row in rows:
        record = analyze(row, budget_parser=budget_parser)
        overall.append(record)
        try:
            task_id = int(record["task_id"])
        except (TypeError, ValueError):
            continue
        by_tier[tiers.get(task_id, "unknown")].append(record)

    report: dict = {"overall": {}, "by_tier": {}, "counts": {}}
    for key in METRIC_KEYS:
        report["overall"][key] = _rate(overall, key)
    report["overall"]["avoided_top_over_budget"] = _rate(overall, "avoided_top_over_budget")
    report["overall"]["s3_judgeable"] = _rate(overall, "s3_judgeable")

    for tier, records in sorted(by_tier.items()):
        entry: dict = {
            "count": len(records),
            "tasks": len({str(r["task_id"]) for r in records}),
            "mean_steps": (
                sum(r["step_count"] for r in records) / len(records) if records else 0.0
            ),
            "mean_distinct_opened": (
                sum(r["distinct_count"] for r in records) / len(records) if records else 0.0
            ),
        }
        for key in METRIC_KEYS:
            entry[key] = _rate(records, key)
        entry["avoided_top_over_budget"] = _rate(records, "avoided_top_over_budget")
        entry["s3_judgeable"] = _rate(records, "s3_judgeable")
        report["by_tier"][tier] = entry

    report["counts"] = {
        "trajectories": len(rows),
        "tasks": len({str(r.get("task_id")) for r in rows}),
        "with_tier": sum(1 for r in overall if tiers),
    }
    return report


def _gate_entry(report: dict, tier: str) -> tuple[float, int, int] | None:
    """返回 (rate, hit, total)。S3 用「可判定」作分母。"""
    entry = report["by_tier"].get(tier)
    if not entry:
        return None
    key, denominator = GATE_MAP[tier]
    if denominator == "judgeable":
        hit = entry[key]["hit"]
        total = entry["s3_judgeable"]["hit"]
        return (hit / total if total else 0.0), hit, total
    stats = entry[key]
    return stats["rate"], stats["hit"], stats["total"]


def render(report: dict, min_rate: float, close_rate: float) -> None:
    counts = report["counts"]
    print(f"轨迹 {counts['trajectories']} 条，覆盖 {counts['tasks']} 道题")
    print("\n整体动作率：")
    labels = {
        "used_next_page": "用过 next_page（S1 目标动作）",
        "bought_earlier_candidate": "回头买前面打开过的候选（S2 目标动作）",
        "excluded_over_budget": "排除超预算候选后买预算内的（S3 目标动作）",
        "compared_before_buy": "买前开过 ≥2 个不同商品（S2 前置条件）",
        "success": "最终买对（gold_purchase / valid_alt）",
    }
    for key, label in labels.items():
        stats = report["overall"][key]
        print(f"  {label}: {stats['hit']}/{stats['total']} = {stats['rate']:.1%}")
    loose = report["overall"]["avoided_top_over_budget"]
    print(f"  〔宽松口径〕第 1 名超预算但买了预算内的: {loose['hit']}/{loose['total']} = {loose['rate']:.1%}")

    if report["by_tier"]:
        print("\n按场景：")
        print("| 场景 | 轨迹 | 题数 | 翻页率 | 回头买率 | 排除超预算率 | 成功率 | 平均步数 | 平均开几个 |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for tier, entry in report["by_tier"].items():
            print(
                f"| {tier} | {entry['count']} | {entry['tasks']} | "
                f"{entry['used_next_page']['rate']:.1%} | "
                f"{entry['bought_earlier_candidate']['rate']:.1%} | "
                f"{entry['excluded_over_budget']['rate']:.1%} | "
                f"{entry['success']['rate']:.1%} | "
                f"{entry['mean_steps']:.1f} | "
                f"{entry['mean_distinct_opened']:.2f} |"
            )
        s3 = report["by_tier"].get("S3_budget_exclude")
        if s3:
            judgeable = s3["s3_judgeable"]
            print(
                f"\nS3 可判定样本（指令有明确预算 + 拿到成交价）："
                f"{judgeable['hit']}/{judgeable['total']}"
            )
            if judgeable["total"] and judgeable["hit"] / judgeable["total"] < 0.8:
                print(
                    "  ⚠ 可判定样本不足 80%，S3 的比例只能当下限看，先别据此关路。"
                )

    print(f"\n闸门 G2 判定（≥{min_rate:.0%} 通过 / {close_rate:.0%}–{min_rate:.0%} 部分有效 / <{close_rate:.0%} 关闭）：")
    for tier in GATE_MAP:
        result = _gate_entry(report, tier)
        if result is None:
            print(f"  {tier}: 本次没有采样")
            continue
        value, hit, total = result
        if value >= min_rate:
            verdict = "**通过** → 进入批量采集"
        elif value >= close_rate:
            verdict = "部分有效 → 提 temperature 多次采样再筛"
        else:
            verdict = "**关闭该场景**（改选题策略后重跑，不要批量采）"
        print(f"  {tier}: {hit}/{total} = {value:.1%} → {verdict}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("trajectories", type=Path, help="必须传 raw.jsonl，不是 accepted.jsonl")
    parser.add_argument("--targets", type=Path, default=None, help="带 tier 的选题清单")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--detail", action="store_true", help="逐题明细")
    parser.add_argument("--min-rate", type=float, default=0.5, help="闸门 G2 的通过线")
    parser.add_argument("--close-rate", type=float, default=0.2, help="低于此值就地关闭该场景")
    args = parser.parse_args(argv)

    if not args.trajectories.is_file():
        raise SystemExit(f"轨迹文件不存在: {args.trajectories}")
    if "accepted" in args.trajectories.name:
        raise SystemExit(
            "读错文件了：必须用 raw.jsonl。accepted.jsonl 只剩成功轨迹，"
            "看不到老师有没有做目标动作。"
        )

    rows = load_jsonl(args.trajectories)
    tiers = load_tiers(args.targets)
    report = summarize(rows, tiers, budget_parser=_budget_parser())
    report["gate"] = {"min_rate": args.min_rate, "close_rate": args.close_rate}

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    render(report, args.min_rate, args.close_rate)

    if args.detail:
        print("\n逐题明细：")
        print("| task_id | 场景 | 步数 | 开几个 | 翻页 | 回头买 | 排除超预算 | 买到gold |")
        print("|---|---|---:|---:|---|---|---|---|")
        for row in rows:
            record = analyze(row, budget_parser=_budget_parser())
            try:
                tier = tiers.get(int(record["task_id"]), "unknown")
            except (TypeError, ValueError):
                tier = "unknown"
            mark = lambda flag: "Y" if flag else "."  # noqa: E731
            print(
                f"| {record['task_id']} | {tier} | {record['step_count']} | "
                f"{record['distinct_count']} | {mark(record['used_next_page'])} | "
                f"{mark(record['bought_earlier_candidate'])} | "
                f"{mark(record['excluded_over_budget'])} | {mark(record['bought_gold'])} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
