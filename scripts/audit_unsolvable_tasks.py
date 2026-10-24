#!/usr/bin/env python3
"""哪些评测题**从来没有被任何模型做对过**？它们是真的难，还是根本无解？

为什么要有这个
--------------
M10-H 之后 Final-200 卡在 63.5%，怎么改都上不去。分桶发现一个反常现象：
有 13 道题「买的 ASIN 就是 gold」，但奖励只给了 partial——品牌维度 0.08、
选项维度 0.15。如果商品就是 gold，它的品牌和选项为什么对不上？

两种可能，含义完全不同：

* **模型选错了规格**（商品对、变体错）⇒ 该教「核对选项」，数据能救；
* **这道题本身无解**（gold 商品的品牌文本跟题面要求对不上，或最便宜的变体
  也超预算）⇒ 再怎么训也拿不到，评测上限被压低，**该修评测集而不是训模型**。

单看一次评测分不清这两种。但如果把**历史上所有模型**的轨迹叠起来看：
一道题被 10 个不同模型试过、每个模型都买了同一个 ASIN、每次都卡在同一道硬门上，
那它无解的概率就很高了。

判据（保守，宁可漏报）：
* `never_solved`：所有轨迹里从没出现过 `gold_purchase`；
* 其中 `bought_gold_but_blocked`：**至少一条轨迹买到的 ASIN 等于 goal.asin，
  但奖励仍不是 gold** ⇒ 疑似无解（商品找对了却过不了门）；
* 其余 `never_solved` 只是「没人做到过」，可能只是难。

纯读 jsonl，不需要显卡、不需要起环境服务。

用法::

    .venv/bin/python scripts/audit_unsolvable_tasks.py outputs/evaluation/*/trajectories.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

GOLD = "gold_purchase"
SUCCESS_TYPES = {"gold_purchase", "valid_alternative_purchase"}
# 这三个维度只看**商品本身**（品牌/型号/核心功能），选什么变体都改不了。
# 它们失分却还买到了 gold，说明题面的要求和 gold 商品对不上 ⇒ 谁也拿不到满分。
PRODUCT_LEVEL_DIMS = ("brand", "model", "core_functions")
# 这个维度只看**选中的变体**，换一个变体就能补回来 ⇒ 是策略问题，不是题的问题。
SELECTION_DIMS = ("key_options",)


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract(row: Mapping[str, Any]) -> dict:
    terminal = row.get("terminal_result") or {}
    detail = terminal.get("reward_detail") or {}
    gates = detail.get("hard_gates") or {}
    budget = gates.get("budget") if isinstance(gates, dict) else None
    scores = detail.get("dimension_scores")
    scores = (
        {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}
        if isinstance(scores, dict)
        else {}
    )
    return {
        "task_id": str(row.get("task_id")),
        "reward_type": str(detail.get("reward_type") or "unknown"),
        "gold": (terminal.get("goal") or {}).get("asin"),
        "purchased": (terminal.get("purchase") or {}).get("asin"),
        "budget_status": str((budget or {}).get("status") or "unknown"),
        "budget_required": (budget or {}).get("required"),
        "budget_actual": (budget or {}).get("actual"),
        "target_asin_match": bool(detail.get("target_asin_match")),
        "dims": scores,
    }


def collect(paths: Sequence[Path]) -> tuple[dict[str, list[dict]], int]:
    """按 task_id 汇总所有评测目录的轨迹。"""
    per_task: dict[str, list[dict]] = defaultdict(list)
    used = 0
    for path in paths:
        label = path.parent.name or path.name
        for row in load_rows(path):
            record = extract(row)
            record["model"] = label
            per_task[record["task_id"]].append(record)
        used += 1
    return per_task, used


def _product_level_miss(record: Mapping[str, Any]) -> list[str]:
    """商品级维度里没拿满分的那些（选变体救不回来）。"""
    dims = record.get("dims") or {}
    return [name for name in PRODUCT_LEVEL_DIMS if dims.get(name, 1.0) < 1.0]


def classify(records: Sequence[Mapping[str, Any]]) -> dict:
    """给一道题定性。

    买到 gold 的轨迹要再分三种，因为处方完全不同：
    * 商品级维度失分 ⇒ 题面与 gold 商品对不上，**疑似无解**；
    * 只有 key_options 失分 ⇒ 变体选错，**可救**；
    * 预算 fail ⇒ 可能最便宜变体也超预算（也可能只是没挑便宜的），单列待查。
    """
    solved = [r for r in records if r["reward_type"] == GOLD]
    bought_gold = [r for r in records if r["purchased"] and r["purchased"] == r["gold"]]
    blocked = [r for r in bought_gold if r["reward_type"] not in SUCCESS_TYPES]
    mismatch = [r for r in blocked if _product_level_miss(r)]
    budget_fail = [r for r in bought_gold if r["budget_status"] == "fail"]
    variant_only = [
        r for r in blocked if r not in mismatch and r["budget_status"] != "fail"
    ]

    if solved:
        status = "solved"
    elif mismatch:
        status = "product_mismatch"
    elif budget_fail:
        status = "gold_over_budget"
    elif variant_only:
        status = "variant_only"
    else:
        status = "never_solved"

    miss = Counter()
    for record in blocked:
        for name in _product_level_miss(record):
            miss[name] += 1

    models = {r.get("model") or "?" for r in records}
    reasons = Counter(r["reward_type"] for r in records)
    return {
        "task_id": records[0]["task_id"],
        "status": status,
        "attempts": len(records),
        "models": len(models),
        "models_solved": len({r.get("model") or "?" for r in solved}),
        "bought_gold": len(bought_gold),
        "blocked": len(blocked),
        "budget_fail_on_gold": len(budget_fail),
        "product_miss": dict(miss),
        "reasons": dict(reasons),
        "budget_hint": next(
            (
                f"要求 ≤{r['budget_required']}，实际 {r['budget_actual']}"
                for r in budget_fail
                if r["budget_required"] is not None
            ),
            None,
        ),
    }


STATUS_ORDER = ("solved", "product_mismatch", "gold_over_budget", "variant_only", "never_solved")
STATUS_LABEL = {
    "solved": "至少被一个模型做对过",
    "product_mismatch": "**买到 gold 但商品级维度对不上**（疑似无解）",
    "gold_over_budget": "买到 gold 但超预算（可能最便宜变体也超）",
    "variant_only": "买到 gold 只是变体选错（可救）",
    "never_solved": "从没买过 gold（只是难）",
}


def summarize(items: Sequence[Mapping[str, Any]], dirs: int) -> dict:
    total = len(items)
    status = Counter(item["status"] for item in items)
    groups = {key: [] for key in STATUS_ORDER}
    for item in items:
        groups[item["status"]].append(item)
    # 被最多模型试过却仍无解的，嫌疑最大。
    for key in ("product_mismatch", "gold_over_budget", "variant_only", "never_solved"):
        groups[key].sort(key=lambda item: (-item["models"], item["task_id"]))
    return {
        "dirs": dirs,
        "tasks": total,
        "status": {key: status.get(key, 0) for key in STATUS_ORDER},
        "groups": groups,
    }


def _detail_table(items: Sequence[Mapping[str, Any]], limit: int = 30) -> list[str]:
    lines = [
        "| task_id | 试过的模型数 | 买到 gold | 其中超预算 | 商品级失分 | 预算提示 | 主要失败原因 |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for item in items[:limit]:
        top = ", ".join(
            f"{name}×{count}"
            for name, count in sorted(item["reasons"].items(), key=lambda kv: -kv[1])[:3]
        )
        miss = ", ".join(
            f"{name}×{count}" for name, count in sorted(item["product_miss"].items())
        ) or "—"
        lines.append(
            f"| {item['task_id']} | {item['models']} | {item['bought_gold']} | "
            f"{item['budget_fail_on_gold']} | {miss} | {item['budget_hint'] or '—'} | {top} |"
        )
    if len(items) > limit:
        lines.append(f"| … | 省略 {len(items) - limit} 道 | | | | | |")
    return lines


def format_report(summary: Mapping[str, Any]) -> str:
    total = summary["tasks"]
    counts = summary["status"]
    lines = [
        f"# 评测题可解性审计（{summary['dirs']} 个评测目录，{total} 道题）",
        "",
        "| 类别 | 题数 | 占全部 |",
        "|---|---:|---:|",
    ]
    for key in STATUS_ORDER:
        lines.append(f"| {STATUS_LABEL[key]} | {counts[key]} | {counts[key] / total:.1%} |")

    lines += [
        "",
        "> 判据：只有「买到的 ASIN 就是 `goal.asin`，却仍拿不到 gold」才算嫌疑。"
        "再按失分维度分三层——`brand/model/core_functions` 只看商品本身，选什么变体都改不了，"
        "它们失分说明**题面要求和 gold 商品对不上**；`key_options` 只看选中的变体，"
        "换一个变体就能补回来。",
        "",
    ]

    mismatch = summary["groups"]["product_mismatch"]
    if mismatch:
        lines += [
            f"## 1. 疑似无解（{len(mismatch)} 道）：买到 gold，但商品级维度对不上",
            "",
            *_detail_table(mismatch),
            "",
            f"**判读**：这 {len(mismatch)} 道题（{len(mismatch) / total:.1%}）无论策略多好都拿不到 "
            "`gold_purchase`。扣掉它们，评测真实上限是 "
            f"{(total - len(mismatch)) / total:.1%}；"
            f"当前已达成上限的 {counts['solved'] / max(total - len(mismatch), 1):.1%}。",
            "如果这个数不小，**下一步该修评测集，不是训模型**。",
            "",
        ]
    else:
        lines += ["## 1. 疑似无解：无", "", "⇒ 没有「买到 gold 却商品级不匹配」的题。", ""]

    budget = summary["groups"]["gold_over_budget"]
    if budget:
        lines += [
            f"## 2. 买到 gold 但超预算（{len(budget)} 道，待查）",
            "",
            *_detail_table(budget),
            "",
            "> 两种可能：模型挑了贵变体（可救），或最便宜的变体也超预算（无解）。"
            "要区分得查语料里该商品的最低变体价，脚本不猜。",
            "",
        ]

    variant = summary["groups"]["variant_only"]
    if variant:
        lines += [
            f"## 3. 买到 gold 只是变体选错（{len(variant)} 道，可救）",
            "",
            *_detail_table(variant),
            "",
            "> 商品对、只差选项 ⇒ 数据该教「选完规格后逐条核对题面要求」。",
            "",
        ]

    never = summary["groups"]["never_solved"]
    if never:
        lines += [
            f"## 4. 从没买过 gold（{len(never)} 道，只是难）",
            "",
            *_detail_table(never, limit=15),
        ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="审计哪些评测题从没被任何模型做对过")
    parser.add_argument("trajectories", type=Path, nargs="+", help="多个评测轨迹 jsonl")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    per_task, dirs = collect(args.trajectories)
    items = [classify(records) for records in per_task.values()]
    summary = summarize(items, dirs)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(format_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
