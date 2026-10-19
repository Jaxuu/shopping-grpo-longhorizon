#!/usr/bin/env python3
"""M8-B：失败的那几次，是「没看到对的货」还是「看到了却没买」？

为什么要有这个
--------------
M8-0 证明了两件事：全错组不是复制品（模型确实在换路试），而「殊途同归」组的分歧
**只发生在搜索词上**——搜的词重合度 0，但翻到的商品、选的规格、买的东西完全一样。
也就是说，奖励函数这条线可以关掉了：路径分打不出信号，因为它不影响结果。

那剩下的唯一问题是**它到底卡在哪一环**。同一道题 4 次采样里只要有一次买对，
我们就能拿到这道题的「正确商品」，于是可以反查另外几次：

* **看到了却没买**（打开过正确商品，最后买了别的）⇒ 瓶颈在**判断/比较**。
  数据该教的是「在几个候选里怎么取舍」，奖励改不动它。
* **根本没看到**（从没打开过正确商品）⇒ 瓶颈在**检索/浏览覆盖**。
  数据该教的是「怎么找到候选」。

这两条对应的数据改动完全相反，所以必须先量，不能猜（P8）。

关键设计：正确商品从轨迹里自己推出来
------------------------------------
`reward_detail` 只落 `target_asin_match`（布尔），**不落正确 ASIN**。有两个来源可以反推：

1. `gold_purchase` 行买到的 ASIN（gold 的定义就是 `product.asin == goal.asin`）；
2. **任何 `target_asin_match == True` 的行**买到的 ASIN——即使它因为超预算被判成
   `wrong_purchase`。

第 2 条不能漏：漏了会把「买对了商品但超支」的整批组跳过，而那恰好是最容易修的一类
（只需教「下单前核对选完规格后的实际价格」）。实测里这类占买错的 26.4%。

四个桶（互斥）：

* `bought_gold_failed_gate` —— 买的就是正确商品，只是没过硬门（多为超预算）
* `saw_gold_bought_other`   —— 打开过正确商品，最后买了别的
* `saw_gold_no_purchase`    —— 打开过正确商品，但没下单
* `never_saw_gold`          —— 从没打开过正确商品

不需要外部数据、不需要起环境服务、不需要显卡。

用法::

    .venv/bin/python scripts/analyze_selection_vs_retrieval.py \
        outputs/probe/m6r-full/trajectories.jsonl \
        --out outputs/probe/m6r-full/selection_vs_retrieval.md

    # 机器可读
    .venv/bin/python scripts/analyze_selection_vs_retrieval.py <file.jsonl> --json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# ── 判据（写死并写进报告，避免事后口径漂移）──────────────────────────────
# 「看到过正确商品却没买」占失败尝试的比例：
#   ≥ HIGH  ⇒ 瓶颈在判断/比较，数据该教「怎么选」
#   ≤ LOW   ⇒ 瓶颈在检索/浏览，数据该教「怎么找」
#   中间     ⇒ 两环都漏，按 reward_type 分层看
SAW_GOLD_SHARE_HIGH = 0.60
SAW_GOLD_SHARE_LOW = 0.30

# 严格成功口径：只有这两类算「做对了」，与 `evaluation/metrics.py` 一致。
SUCCESS_TYPES = frozenset({"gold_purchase", "valid_alternative_purchase"})
GOLD_TYPE = "gold_purchase"
WRONG_TYPE = "wrong_purchase"
PURCHASE_TOOL = "buy_now"
OPEN_TOOL = "open_product"
SEARCH_TOOL = "search_products"
OPTION_TOOL = "select_option"

# 「买到的替代品有多接近」的分档：ms 高 = 差一点，ms 低 = 没抓住需求。
CLOSE_MATCH = 0.70
MID_MATCH = 0.40

# `hard_gates` 在轨迹里有两种序列化形态，解析时必须都认：
#   1) 完整形态（RewardResult.hard_gates）：{"budget": {"status": "fail", "required": ..}}
#   2) 精简形态（process_evidence）：      {"budget": "fail"}
# 只认形态 1 会让形态 2 的门槛状态静默变成「没失败」，把超支失败漏计。
GATE_FAIL = "fail"


# ── 轨迹解析 ────────────────────────────────────────────────────────────
def _detail(row: Mapping[str, Any]) -> Mapping[str, Any]:
    terminal = row.get("terminal_result")
    if isinstance(terminal, Mapping):
        detail = terminal.get("reward_detail")
        if isinstance(detail, Mapping) and detail:
            return detail
    detail = row.get("reward_detail")
    return detail if isinstance(detail, Mapping) else {}


def _steps(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    steps = row.get("steps")
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, Mapping)]


def _tool_name(step: Mapping[str, Any]) -> str:
    return str(step.get("tool_name") or step.get("tool") or "")


def _params(step: Mapping[str, Any]) -> Mapping[str, Any]:
    params = step.get("parameters")
    return params if isinstance(params, Mapping) else {}


def _finite(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _gate_status(gate: object) -> str:
    """把两种形态的门槛统一成 ``pass`` / ``fail`` / ``unverifiable`` / ``""``。"""
    if isinstance(gate, Mapping):
        return str(gate.get("status") or "").strip().lower()
    if isinstance(gate, str):
        return gate.strip().lower()
    return ""


def _gate_price_known(gate: object) -> bool:
    """只有完整形态才带 required/actual；精简形态无法反推超支幅度。"""
    if not isinstance(gate, Mapping):
        return False
    return gate.get("required") is not None and gate.get("actual") is not None


def extract(row: Mapping[str, Any]) -> dict[str, Any]:
    """抽出一条轨迹用于本诊断的字段。"""
    detail = _detail(row)
    gates = detail.get("hard_gates")
    gates = gates if isinstance(gates, Mapping) else {}
    budget_gate = gates.get("budget")
    category_gate = gates.get("category")
    budget_status = _gate_status(budget_gate)
    category_status = _gate_status(category_gate)
    opened: list[str] = []
    purchased: str | None = None
    searches = 0
    selects = 0
    for step in _steps(row):
        name = _tool_name(step)
        if name == OPEN_TOOL:
            asin = _params(step).get("asin")
            if isinstance(asin, str) and asin:
                opened.append(asin)
        elif name == SEARCH_TOOL:
            searches += 1
        elif name == OPTION_TOOL:
            selects += 1
        elif name == PURCHASE_TOOL and purchased is None and opened:
            purchased = opened[-1]
    return {
        "task_id": str(row.get("task_id")),
        "reward_type": str(detail.get("reward_type", "unknown")),
        "match_score": _finite(detail.get("weighted_score")),
        "target_asin_match": bool(detail.get("target_asin_match")),
        "budget_status": budget_status,
        "category_status": category_status,
        "budget_failed": budget_status == GATE_FAIL,
        "category_failed": category_status == GATE_FAIL,
        "budget_price_known": _gate_price_known(budget_gate),
        "opened": opened,
        "opened_unique": len(set(opened)),
        "searches": searches,
        "selects": selects,
        "steps": len(_steps(row)),
        "purchased_asin": purchased,
    }


def is_success(reward_type: str) -> bool:
    return reward_type in SUCCESS_TYPES


# ── 组内诊断 ────────────────────────────────────────────────────────────
def diagnose_group(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """对一个同题组做「看到没买 / 根本没看到」的判定。

    正确 ASIN 有两个来源，缺一不可：

    * `gold_purchase` 行买到的 ASIN；
    * 任何 `target_asin_match == True` 的行买到的 ASIN——**即使它因为超预算被判
      `wrong_purchase`**。漏掉这个来源会把「买对了商品但超支」的组整批跳过，
      而那恰好是最容易修的一类（只需教「下单前核对实际价格」）。

    返回 ``None`` 表示该组无法诊断（没有任何一条买到过正确商品）。
    """
    evidence = [
        r for r in records
        if r["purchased_asin"]
        and (r["reward_type"] == GOLD_TYPE or r["target_asin_match"])
    ]
    if not evidence:
        return None
    gold_asins = {r["purchased_asin"] for r in evidence}
    if len(gold_asins) > 1:
        # 同一题出现两个不同 gold ASIN ⇒ 数据或口径有问题，不猜，直接标记。
        return {
            "task_id": records[0]["task_id"],
            "status": "ambiguous_gold",
            "gold_asins": sorted(gold_asins),
            "failures": 0,
            "saw_gold": 0,
            "never_saw": 0,
            "saw_gold_bought_other": 0,
            "saw_gold_no_purchase": 0,
            "bought_gold_failed_gate": 0,
            "detail": [],
        }
    gold_asin = next(iter(gold_asins))
    detail: list[dict[str, Any]] = []
    for record in records:
        if is_success(record["reward_type"]):
            continue
        saw = gold_asin in record["opened"]
        bought_gold = record["purchased_asin"] == gold_asin
        if bought_gold:
            bucket = "bought_gold_failed_gate"
        elif saw and record["purchased_asin"]:
            bucket = "saw_gold_bought_other"
        elif saw:
            bucket = "saw_gold_no_purchase"
        else:
            bucket = "never_saw_gold"
        detail.append(
            {
                "reward_type": record["reward_type"],
                "bucket": bucket,
                "match_score": record["match_score"],
                "opened_unique": record["opened_unique"],
                "searches": record["searches"],
                "selects": record["selects"],
                "steps": record["steps"],
                "budget_status": record["budget_status"],
                "category_status": record["category_status"],
                "budget_failed": record["budget_failed"],
                "category_failed": record["category_failed"],
                "budget_price_known": record["budget_price_known"],
                # 「买了 gold 却判失败」有两个完全不同的原因，必须分开看：
                #   True  —— 硬门真没过（reward.py 判定树里的 wrong_purchase）
                #   False —— 硬门都过了，只是 match_score / evidence_coverage < 1.0，
                #            被归成 partial_alternative_purchase（0.25 而非 1.0）
                "gate_failed": record["budget_failed"] or record["category_failed"],
            }
        )
    saw_gold = sum(1 for item in detail if item["bucket"].startswith(("saw_gold", "bought_gold")))
    return {
        "task_id": records[0]["task_id"],
        "status": "ok",
        "gold_asin": gold_asin,
        "failures": len(detail),
        # 同组内成功样本的选规格次数，作为「选全规格」的对照基线。
        "success_selects": [
            r["selects"] for r in records if is_success(r["reward_type"])
        ],
        "saw_gold": saw_gold,
        "never_saw": len(detail) - saw_gold,
        "saw_gold_bought_other": sum(
            1 for item in detail if item["bucket"] == "saw_gold_bought_other"
        ),
        "saw_gold_no_purchase": sum(
            1 for item in detail if item["bucket"] == "saw_gold_no_purchase"
        ),
        "bought_gold_failed_gate": sum(
            1 for item in detail if item["bucket"] == "bought_gold_failed_gate"
        ),
        "detail": detail,
    }


def build_groups(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        record = extract(row)
        buckets[record["task_id"]].append(record)
    return dict(buckets)


def collect(
    groups: Mapping[str, Sequence[Mapping[str, Any]]], *, min_group_size: int = 2
) -> dict[str, Any]:
    diagnosed: list[dict[str, Any]] = []
    skipped_no_gold = 0
    ambiguous = 0
    for records in groups.values():
        if len(records) < min_group_size:
            continue
        result = diagnose_group(records)
        if result is None:
            skipped_no_gold += 1
            continue
        if result["status"] == "ambiguous_gold":
            ambiguous += 1
            continue
        if result["failures"]:
            diagnosed.append(result)
    return {
        "groups": len(diagnosed),
        "skipped_no_gold": skipped_no_gold,
        "ambiguous_gold": ambiguous,
        "diagnosed": diagnosed,
    }


# ── 汇总与裁定 ──────────────────────────────────────────────────────────
def _share(count: int, total: int) -> str:
    return f"{count}/{total}（{count / total:.1%}）" if total else f"{count}/0"


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def aggregate(diagnosed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures = sum(item["failures"] for item in diagnosed)
    saw = sum(item["saw_gold"] for item in diagnosed)
    never = sum(item["never_saw"] for item in diagnosed)
    buckets = Counter(
        row["bucket"] for item in diagnosed for row in item["detail"]
    )
    by_type: dict[str, Counter] = defaultdict(Counter)
    budget_by_bucket: dict[str, Counter] = defaultdict(Counter)
    for item in diagnosed:
        for row in item["detail"]:
            by_type[row["reward_type"]][row["bucket"]] += 1
            budget_by_bucket[row["bucket"]]["total"] += 1
            if row["budget_failed"]:
                budget_by_bucket[row["bucket"]]["budget_failed"] += 1
            if row["category_failed"]:
                budget_by_bucket[row["bucket"]]["category_failed"] += 1
            if row["gate_failed"]:
                budget_by_bucket[row["bucket"]]["gate_failed"] += 1
            else:
                budget_by_bucket[row["bucket"]]["no_gate_failed"] += 1

    # §4 只看「看到过正确商品、但买了别的」——买对了商品却超支的属于另一桶。
    substitute_rows = [
        row for item in diagnosed for row in item["detail"]
        if row["bucket"] == "saw_gold_bought_other"
    ]
    never_rows = [
        row for item in diagnosed for row in item["detail"]
        if row["bucket"] == "never_saw_gold"
    ]
    # 「买了 gold 却判失败」拆成两类：硬门真没过 vs 硬门过了但只拿到 partial。
    gold_bought_rows = [
        row for item in diagnosed for row in item["detail"]
        if row["bucket"] == "bought_gold_failed_gate"
    ]
    gold_gate_failed = sum(1 for row in gold_bought_rows if row["gate_failed"])
    gold_short_rows = [row for row in gold_bought_rows if not row["gate_failed"]]
    gold_short_types = Counter(row["reward_type"] for row in gold_short_rows)
    close = sum(1 for row in substitute_rows if row["match_score"] >= CLOSE_MATCH)
    mid = sum(
        1 for row in substitute_rows if MID_MATCH <= row["match_score"] < CLOSE_MATCH
    )
    far = sum(1 for row in substitute_rows if row["match_score"] < MID_MATCH)

    # 「规格是否选全」的直接检验：同组内成功样本 vs 失败样本的 select_option 次数。
    success_selects = [
        value for item in diagnosed for value in item.get("success_selects", [])
    ]
    bucket_selects: dict[str, list[int]] = defaultdict(list)
    for item in diagnosed:
        for row in item["detail"]:
            bucket_selects[row["bucket"]].append(row["selects"])
    return {
        "groups": len(diagnosed),
        "failures": failures,
        "saw_gold": saw,
        "never_saw": never,
        "saw_share": saw / failures if failures else None,
        "buckets": dict(buckets),
        "by_type": {name: dict(counter) for name, counter in by_type.items()},
        "budget_by_bucket": {
            name: dict(counter) for name, counter in budget_by_bucket.items()
        },
        "budget_failed_total": sum(
            1 for item in diagnosed for row in item["detail"] if row["budget_failed"]
        ),
        # 门槛字段的可读性自检：形态 2（精简）读不出 required/actual，超支幅度不可算。
        # 这两项用来告诉读者「超支占比」这个数是全量还是只覆盖了一部分行。
        "gate_readable": sum(
            1 for item in diagnosed for row in item["detail"] if row["budget_status"]
        ),
        "gate_price_known": sum(
            1 for item in diagnosed for row in item["detail"] if row["budget_price_known"]
        ),
        # 不变式：`wrong_purchase` 必须至少有一个硬门 fail（reward.py 判定树）。
        # 若这里 > 0，说明门槛解析仍有漏，超支占比是下限。
        "gate_unexplained": sum(
            1
            for item in diagnosed
            for row in item["detail"]
            if row["reward_type"] == WRONG_TYPE
            and not row["budget_failed"]
            and not row["category_failed"]
        ),
        # 「买对了 gold 却判失败」的原因拆解。
        "gold_bought_total": len(gold_bought_rows),
        "gold_gate_failed": gold_gate_failed,
        "gold_short_of_gold": len(gold_short_rows),
        "gold_short_types": dict(gold_short_types),
        "saw_close": close,
        "saw_mid": mid,
        "saw_far": far,
        "saw_match_median": (
            statistics.median([row["match_score"] for row in substitute_rows])
            if substitute_rows else None
        ),
        "never_opened_median": (
            statistics.median([row["opened_unique"] for row in never_rows])
            if never_rows else None
        ),
        "never_searches_median": (
            statistics.median([row["searches"] for row in never_rows])
            if never_rows else None
        ),
        "saw_opened_median": (
            statistics.median([row["opened_unique"] for row in substitute_rows])
            if substitute_rows else None
        ),
        "single_open_share_never": (
            sum(1 for row in never_rows if row["opened_unique"] <= 1) / len(never_rows)
            if never_rows else None
        ),
        "success_select_median": (
            statistics.median(success_selects) if success_selects else None
        ),
        "success_select_count": len(success_selects),
        "bucket_select_median": {
            name: statistics.median(values) for name, values in bucket_selects.items()
            if values
        },
        "bucket_select_count": {
            name: len(values) for name, values in bucket_selects.items()
        },
    }


def verdict(summary: Mapping[str, Any], *, high: float, low: float) -> dict[str, Any]:
    if not summary["failures"]:
        return {
            "status": "not_measured",
            "reason": "没有可诊断的失败尝试（需要同题组里有轨迹买到过正确商品）",
            "next": "换用含成功样本的探针产物；全错的题推不出正确商品",
        }
    share = float(summary["saw_share"])
    if share >= high:
        return {
            "status": "selection_bound",
            "reason": f"失败尝试里 {share:.1%} 打开过正确商品却没买 ≥ {high:.0%}："
            "瓶颈在判断/比较，不在检索",
            "next": "数据该教「在几个候选里怎么取舍」：同一批候选的正确/错误对比示范，"
            "重点在关键规格与硬约束的核验顺序",
        }
    if share <= low:
        return {
            "status": "retrieval_bound",
            "reason": f"失败尝试里只有 {share:.1%} 打开过正确商品 ≤ {low:.0%}："
            "瓶颈在检索/浏览覆盖",
            "next": "数据该教「怎么找到候选」：搜索词构造、翻页、类目过滤",
        }
    return {
        "status": "mixed",
        "reason": f"失败尝试里 {share:.1%} 看到过正确商品，落在 [{low:.0%}, {high:.0%}]："
        "两环都在漏",
        "next": "按 reward_type 分层看报告 §3：wrong_purchase 偏向哪一侧，就先改哪一侧的数据",
    }


# ── 报告 ────────────────────────────────────────────────────────────────
def render_report(
    summary: Mapping[str, Any],
    judgement: Mapping[str, Any],
    diagnosed: Sequence[Mapping[str, Any]],
    *,
    source: str,
    rows: int,
    groups: int,
    skipped_no_gold: int,
    ambiguous_gold: int,
    examples: int = 12,
) -> str:
    lines: list[str] = []
    lines.append("# 失败归因：没看到 vs 看到了没买（M8-B）")
    lines.append("")
    lines.append(f"> 输入：`{source}`，{rows} 条轨迹 / {groups} 个同题组。")
    lines.append(
        "> 口径：正确商品从组内 `gold_purchase` **或 `target_asin_match=True`** 的轨迹反推"
        "（后者覆盖「买对了商品但超支」）；只诊断能反推出正确商品的组。"
    )
    lines.append(
        f"> 跳过：无法反推正确商品的组 {skipped_no_gold} 个"
        + (f"；gold ASIN 不一致的组 {ambiguous_gold} 个" if ambiguous_gold else "")
        + "。"
    )
    lines.append("")

    lines.append("## 1. 总览")
    lines.append("")
    lines.append("| 观察 | 数值 |")
    lines.append("|---|---|")
    lines.append(f"| 可诊断的组 | {summary['groups']} |")
    lines.append(f"| 其中失败尝试 | {summary['failures']} 条 |")
    lines.append(f"| **看到过正确商品** | {_share(summary['saw_gold'], summary['failures'])} |")
    lines.append(f"| **从没打开过正确商品** | {_share(summary['never_saw'], summary['failures'])} |")
    lines.append("")

    lines.append("## 2. 分桶")
    lines.append("")
    lines.append("| 桶 | 含义 | 条数 | 占比 |")
    lines.append("|---|---|---:|---:|")
    total = summary["failures"]
    bucket_text = {
        "bought_gold_failed_gate": (
            "`bought_gold_failed_gate`",
            "**买的就是正确商品**，只是没过硬门（多为超预算）",
        ),
        "saw_gold_bought_other": ("`saw_gold_bought_other`", "打开过正确商品，最后买了别的"),
        "saw_gold_no_purchase": ("`saw_gold_no_purchase`", "打开过正确商品，但没下单"),
        "never_saw_gold": ("`never_saw_gold`", "从没打开过正确商品"),
    }
    for key in (
        "bought_gold_failed_gate",
        "saw_gold_bought_other",
        "saw_gold_no_purchase",
        "never_saw_gold",
    ):
        count = int(summary["buckets"].get(key, 0))
        label, meaning = bucket_text[key]
        lines.append(f"| {label} | {meaning} | {count} | {_share(count, total)} |")
    lines.append("")

    lines.append("### 2.1 硬门交叉：买了 gold 却判失败，卡在哪一环")
    lines.append("")
    lines.append("| 桶 | 条数 | 硬门真没过 | 其中超预算 | 硬门都过但仍判失败 |")
    lines.append("|---|---:|---:|---:|---:|")
    for key in (
        "bought_gold_failed_gate",
        "saw_gold_bought_other",
        "saw_gold_no_purchase",
        "never_saw_gold",
    ):
        counter = summary["budget_by_bucket"].get(key, {})
        rows_total = counter.get("total", 0)
        if not rows_total:
            continue
        lines.append(
            f"| `{key}` | {rows_total} | {counter.get('gate_failed', 0)} | "
            f"{counter.get('budget_failed', 0)} | {counter.get('no_gate_failed', 0)} |"
        )
    lines.append("")
    lines.append(
        f"全部失败尝试里 **{_share(summary['budget_failed_total'], total)}** 涉及超预算。"
    )
    lines.append("")
    gold_total = int(summary.get("gold_bought_total", 0))
    gold_gate = int(summary.get("gold_gate_failed", 0))
    gold_short = int(summary.get("gold_short_of_gold", 0))
    if gold_total:
        short_types = summary.get("gold_short_types", {})
        type_text = "、".join(
            f"`{name}` {count}" for name, count in
            sorted(short_types.items(), key=lambda kv: -kv[1])
        )
        lines.append(
            f"**买对 gold 却判失败的 {gold_total} 条，原因不是同一个：**"
            f"硬门真没过 {gold_gate} 条（{_share(gold_gate, gold_total)}），"
            f"硬门都过、却因为 `match_score` / `evidence_coverage` 不到 1.0 "
            f"被归成 partial 的 {gold_short} 条（{_share(gold_short, gold_total)}）"
            + (f"，后者类型为 {type_text}。" if type_text else "。")
        )
        lines.append("")

    readable = int(summary.get("gate_readable", 0))
    price_known = int(summary.get("gate_price_known", 0))
    unexplained = int(summary.get("gate_unexplained", 0))
    lines.append(
        f"口径自检：硬门状态可读 **{readable}/{total}** 行"
        f"（{_share(readable, total)}）；其中能算出超支幅度的 "
        f"**{price_known}/{total}** 行（{_share(price_known, total)}）。"
        f"`wrong_purchase` 却两个硬门都没 fail 的异常行 **{unexplained}** 行"
        + ("——为 0，说明门槛解析与判定树一致。" if unexplained == 0
           else "——不为 0，超支占比只能当**下限**读。")
    )
    lines.append("")

    lines.append("## 3. 按失败类型分层")
    lines.append("")
    lines.append("| reward_type | 失败条数 | 看到过 | 没看到 | 看到占比 |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, counter in sorted(
        summary["by_type"].items(), key=lambda kv: -sum(kv[1].values())
    ):
        seen = (
            counter.get("bought_gold_failed_gate", 0)
            + counter.get("saw_gold_bought_other", 0)
            + counter.get("saw_gold_no_purchase", 0)
        )
        missed = counter.get("never_saw_gold", 0)
        total_type = seen + missed
        ratio = f"{seen / total_type:.1%}" if total_type else "—"
        lines.append(f"| `{name}` | {total_type} | {seen} | {missed} | {ratio} |")
    lines.append("")

    lines.append("## 4. 「看到了没买」的那批，买的替代品有多接近")
    lines.append("")
    lines.append("| 替代品的 `weighted_score` | 条数 | 读法 |")
    lines.append("|---|---:|---|")
    lines.append(
        f"| ≥ {CLOSE_MATCH}（差一点） | {summary['saw_close']} | "
        "几乎对，只差关键规格 ⇒ 教「关键规格核验」 |"
    )
    lines.append(
        f"| {MID_MATCH} ~ {CLOSE_MATCH} | {summary['saw_mid']} | "
        "部分对上 ⇒ 教「硬约束优先级」 |"
    )
    lines.append(
        f"| < {MID_MATCH} | {summary['saw_far']} | "
        "差得远 ⇒ 没抓住需求 ⇒ 教「需求解析」 |"
    )
    lines.append("")
    lines.append(
        f"看到过正确商品的失败尝试，替代品 `weighted_score` 中位数 "
        f"**{_fmt(summary['saw_match_median'])}**。"
    )
    lines.append("")

    lines.append("## 5. 「没看到」的那批，探索够不够")
    lines.append("")
    lines.append(
        f"- 打开过的**不同**商品数中位数：**{_fmt(summary['never_opened_median'], 1)}**"
    )
    lines.append(
        f"- 搜索次数中位数：**{_fmt(summary['never_searches_median'], 1)}**"
    )
    single = summary["single_open_share_never"]
    lines.append(
        f"- 只打开过 ≤1 个商品的比例：**{'—' if single is None else f'{single:.1%}'}**"
        "（高 ⇒ 根本没在比较，只是搜到就买）"
    )
    lines.append("")

    lines.append("## 5.1 规格选全了吗（超预算的直接检验）")
    lines.append("")
    lines.append(
        "商品价格是**区间**（`price: 14.0 to 37.0`），实际价格取决于选完规格后的 variant。"
        "所以「选规格次数」少于成功样本 ⇒ 价格还停在区间上，只能猜，容易超支。"
    )
    lines.append("")
    lines.append("| 口径 | 样本数 | `select_option` 次数中位 |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| **同组成功样本（基线）** | {summary['success_select_count']} | "
        f"**{_fmt(summary['success_select_median'], 1)}** |"
    )
    for key in (
        "bought_gold_failed_gate",
        "saw_gold_bought_other",
        "saw_gold_no_purchase",
        "never_saw_gold",
    ):
        count = summary["bucket_select_count"].get(key, 0)
        if not count:
            continue
        lines.append(
            f"| `{key}` | {count} | {_fmt(summary['bucket_select_median'].get(key), 1)} |"
        )
    lines.append("")
    base = summary["success_select_median"]
    bought = summary["bucket_select_median"].get("bought_gold_failed_gate")
    if base and bought is not None and bought < base:
        lines.append(
            f"**支持「规格没选全」**：买对商品却超支的那批，选规格次数 {bought:.1f} "
            f"< 成功样本 {base:.1f}。"
        )
    elif base and bought is not None:
        lines.append(
            f"**不支持「规格没选全」**：买对商品却超支的那批，选规格次数 {bought:.1f} "
            f"≥ 成功样本 {base:.1f} ⇒ 它选全了规格、知道实际价格，仍然买了超预算的东西。"
        )
    lines.append("")

    if diagnosed:
        lines.append(f"## 6. 明细（前 {min(examples, len(diagnosed))} 个组）")
        lines.append("")
        lines.append("| task_id | 失败次数 | 看到过 | 没看到 | 替代品 ms |")
        lines.append("|---|---:|---:|---:|---|")
        for item in diagnosed[:examples]:
            scores = ", ".join(
                _fmt(row["match_score"], 2) for row in item["detail"]
            )
            lines.append(
                f"| {item['task_id']} | {item['failures']} | {item['saw_gold']} | "
                f"{item['never_saw']} | {scores} |"
            )
        lines.append("")

    lines.append("## 7. 裁定")
    lines.append("")
    lines.append(f"**{judgement['status']}**：{judgement['reason']}")
    lines.append("")
    lines.append(f"下一步：{judgement['next']}")
    lines.append("")
    return "\n".join(lines)


# ── 入口 ────────────────────────────────────────────────────────────────
def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def build_report(
    path: Path, *, min_group_size: int = 2,
    high: float = SAW_GOLD_SHARE_HIGH, low: float = SAW_GOLD_SHARE_LOW,
) -> dict[str, Any]:
    rows = load_rows(path)
    groups = build_groups(rows)
    collected = collect(groups, min_group_size=min_group_size)
    summary = aggregate(collected["diagnosed"])
    return {
        "source": str(path),
        "rows": len(rows),
        "groups": len(groups),
        "skipped_no_gold": collected["skipped_no_gold"],
        "ambiguous_gold": collected["ambiguous_gold"],
        "summary": summary,
        "verdict": verdict(summary, high=high, low=low),
        "diagnosed": collected["diagnosed"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("trajectories", type=Path, help="探针轨迹 jsonl（同题多采样）")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而非 markdown")
    parser.add_argument("--out", type=Path, default=None, help="把报告写到该路径")
    parser.add_argument(
        "--min-group-size", type=int, default=2, help="至少几条同题轨迹才算一组（默认 2）"
    )
    parser.add_argument(
        "--saw-high", type=float, default=SAW_GOLD_SHARE_HIGH,
        help=f"判「瓶颈在判断」的看到占比下限（默认 {SAW_GOLD_SHARE_HIGH}）",
    )
    parser.add_argument(
        "--saw-low", type=float, default=SAW_GOLD_SHARE_LOW,
        help=f"判「瓶颈在检索」的看到占比上限（默认 {SAW_GOLD_SHARE_LOW}）",
    )
    parser.add_argument("--examples", type=int, default=12, help="明细表最多列几个组")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.trajectories.exists():
        print(f"找不到轨迹文件：{args.trajectories}", file=sys.stderr)
        return 2
    report = build_report(
        args.trajectories,
        min_group_size=args.min_group_size,
        high=args.saw_high,
        low=args.saw_low,
    )
    if args.json:
        payload = json.dumps(
            {
                "source": report["source"],
                "rows": report["rows"],
                "groups": report["groups"],
                "skipped_no_gold": report["skipped_no_gold"],
                "ambiguous_gold": report["ambiguous_gold"],
                "summary": report["summary"],
                "verdict": report["verdict"],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    else:
        payload = render_report(
            report["summary"],
            report["verdict"],
            report["diagnosed"],
            source=report["source"],
            rows=report["rows"],
            groups=report["groups"],
            skipped_no_gold=report["skipped_no_gold"],
            ambiguous_gold=report["ambiguous_gold"],
            examples=args.examples,
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
        print(f"已写入 {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
