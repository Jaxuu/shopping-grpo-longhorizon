#!/usr/bin/env python3
"""21 道「别人做对、champion 做错」的题，到底错在哪一环？

为什么要有这个
--------------
Final-200 里可解 180 道，其中 **148 道被至少一个模型做对过**（并集），
当前 champion（M10-R v2）只做对 **127** 道，差 **21** 道。

这 21 道不是「不会」，是**不稳定**——同一道题别的模型能做对，v2 做不对。
它是 M10 线关闭后**唯一还没被量过**的可攻击面（上限 +21 题 ≈ +10.5pt）。

M10 线的教训是「先选靶子、再量靶子」（台账 C14）。所以这次先量：
21 道按「错在哪一环」归类，量完再决定要不要花显卡。

三类处方完全相反，猜错就等于白跑一轮
------------------------------------
* **检索**：gold 从没进入模型的视野 ⇒ 要么查询词不同（换词），要么页数不够（翻页）；
* **选择**：gold 就在模型眼前（列表里/已打开），它却挑了别的，或者规格选错；
* **门控**：商品和规格都对，栽在预算等硬门。

怎么判（只看轨迹里的事实，不猜）
--------------------------------
评测轨迹自带 `terminal_result.goal.asin`（真值）和每步的 `observation`（渲染出来的
搜索结果页，`rank|asin|...` 一行一个商品）。所以能直接回答三个问题：

1. v2 的轨迹里，gold 的 ASIN **有没有在列表里出现过**？在第几页第几名？
2. 如果出现过却没买 ⇒ 看到了不选（列表取舍）；
3. 如果从没出现 ⇒ 看**做对这道题的那个模型**的轨迹：它是用**第几页**看到的？
   第一页 ⇒ 是查询词差异；第二页及以后 ⇒ 是翻页深度差异。

同一查询在同一环境里检索结果是确定的，所以「v2 没看到、赢家第一页就看到了」
只可能意味着**查询词不同**——这一点可以逐题验证（报告里会把两边的 query 并排列出）。

用法::

    .venv/bin/python scripts/analyze_unstable_gap.py outputs/evaluation/*/trajectories.jsonl

    # 指定 champion 目录（默认自动找名字含 m10r-v2 的那个）
    .venv/bin/python scripts/analyze_unstable_gap.py outputs/evaluation/*/trajectories.jsonl \\
        --target m10r-v2 --json-out outputs/analysis/unstable-gap.json

纯读 jsonl，不需要显卡、不需要起环境服务。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

GOLD = "gold_purchase"
SUCCESS_TYPES = {"gold_purchase", "valid_alternative_purchase"}
MIN_TASKS_FULL_RUN = 100  # 低于这个数的是冒烟/探针目录，不进并集

# 第一层：三类处方（用户要的三分法）
FAMILY = {
    "检索-查询词": "检索",
    "翻页-深度": "检索",
    "选择-列表取舍": "选择",
    "选择-规格": "选择",
    "门控-预算": "门控",
    "门控-其他": "门控",
    "未知": "未知",
}

# 第二层：可教的动作，顺序即报告里的顺序
CATEGORIES = (
    ("选择-规格", "商品选对了，规格（颜色/容量/尺码）选错"),
    ("门控-预算", "商品和规格都对，超预算"),
    ("门控-其他", "商品和规格都对，卡在别的硬门"),
    ("选择-列表取舍", "gold 就在它正在看的那一页列表里，它打开了别的商品"),
    ("翻页-深度", "gold 从没进入它的视野，而做对的模型翻到第 2 页以后才看到"),
    ("检索-查询词", "gold 从没进入它的视野，而做对的模型第一页就看到了（查询词不同）"),
    ("未知", "无法判定（缺 gold / 缺观测 / 与预期矛盾）"),
)

PAGE_RE = re.compile(r"Page (\d+) of (\d+) \(Total results: (\d+); ranks (\d+)-(\d+)\)")
ROW_RE = re.compile(r"^(\d+)\|([0-9A-Za-z]{3,})\|", re.MULTILINE)


# ── 读数据 ──────────────────────────────────────────────────────────────


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_listing(observation: str) -> tuple[int, list[tuple[int, str]]]:
    """从一页搜索结果的 observation 里取出（页码，[(名次, asin), ...]）。"""
    match = PAGE_RE.search(observation)
    page_no = int(match.group(1)) if match else 1
    return page_no, [(int(rank), asin) for rank, asin in ROW_RE.findall(observation)]


def _dims(detail: Mapping[str, Any]) -> dict[str, float]:
    scores = detail.get("dimension_scores")
    if not isinstance(scores, Mapping):
        return {}
    return {str(k): float(v) for k, v in scores.items() if isinstance(v, (int, float))}


def _gate_status(gates: Mapping[str, Any], name: str) -> str:
    gate = gates.get(name) if isinstance(gates, Mapping) else None
    if isinstance(gate, Mapping):
        return str(gate.get("status") or "unknown")
    return "unknown"


def extract(row: Mapping[str, Any]) -> dict:
    """抽出一条轨迹里判定所需的全部事实。"""
    terminal = row.get("terminal_result") or {}
    detail = terminal.get("reward_detail") or {}
    gates = detail.get("hard_gates") or {}
    gold = (terminal.get("goal") or {}).get("asin")
    purchased = (terminal.get("purchase") or {}).get("asin")

    steps = row.get("steps")
    steps = [s for s in steps if isinstance(s, Mapping)] if isinstance(steps, list) else []

    queries: list[str] = []
    opened: list[str] = []
    pages = 0
    current_query: str | None = None
    sightings: list[dict] = []  # gold 在列表里出现过的地方

    for index, step in enumerate(steps):
        name = str(step.get("tool_name") or "")
        params = step.get("parameters") if isinstance(step.get("parameters"), Mapping) else {}
        observation = step.get("observation")
        if name == "search_products":
            query = params.get("query")
            if isinstance(query, str):
                queries.append(query)
                current_query = query
        elif name == "next_page":
            pages += 1
        elif name == "open_product":
            asin = params.get("asin")
            if isinstance(asin, str) and asin:
                opened.append(asin)
        if gold and isinstance(observation, str) and "search_results" in observation:
            page_no, rows = parse_listing(observation)
            for rank, asin in rows:
                if asin == gold:
                    sightings.append(
                        {"step": index, "page": page_no, "rank": rank, "query": current_query}
                    )

    reward_type = str(detail.get("reward_type") or "unknown")
    return {
        "task_id": str(row.get("task_id")),
        "reward_type": reward_type,
        "success": reward_type in SUCCESS_TYPES,
        "gold": gold if isinstance(gold, str) else None,
        "purchased": purchased if isinstance(purchased, str) else None,
        "queries": queries,
        "opened": opened,
        "pages": pages,
        "steps": len(steps),
        "sightings": sightings,
        "dims": _dims(detail),
        "budget_status": _gate_status(gates, "budget"),
        "category_status": _gate_status(gates, "category"),
        "target_asin_match": bool(detail.get("target_asin_match")),
    }


def strict_success_ids(path: Path) -> set[str] | None:
    """官方口径的严格成功集合；没有 summary.json 就返回 None（退回按轨迹数）。"""
    summary = path.parent / "summary.json"
    if not summary.is_file():
        return None
    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    ids = data.get("strict_success_task_ids")
    return {str(item) for item in ids} if isinstance(ids, list) else None


# ── 归类 ────────────────────────────────────────────────────────────────


def _best_sighting(sightings: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """gold 最早出现的那个位置（页优先、名次次之）。"""
    items = list(sightings)
    if not items:
        return None
    return min(items, key=lambda item: (int(item["page"]), int(item["rank"])))


def classify(target: Mapping[str, Any], winners: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """给一道「v2 错、别人对」的题定性，返回（类别，依据）。"""
    gold = target.get("gold")
    if not gold:
        return "未知", "轨迹里没有 gold"
    if target.get("success"):
        return "未知", "与预期矛盾：目标模型其实是做对的"

    if gold and target.get("purchased") == gold:
        dims = target.get("dims") or {}
        if target.get("reward_type") == "partial_alternative_purchase" or (
            dims and dims.get("key_options", 1.0) < 1.0
        ):
            return "选择-规格", f"买到 gold，{target.get('reward_type')}，选项维度失分"
        if target.get("budget_status") == "fail":
            return "门控-预算", "买到 gold 和正确规格，预算判 fail"
        return "门控-其他", f"买到 gold，卡在 {target.get('reward_type')}"

    if gold in set(target.get("opened") or []):
        return "选择-列表取舍", "打开过 gold 却买了别的"

    own = _best_sighting(target.get("sightings") or [])
    if own is not None:
        return (
            "选择-列表取舍",
            f"gold 就在它看的那一页列表里（第 {own['page']} 页第 {own['rank']} 名），它打开了别的商品",
        )

    win = _best_sighting(
        sighting for winner in winners for sighting in (winner.get("sightings") or [])
    )
    if win is not None:
        if int(win["page"]) == 1:
            return (
                "检索-查询词",
                f"做对的模型第一页第 {win['rank']} 名就看到（查询词不同），v2 全程没见到",
            )
        return (
            "翻页-深度",
            f"做对的模型翻到第 {win['page']} 页才看到，v2 只翻了 {target.get('pages')} 次",
        )

    if any(gold in (winner.get("opened") or []) for winner in winners):
        return "检索-查询词", "做对的模型直接打开了 gold（v2 全程没见到）"
    return "未知", "v2 没见到 gold，做对的模型也没留下可见位置"


# ── 汇总 ────────────────────────────────────────────────────────────────


def collect(paths: Sequence[Path]) -> tuple[dict[str, dict[str, dict]], dict[str, set[str]]]:
    """按模型目录收集轨迹，并给出每个模型「官方口径」的成功集合。"""
    per_model: dict[str, dict[str, dict]] = {}
    solved: dict[str, set[str]] = {}
    for path in paths:
        label = path.parent.name or path.name
        if label in per_model:
            label = f"{label}/{path.name}"
        records = {record["task_id"]: record for record in map(extract, load_rows(path))}
        per_model[label] = records
        official = strict_success_ids(path)
        if official is None:
            official = {tid for tid, rec in records.items() if rec["reward_type"] == GOLD}
        solved[label] = official & set(records)
    return per_model, solved


def build_gap(
    per_model: Mapping[str, Mapping[str, Mapping[str, Any]]],
    solved: Mapping[str, set[str]],
    target: str,
    min_tasks: int,
) -> dict:
    full = {name for name in per_model if len(per_model[name]) >= min_tasks}
    union: set[str] = set()
    for name in full:
        union |= solved[name]
    all_union: set[str] = set()
    for name in per_model:
        all_union |= solved[name]

    target_solved = solved[target]
    gap = sorted(union - target_solved, key=lambda tid: int(tid) if tid.isdigit() else 0)

    items = []
    for task_id in gap:
        target_record = per_model[target].get(task_id)
        if target_record is None:
            continue
        winners = [
            {"model": name, **per_model[name][task_id]}
            for name in sorted(full)
            if task_id in solved[name] and task_id in per_model[name]
        ]
        category, reason = classify(target_record, winners)
        items.append(
            {
                "task_id": task_id,
                "category": category,
                "family": FAMILY.get(category, "未知"),
                "reason": reason,
                "target": target_record,
                "winners": winners,
                "n_winners": len(winners),
                "n_attempts": sum(
                    1 for name in full if task_id in per_model[name]
                ),
            }
        )

    return {
        "target": target,
        "dirs_full": sorted(full),
        "dirs_all": sorted(per_model),
        "union": sorted(union),
        "union_all_dirs": sorted(all_union),
        "target_solved": sorted(target_solved),
        "gap": items,
    }


# ── 输出 ────────────────────────────────────────────────────────────────


def _short(text: Any, limit: int = 34) -> str:
    value = str(text or "—")
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _median(values: Sequence[float]) -> str:
    if not values:
        return "—"
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return f"{ordered[middle]:g}"
    return f"{(ordered[middle - 1] + ordered[middle]) / 2:g}"


def _query_cell(queries: Sequence[str], limit: int = 3) -> str:
    if not queries:
        return "—"
    shown = " / ".join(_short(q, 22) for q in queries[:limit])
    return shown + (f" (+{len(queries) - limit})" if len(queries) > limit else "")


def format_report(result: Mapping[str, Any], expect: Mapping[str, int] | None = None) -> str:
    gap = result["gap"]
    counts = Counter(item["category"] for item in gap)
    families = Counter(item["family"] for item in gap)
    total = len(gap)

    lines = [
        "# 「别人做对、champion 做错」的题错在哪一环",
        "",
        f"目标模型：`{result['target']}`（{len(result['target_solved'])} 道做对）",
        "",
        "## 1. 口径校验",
        "",
        "| 项 | 值 | 期望 | 一致？ |",
        "|---|---:|---:|---|",
    ]
    checks = [
        ("参与并集的评测目录", len(result["dirs_full"]), None),
        ("并集（至少一个模型做对）", len(result["union"]), (expect or {}).get("union")),
        ("目标模型做对", len(result["target_solved"]), (expect or {}).get("target")),
        ("差值（本报告分析对象）", total, (expect or {}).get("gap")),
    ]
    for label, value, wanted in checks:
        mark = "—" if wanted is None else ("✅" if value == wanted else f"❌ 期望 {wanted}")
        lines.append(f"| {label} | {value} | {wanted if wanted is not None else '—'} | {mark} |")
    if len(result["union_all_dirs"]) != len(result["union"]):
        extra = len(result["union_all_dirs"]) - len(result["union"])
        lines += [
            "",
            (
                f"> 注：把冒烟/探针目录也算进并集是 {len(result['union_all_dirs'])} 道"
                f"（多出 {extra} 道），本报告按正式评测目录（≥100 题）计算。"
            ),
        ]

    lines += [
        "",
        "## 2. 三分法（用户问的三类）",
        "",
        "| 大类 | 道数 | 占比 |",
        "|---|---:|---:|",
    ]
    for name in ("检索", "选择", "门控", "未知"):
        lines.append(f"| {name} | {families.get(name, 0)} | {families.get(name, 0) / total:.0%} |")

    lines += [
        "",
        "## 3. 细分（可教的动作）",
        "",
        "| 类别 | 道数 | gold 在 v2 眼里的名次（中位） | v2 打开商品数（中位） | 含义 | task_id |",
        "|---|---:|---:|---:|---|---|",
    ]
    for key, label in CATEGORIES:
        group = [item for item in gap if item["category"] == key]
        if not group:
            continue
        ranks = [
            int(sighting["rank"])
            for item in group
            for sighting in [_best_sighting(item["target"].get("sightings") or [])]
            if sighting
        ]
        opened = [len(set(item["target"].get("opened") or [])) for item in group]
        ids = ", ".join(item["task_id"] for item in group)
        lines.append(
            f"| `{key}` | {len(group)} | {_median(ranks)} | {_median(opened)} | {label} | {ids} |"
        )

    lines += [
        "",
        "## 4. 逐题证据",
        "",
        "| task_id | 类别 | v2 买了 | reward_type | gold 在 v2 视野里 | v2 打开数 | v2 查询 | 做对的模型 |",
        "|---|---|---|---|---|---:|---|---:|",
    ]
    for item in sorted(gap, key=lambda x: (x["category"], x["task_id"])):
        target = item["target"]
        sighting = _best_sighting(target.get("sightings") or [])
        where = (
            f"第{sighting['page']}页第{sighting['rank']}名" if sighting else "从没出现"
        )
        gold = target.get("gold")
        if gold and target.get("purchased") == gold:
            bought = "gold"
        else:
            bought = _short(target.get("purchased"), 14) if target.get("purchased") else "没下单"
        lines.append(
            f"| {item['task_id']} | `{item['category']}` | {bought} | "
            f"{target.get('reward_type')} | {where} | "
            f"{len(set(target.get('opened') or []))} | "
            f"{_query_cell(target.get('queries') or [])} | "
            f"{item['n_winners']}/{item['n_attempts']} |"
        )

    # 逐题的「查询词对照」：只在检索类里展开，因为那是唯一需要靠换词才能救的一类
    retrieval = [item for item in gap if item["category"] == "检索-查询词"]
    section = 5
    if retrieval:
        lines += [
            "",
            f"## {section}. 检索类：v2 的查询 vs 做对的模型的查询",
            "",
            (
                "> 同一环境同一查询，检索结果是确定的。所以「v2 没见到、赢家第一页就见到」"
                "只可能是查询词不同。下表逐题给出两边的原始 query，可自行核对。"
            ),
            "",
            "| task_id | v2 的查询 | 做对的模型的查询（首个做对者） | gold 名次 |",
            "|---|---|---|---:|",
        ]
        for item in retrieval:
            winner = item["winners"][0] if item["winners"] else {}
            sighting = _best_sighting(winner.get("sightings") or [])
            lines.append(
                f"| {item['task_id']} | {_query_cell(item['target'].get('queries') or [])} | "
                f"{_query_cell(winner.get('queries') or [])} | "
                f"{sighting['rank'] if sighting else '—'} |"
            )
        section += 1

    lines += ["", f"## {section}. 判读（按交接文档 §4 的预登记表）", ""]
    lines += _verdict(counts, total)
    return "\n".join(lines)


def _verdict(counts: Mapping[str, int], total: int) -> list[str]:
    """把预登记的三条分支逐条对照，不做事后改判。"""
    teachable = {key: count for key, count in counts.items() if key != "未知"}
    lines: list[str] = []
    if not teachable or total == 0:
        return ["- 没有可分析的题。"]
    top_key, top_count = max(teachable.items(), key=lambda pair: pair[1])
    label = dict(CATEGORIES)[top_key]
    causes = sum(1 for count in teachable.values() if count > 0)
    lines.append(
        f"- {total} 道里最大的一类是 **{label}**：{top_count} 道（{top_count / total:.0%}）；"
        f"共出现 {causes} 种不同原因。"
    )
    lines.append("")
    lines.append("| 预登记分支 | 条件 | 实际 | 触发？ |")
    lines.append("|---|---|---|---|")
    branch_a = top_count >= 12
    branch_b = causes >= 5
    branch_c = counts.get("检索-查询词", 0) / total > 0.5 if total else False
    lines.append(
        f"| A 值得跑 SFT | 单一可教动作 ≥12 道 | 最大 {top_count} 道（{top_key}） | "
        f"{'✅ 触发' if branch_a else '❌ 未达'} |"
    )
    lines.append(
        f"| B 停，写报告收尾 | 分散在 5 种以上原因 | {causes} 种 | "
        f"{'✅ 触发' if branch_b else '❌ 未达'} |"
    )
    lines.append(
        f"| C 查询词问题 | 主要是检索-查询词 | {counts.get('检索-查询词', 0)} 道"
        f"（{counts.get('检索-查询词', 0) / total:.0%}） | "
        f"{'✅ 触发' if branch_c else '❌ 未达'} |"
    )
    lines.append("")
    if branch_a and not branch_b:
        lines.append(f"- ⇒ **A 分支**：靶子集中在 `{top_key}`，上限 +{top_count} 题。下一步是写数据方案（要显卡）。")
    elif branch_b:
        lines.append("- ⇒ **B 分支**：原因是散的，不是单一数据缺口 ⇒ 按预登记规则停止训练，转写报告收尾。")
    else:
        lines.append("- ⇒ 未达任何分支门槛 ⇒ 先补量（扩大分析口径）再决定，不直接开训。")

    if top_key == "选择-列表取舍" and branch_a:
        lines.append(
            "- ⚠️ **A 分支若落在「选择-列表取舍」上，开训前必须先回答一个问题**："
            "这类题要的动作是「从结果列表的某一行里认出正确商品」。它**不需要**打开多个候选"
            "（列表行里就有价格/品牌/类目/属性/标题），所以不是已关闭的 M8/M9「多开」路线；"
            "但训练池里 93.5% 的轨迹本来就是「单搜单开」，示范量接近 100%，"
            "而 v2 的 73 道失败里 51 道已经是同一类 ⇒ **加同类数据可能只是重复 D 桶的错误**"
            "（`docs/project-history.md` §5.1：补同类示范已证无效）。"
            "真正要新增的信息量是「同一页里有更符合的候选时怎么挑」，"
            "而不是「再多几条单搜单开」。"
        )
    if counts.get("检索-查询词"):
        lines.append(
            "- ⚠️ 关于「换查询」：交接文档 §4 引用的 2-gram 证据**已在 "
            "`docs/project-history.md` §5.3 撤回**（那是措辞相似度，不是可检索性；"
            "实查索引 18/20 题 gold 都在召回内、中位第 42 名）。所以检索类要按"
            "「查询词是否真的不同」判断，不能按词表覆盖率判断。"
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="21 道不稳定题错在哪一环")
    parser.add_argument("trajectories", type=Path, nargs="+", help="评测轨迹 jsonl（多个）")
    parser.add_argument("--target", default=None, help="champion 的评测目录名（子串匹配）")
    parser.add_argument(
        "--min-tasks", type=int, default=MIN_TASKS_FULL_RUN, help="进并集的最少题数（滤掉冒烟目录）"
    )
    parser.add_argument("--expect-union", type=int, default=None)
    parser.add_argument("--expect-target", type=int, default=None)
    parser.add_argument("--expect-gap", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None, help="把逐题明细写成 json")
    parser.add_argument("--md-out", type=Path, default=None, help="把报告写成 markdown")
    args = parser.parse_args(argv)

    per_model, solved = collect(args.trajectories)
    if not per_model:
        print("没有读到任何轨迹。")
        return 1

    target = args.target
    if target is None:
        candidates = [name for name in per_model if "m10r-v2" in name]
        if len(candidates) != 1:
            print("无法自动确定 champion 目录，请用 --target 指定。可选：")
            for name in sorted(per_model):
                print(f"  - {name}（{len(per_model[name])} 题，做对 {len(solved[name])}）")
            return 2
        target = candidates[0]
    if target not in per_model:
        print(f"找不到目录 {target}。可选：{', '.join(sorted(per_model))}")
        return 2

    expect = {
        "union": args.expect_union,
        "target": args.expect_target,
        "gap": args.expect_gap,
    }
    result = build_gap(per_model, solved, target, args.min_tasks)
    report = format_report(result, expect)
    print(report)

    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(report + "\n", encoding="utf-8")
        print(f"\n（已写入 {args.md_out}）")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"（已写入 {args.json_out}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
