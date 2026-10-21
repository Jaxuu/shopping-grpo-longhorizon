#!/usr/bin/env python3
"""M10-R 探针判定：老师在高温度、多次采样下**会不会**做出目标动作。

要回答的问题
------------
M10 Step 2 的 90 题（每题 1 次、temperature 0.0）给出 6.7% / 6.7% / 3.6%，
按「<20% 就地关闭」把三条路全关了。但那只测了**默认行为**。本脚本判定的是：

    把温度提上去、同一道题多跑几次，老师会不会自己翻页 / 换词重搜 / 多开候选？
    而且——这些轨迹里有几条**真能进训练池**？

两个率，别混（2026-10-21 用户指出的口径错误）
--------------------------------------------
* **动作率**（``hit_rate``）：做了目标动作的轨迹占比。这是**能力证据**。
* **可收率**（``harvest_rate``）：既做了目标动作、又**买对 gold 且过验收**的占比。
  这是**能进池子的量**——SFT 池只收 ``gold_purchase``（``acceptance_reasons``），
  「翻了页但没买对」的轨迹会被直接筛掉，白跑。

**判定用可收率，不用动作率。** 用动作率会把「翻页 30% 但翻完买不对」误判成可采集。

* 可收率 ≥ 20% ⇒ 走真实采集。
* 可收率 < 20% ⇒ 采集补不进配比，才轮到脚本合成。

阈值 20% 是预登记的（`plans/m10-dataset-design.md` 与 M10 Step 2 同一条规则），
不随结果调整。理由：20% 的可收率，在 158 道题 × 4 次的采集量下约能收 120+ 条，
够做示范；低于这个数补不进配比。

三个场景 → 目标动作
------------------
* ``paging``  → 出现过 ``next_page``
* ``requery`` → 出现过 **两个及以上不同** 的 ``search_products`` 查询词
* ``compare`` → 打开过 **两个及以上不同** ASIN

判定必须读 ``raw.jsonl``，不能读 ``accepted.jsonl``
--------------------------------------------------
``accepted.jsonl`` 只留买到 gold 的成功轨迹，而「老师有没有做这个动作」的证据
恰恰大量存在于失败轨迹里。读错文件等于把分母换成「已经成功的那些」。

用法::

    .venv/bin/python scripts/analyze_teacher_action_rate.py \\
        outputs/m10-teacher-probe/raw.jsonl \\
        --targets data/collection/m10_teacher_probe_targets.jsonl

    ... --detail                 # 逐题明细
    ... --json                   # 机器可读
    ... --pool data/sft_pure_v4/all.jsonl   # 附上现有 SFT 池的同口径基线
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.collection.sft import acceptance_reasons  # noqa: E402

#: 预登记阈值：轨迹级命中率低于它就不值得采（见模块 docstring）。
PROBE_THRESHOLD = 0.20

#: 场景 → 目标动作键。键名与 ``_summarize`` 返回的字段一一对应。
SCENARIO_ACTION = {
    "paging": "used_next_page",
    "requery": "requeried",
    "compare": "multi_open",
}

#: 场景的中文名，仅用于打印。
SCENARIO_LABEL = {
    "paging": "翻页",
    "requery": "换词重搜",
    "compare": "对比候选",
}

#: 全量采集时的口径内题数（= 三份清单里可用的题数），用来把样本可收率换算成
#: 「全量大概能收多少条」。与 data/collection/*.jsonl 不一致时测试会报错。
SCENARIO_CANDIDATES = {"paging": 158, "requery": 160, "compare": 200}

SEARCH_TOOL = "search_products"
OPEN_TOOL = "open_product"
PAGE_TOOL = "next_page"
GIVEUP_TOOL = "finish_without_purchase"


def norm(value: str) -> str:
    """查询词归一：折叠空白 + casefold。只用来判「是不是同一个词」。"""
    return " ".join(str(value).split()).casefold()


def _summarize(names, queries, opened) -> dict:
    """三个场景共用的动作统计。

    ``names`` 是工具名序列（按发生顺序），``queries`` / ``opened`` 是**已归一/去重前**
    的原始序列。``think`` 不算动作，从步数里排除——它在环境里不产生新观测。
    """
    distinct_queries = {norm(q) for q in queries if str(q).strip()}
    distinct_opened = {str(a).strip() for a in opened if str(a).strip()}
    actions = [name for name in names if name != "think"]
    return {
        "tool_calls": len(actions),
        "next_page_calls": actions.count(PAGE_TOOL),
        "used_next_page": PAGE_TOOL in actions,
        "search_calls": actions.count(SEARCH_TOOL),
        "distinct_queries": len(distinct_queries),
        "requeried": len(distinct_queries) >= 2,
        "opened_distinct": len(distinct_opened),
        "multi_open": len(distinct_opened) >= 2,
        "gave_up": GIVEUP_TOOL in actions,
    }


def extract_trajectory(row: dict) -> dict:
    """从采集产物（``raw.jsonl`` 一行）抽动作统计。"""
    names: list[str] = []
    queries: list[str] = []
    opened: list[str] = []
    for step in row.get("steps") or []:
        name = str(step.get("tool_name") or "")
        if not name:
            continue
        names.append(name)
        parameters = step.get("parameters") or {}
        if name == SEARCH_TOOL and parameters.get("query"):
            queries.append(str(parameters["query"]))
        if name == OPEN_TOOL and parameters.get("asin"):
            opened.append(str(parameters["asin"]))

    summary = _summarize(names, queries, opened)
    terminal = row.get("terminal_result") or {}
    detail = terminal.get("reward_detail") or {}
    accepted, reasons = acceptance_reasons(row)
    error = row.get("error") or {}
    summary.update(
        {
            "task_id": int(row.get("task_id") or 0),
            "attempt_index": int(row.get("attempt_index") or 0),
            "status": row.get("status"),
            "bought_gold": bool(detail.get("target_asin_match")),
            "reward_type": detail.get("reward_type"),
            "accepted": bool(accepted),
            "reject_reasons": list(reasons),
            "error_type": str(error.get("type") or ""),
            "error_message": str(error.get("message") or ""),
            "error_tail": traceback_tail(error.get("traceback")),
        }
    )
    return summary


def traceback_tail(traceback_text, *, lines: int = 3) -> str:
    """调用栈末尾若干行——异常类型往往看不出原因，最后几行才有信息。"""
    if not traceback_text:
        return ""
    parts = [line.strip() for line in str(traceback_text).splitlines() if line.strip()]
    return "\n".join(parts[-lines:])


def extract_messages(row: dict) -> dict:
    """从 SFT 池（``messages`` + ``tool_calls``）抽同口径的动作统计。

    池子里的行没有 ``reward_detail``，所以只有动作指标、没有「买对没买对」。
    """
    names: list[str] = []
    queries: list[str] = []
    opened: list[str] = []
    for message in row.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            if not name:
                continue
            names.append(name)
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except (TypeError, json.JSONDecodeError):
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            if name == SEARCH_TOOL and arguments.get("query"):
                queries.append(str(arguments["query"]))
            if name == OPEN_TOOL and arguments.get("asin"):
                opened.append(str(arguments["asin"]))
    summary = _summarize(names, queries, opened)
    summary["task_id"] = int(row.get("task_id") or 0)
    return summary


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_scenario(records: list[dict], action: str) -> dict:
    """一个场景的汇总。``records`` 是该场景下所有轨迹（不是题）。"""
    trajectories = len(records)
    hits = sum(1 for record in records if record[action])
    per_task: dict[int, list[dict]] = collections.defaultdict(list)
    for record in records:
        per_task[record["task_id"]].append(record)

    task_total = len(per_task)
    task_hit = sum(
        1 for rows in per_task.values() if any(row[action] for row in rows)
    )
    task_always = sum(
        1 for rows in per_task.values() if all(row[action] for row in rows)
    )
    gold = sum(1 for record in records if record["bought_gold"])
    gold_without_action = sum(
        1 for record in records if record["bought_gold"] and not record[action]
    )
    # 可收 = 做了目标动作 **且** 过验收（= 能进 SFT 池）。只做动作但没买对的会被筛掉。
    # ``accepted`` 缺失时按「不可收」处理——宁可少算，不可把进不了池子的算成可收。
    harvest = sum(1 for record in records if record[action] and record.get("accepted"))
    return {
        "tasks": task_total,
        "trajectories": trajectories,
        "attempts_per_task": rate(trajectories, task_total),
        "hits": hits,
        "hit_rate": rate(hits, trajectories),
        "harvest": harvest,
        "harvest_rate": rate(harvest, trajectories),
        "action_only": hits - harvest,
        "task_hit": task_hit,
        "task_hit_rate": rate(task_hit, task_total),
        "task_always": task_always,
        "gold": gold,
        "gold_rate": rate(gold, trajectories),
        "gold_without_action": gold_without_action,
        "used_next_page": sum(1 for record in records if record["used_next_page"]),
        "requeried": sum(1 for record in records if record["requeried"]),
        "multi_open": sum(1 for record in records if record["multi_open"]),
        "gave_up": sum(1 for record in records if record["gave_up"]),
        "mean_tool_calls": rate(
            sum(record["tool_calls"] for record in records), trajectories
        ),
        "mean_opened": rate(
            sum(record["opened_distinct"] for record in records), trajectories
        ),
        "mean_queries": rate(
            sum(record["distinct_queries"] for record in records), trajectories
        ),
    }


def verdict(hit_rate: float, *, threshold: float = PROBE_THRESHOLD) -> str:
    if hit_rate >= threshold:
        return "可采集"
    return "采不到，才考虑合成"


def projected_harvest(scenario: str, row: dict) -> int:
    """把样本可收率外推到全量题数 × 每题次数，估「采集到底能收多少条」。"""
    candidates = SCENARIO_CANDIDATES.get(scenario)
    if not candidates:
        return 0
    attempts = row.get("attempts_per_task") or 0
    return int(round(row["harvest_rate"] * candidates * attempts))


def render(summaries: dict[str, dict], *, threshold: float = PROBE_THRESHOLD) -> str:
    lines = [
        "| 场景 | 题数 | 轨迹数 | 做了目标动作 | 动作率 | **可收（动作+买对）** | **可收率** | 做了动作但没买对 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for scenario in ("paging", "requery", "compare"):
        if scenario not in summaries:
            continue
        row = summaries[scenario]
        lines.append(
            f"| {SCENARIO_LABEL[scenario]}（{scenario}） "
            f"| {row['tasks']} | {row['trajectories']} "
            f"| {row['hits']} | {row['hit_rate']:.1%} "
            f"| {row['harvest']} | {row['harvest_rate']:.1%} "
            f"| {row['action_only']} |"
        )

    lines.append("")
    lines.append(
        "| 场景 | 买到 gold | 买对但没做动作 | 至少命中一次的题 | 翻页次数 | 换词重搜 | 多开≥2 | 放弃 | 平均步数 | 平均开商品 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for scenario in ("paging", "requery", "compare"):
        if scenario not in summaries:
            continue
        row = summaries[scenario]
        lines.append(
            f"| {SCENARIO_LABEL[scenario]} "
            f"| {row['gold']}（{row['gold_rate']:.1%}） "
            f"| {row['gold_without_action']} "
            f"| {row['task_hit']}/{row['tasks']}（{row['task_hit_rate']:.0%}） "
            f"| {row['used_next_page']} | {row['requeried']} | {row['multi_open']} "
            f"| {row['gave_up']} | {row['mean_tool_calls']:.1f} "
            f"| {row['mean_opened']:.2f} |"
        )

    lines.append("")
    lines.append(
        f"**预登记阈值 {threshold:.0%}，按「可收率」判**（不随结果调整）。"
        "「动作率」只说明老师**会**做，不说明轨迹**能进池子**——"
        "SFT 池只收买对 gold 的轨迹，翻了页但没买对的会被直接筛掉。"
    )
    for scenario in ("paging", "requery", "compare"):
        if scenario not in summaries:
            continue
        row = summaries[scenario]
        projected = projected_harvest(scenario, row)
        lines.append(
            f"- {SCENARIO_LABEL[scenario]}：可收率 {row['harvest_rate']:.1%}"
            f"（动作率 {row['hit_rate']:.1%}）→ "
            f"**{verdict(row['harvest_rate'], threshold=threshold)}**"
            f"；按全量 {SCENARIO_CANDIDATES.get(scenario, 0)} 题 × "
            f"{row['attempts_per_task']:.0f} 次推算，约可收 **{projected} 条**"
        )
    return "\n".join(lines)


#: 异常占比超过它就拒绝给判定：分不清「老师不做」和「根本没跑起来」。
HEALTHY_ERROR_CEILING = 0.5


def _cell(text, *, limit: int = 160) -> str:
    """把任意文本压成一行表格单元。"""
    flat = " ".join(str(text).split())
    if len(flat) > limit:
        flat = flat[: limit - 1] + "…"
    return flat.replace("|", "\\|")


def error_share(records: list[dict]) -> float:
    return rate(sum(1 for record in records if record.get("error_type")), len(records))


def render_health(records: list[dict]) -> str:
    """运行健康度：没跑起来的轨迹必须先修，否则命中率是假 0。"""
    status = collections.Counter(str(record.get("status")) for record in records)
    lines = ["", "### 运行健康度", "", "| status | 条数 |", "|---|---|"]
    for name, count in status.most_common():
        lines.append(f"| {name} | {count} |")

    errors = collections.Counter(
        record["error_type"] for record in records if record.get("error_type")
    )
    if not errors:
        lines.append("")
        lines.append("无异常轨迹。")
        return "\n".join(lines)

    lines += ["", "| 异常类型 | 条数 | 示例 |", "|---|---|---|"]
    for error_type, count in errors.most_common():
        sample = next(
            record for record in records if record.get("error_type") == error_type
        )
        lines.append(
            f"| {error_type} | {count} | {_cell(sample.get('error_message'))} |"
        )
    first = next(record for record in records if record.get("error_type"))
    if first.get("error_tail"):
        lines += ["", "第一条异常的调用栈末尾：", "", "```", first["error_tail"], "```"]

    share = error_share(records)
    if share > HEALTHY_ERROR_CEILING:
        lines += [
            "",
            f"⚠️ **{share:.0%} 的轨迹带异常，超过 {HEALTHY_ERROR_CEILING:.0%} 上限："
            "下面的命中率不可用于判定。先修异常再重跑。**",
        ]
    else:
        lines += ["", f"异常占比 {share:.0%}（上限 {HEALTHY_ERROR_CEILING:.0%}）。"]
    return "\n".join(lines)


def render_detail(records: list[dict], scenarios: dict[int, str]) -> str:
    grouped: dict[tuple[str, int], list[dict]] = collections.defaultdict(list)
    for record in records:
        grouped[(scenarios.get(record["task_id"], "unknown"), record["task_id"])].append(
            record
        )
    lines = ["", "逐题明细（每格 = 命中次数/该题次数）："]
    for (scenario, task_id) in sorted(grouped):
        rows = grouped[(scenario, task_id)]
        total = len(rows)
        page = sum(1 for row in rows if row["used_next_page"])
        requery = sum(1 for row in rows if row["requeried"])
        multi = sum(1 for row in rows if row["multi_open"])
        gold = sum(1 for row in rows if row["bought_gold"])
        lines.append(
            f"- task {task_id} [{scenario}] 翻页 {page}/{total}"
            f" 换词 {requery}/{total} 多开 {multi}/{total} 买对 {gold}/{total}"
        )
    return "\n".join(lines)


def load_scenarios(path: Path | None) -> dict[int, str]:
    if path is None or not Path(path).exists():
        return {}
    mapping = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            scenario = row.get("scenario")
            if scenario:
                mapping[int(row["task_id"])] = str(scenario)
    return mapping


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, help="采集产物 raw.jsonl")
    parser.add_argument(
        "--targets",
        type=Path,
        default=ROOT / "data" / "collection" / "m10_teacher_probe_targets.jsonl",
        help="探针清单（提供 task_id → scenario 映射）",
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=None,
        help="可选：现有 SFT 池 all.jsonl，打印同口径基线",
    )
    parser.add_argument("--detail", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--threshold", type=float, default=PROBE_THRESHOLD)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.raw.exists():
        print(f"没有 {args.raw}，先跑采集。", file=sys.stderr)
        return 1

    scenarios = load_scenarios(args.targets)
    records = []
    unknown = 0
    for row in load_jsonl(args.raw):
        task_id = int(row.get("task_id") or 0)
        scenario = scenarios.get(task_id)
        if scenario is None:
            unknown += 1
            continue
        record = extract_trajectory(row)
        record["scenario"] = scenario
        records.append(record)

    if not records:
        print("raw.jsonl 里没有一道题能对上探针清单的 task_id。", file=sys.stderr)
        return 1

    summaries = {}
    for scenario in ("paging", "requery", "compare"):
        subset = [record for record in records if record["scenario"] == scenario]
        if subset:
            summaries[scenario] = summarize_scenario(subset, SCENARIO_ACTION[scenario])

    pool_baseline = None
    if args.pool is not None and Path(args.pool).exists():
        pool_records = [extract_messages(row) for row in load_jsonl(args.pool)]
        pool_baseline = {
            "trajectories": len(pool_records),
            "next_page": sum(1 for row in pool_records if row["used_next_page"]),
            "requeried": sum(1 for row in pool_records if row["requeried"]),
            "multi_open": sum(1 for row in pool_records if row["multi_open"]),
        }

    if args.as_json:
        payload = {
            "threshold": args.threshold,
            "scenarios": summaries,
            "unknown_task_ids": unknown,
            "error_share": error_share(records),
            "error_types": collections.Counter(
                record["error_type"] for record in records if record.get("error_type")
            ),
            "pool_baseline": pool_baseline,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(render(summaries, threshold=args.threshold))
    print(render_health(records))
    if error_share(records) > HEALTHY_ERROR_CEILING:
        print()
        print(
            "**判定作废**：异常太多，命中率不能解释成「老师不做」。"
            "先跑 `bash scripts/run_m10_teacher_probe.sh api` 看老师 API 到底返回什么。"
        )
        # 退出码 3 = 判定无效（区别于 0 正常、1 输入有问题）。
        # 调用方（run_m10_teacher_probe.sh）靠它决定要不要继续跑全量，
        # 不要用 grep 找 JSON 里的 "error" 子串——那是结构判断，不是文本匹配。
        return 3
    if pool_baseline:
        total = pool_baseline["trajectories"]
        print()
        print(
            f"现有 SFT 池基线（{total} 条）：翻页 {pool_baseline['next_page']}"
            f"（{rate(pool_baseline['next_page'], total):.1%}）"
            f"、换词重搜 {pool_baseline['requeried']}"
            f"（{rate(pool_baseline['requeried'], total):.1%}）"
            f"、多开≥2 {pool_baseline['multi_open']}"
            f"（{rate(pool_baseline['multi_open'], total):.1%}）"
        )
    if unknown:
        print()
        print(f"⚠️ {unknown} 条轨迹的 task_id 不在探针清单里（已跳过）。")
    if args.detail:
        print(render_detail(records, scenarios))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
