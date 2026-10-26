#!/usr/bin/env python3
"""把「老师做对、我们做错」的题按病因拆开。

为什么必须拆：老师 147/200、我们 127/200，缺口 38 题。这 38 题看起来是同一件事，
实际上至少四种病，修法互不相同：

- **A 检索没覆盖**：gold 从没出现在我们的搜索结果里 ⇒ 只能改检索/换词重搜，多开候选没用；
- **B 看到没打开**：gold 在结果页里，但我们没点开它 ⇒ 是「标题里没认出来」；
- **C 打开过却买了别的**：点开过 gold 的详情，最后还是买了别家 ⇒ 比完了仍选错；
- **D 同一商品选错规格**：asin 对、购买身份不同 ⇒ 只能改选规格这一步。

另外统计一件事：A 组里老师的 gold 是**第几次搜索**才出现的。如果多数是第 2 次以后，
说明老师的优势就是「第一次没搜到就换词再搜一轮」，这是一个具体可教的动作，
而不是笼统的「多比较」。本脚本只统计，不产出任何评测期决策规则。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ens = _load("eval_ensemble", ROOT / "scripts" / "eval_ensemble.py")
vc = _load("analyze_visible_ceiling", ROOT / "scripts" / "analyze_visible_ceiling.py")

GROUPS = {
    "A": "gold 从未出现在我们的搜索结果",
    "B": "gold 在结果页但我们没打开",
    "C": "打开过 gold 详情却买了别的",
    "D": "买的是同一商品、规格不对",
    "E": "其他（没买 / 无法判定）",
}


def purchased_asin(record: Mapping) -> str | None:
    purchase = (record.get("terminal_result") or {}).get("purchase") or {}
    asin = purchase.get("asin")
    return str(asin) if asin else None


def opened_asins(record: Mapping) -> set[str]:
    return {
        str((step.get("parameters") or {}).get("asin"))
        for step in record.get("steps") or []
        if step.get("tool_name") == "open_product"
    } - {"None"}


def search_queries(record: Mapping) -> list[str]:
    return [
        str((step.get("parameters") or {}).get("query"))
        for step in record.get("steps") or []
        if step.get("tool_name") == "search_products"
    ]


def first_search_hit(record: Mapping, asin: str) -> int | None:
    """第几次搜索（1 起）第一次在结果页看到这个 asin；没看到返回 None。"""
    seen = 0
    for step in record.get("steps") or []:
        if step.get("tool_name") != "search_products":
            continue
        seen += 1
        observation = step.get("observation") or ""
        if any(row_asin == asin for _, row_asin in vc.search_rows(observation)):
            return seen
    return None


def best_rank(record: Mapping, asin: str) -> int | None:
    rank = vc.visible_asins(record).get(asin)
    return int(rank) if rank is not None else None


def classify(ours: Mapping | None, gold: str | None) -> str:
    if ours is None or gold is None:
        return "E"
    if not purchased_asin(ours):
        return "E"
    bought = purchased_asin(ours)
    if bought == gold:
        return "D"
    seen = vc.visible_asins(ours)
    if gold not in seen:
        return "A"
    if gold not in opened_asins(ours):
        return "B"
    return "C"


def analyse_task(task_id: int, ours: Mapping | None, teacher: Mapping | None) -> dict:
    gold = (vc.gold_asin(ours) if ours else None) or (vc.gold_asin(teacher) if teacher else None)
    group = classify(ours, gold)
    teacher_searches = search_queries(teacher) if teacher else []
    ours_searches = search_queries(ours) if ours else []
    return {
        "task_id": task_id,
        "group": group,
        "gold_asin": gold,
        "ours_bought": purchased_asin(ours) if ours else None,
        "ours_best_rank": best_rank(ours, gold) if (ours and gold) else None,
        "ours_opened_gold": bool(ours and gold and gold in opened_asins(ours)),
        "ours_searches": len(ours_searches),
        "ours_steps": len(ours.get("steps") or []) if ours else 0,
        "ours_queries": ours_searches,
        "teacher_searches": len(teacher_searches),
        "teacher_steps": len(teacher.get("steps") or []) if teacher else 0,
        "teacher_hit_at_search": first_search_hit(teacher, gold) if (teacher and gold) else None,
        "teacher_queries": teacher_searches,
    }


def build_report(
    teacher_only: Sequence[int],
    ours_by_task: Mapping[int, Mapping],
    teacher_by_task: Mapping[int, Mapping],
) -> dict:
    rows = [analyse_task(task_id, ours_by_task.get(task_id), teacher_by_task.get(task_id)) for task_id in teacher_only]
    groups = Counter(row["group"] for row in rows)

    # A 组里老师是第几次搜索才把 gold 带进来的：这是「换词重搜」的直接证据。
    a_rows = [row for row in rows if row["group"] == "A"]
    hit_positions = Counter(
        row["teacher_hit_at_search"] for row in a_rows if row["teacher_hit_at_search"]
    )

    def _mean(rows: Sequence[Mapping], key: str) -> float:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return sum(values) / len(values) if values else 0.0

    return {
        "tasks": len(rows),
        "groups": dict(sorted(groups.items())),
        "group_labels": GROUPS,
        "rows": rows,
        "a_teacher_hit_positions": {str(k): v for k, v in sorted(hit_positions.items())},
        "ours": {
            "mean_searches": _mean(rows, "ours_searches"),
            "mean_steps": _mean(rows, "ours_steps"),
            "tasks_with_multi_search": sum(1 for row in rows if row["ours_searches"] >= 2),
        },
        "teacher": {
            "mean_searches": _mean(rows, "teacher_searches"),
            "mean_steps": _mean(rows, "teacher_steps"),
            "tasks_with_multi_search": sum(1 for row in rows if row["teacher_searches"] >= 2),
        },
    }


def render_markdown(report: Mapping) -> str:
    lines = [
        f"# 缺口 {report['tasks']} 题的病因分解",
        "",
        "## 分组",
        "",
        "| 组 | 含义 | 题数 |",
        "|---|---|---:|",
    ]
    for key in ("A", "B", "C", "D", "E"):
        count = report["groups"].get(key, 0)
        if count:
            lines.append(f"| {key} | {GROUPS[key]} | {count} |")

    if report["a_teacher_hit_positions"]:
        lines += [
            "",
            "## A 组：老师是第几次搜索才看到 gold",
            "",
            "| 老师的第几次搜索 | 题数 |",
            "|---:|---:|",
        ]
        for position, count in report["a_teacher_hit_positions"].items():
            lines.append(f"| 第 {position} 次 | {count} |")

    ours = report["ours"]
    teacher = report["teacher"]
    lines += [
        "",
        "## 这 38 题上的行为对照",
        "",
        "| 指标 | 老师 | 我们 |",
        "|---|---:|---:|",
        f"| 平均搜索次数 | {teacher['mean_searches']:.1f} | {ours['mean_searches']:.1f} |",
        f"| 搜过 ≥2 次的题 | {teacher['tasks_with_multi_search']} | {ours['tasks_with_multi_search']} |",
        f"| 平均步数 | {teacher['mean_steps']:.1f} | {ours['mean_steps']:.1f} |",
        "",
        "## 逐题清单",
        "",
        "| task_id | 组 | 我们的名次 | 我们买 | gold | 老师搜索 | 老师第几次命中 |",
        "|---:|:--:|---:|---|---|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {task_id} | {group} | {rank} | {bought} | {gold} | {searches} | {hit} |".format(
                task_id=row["task_id"],
                group=row["group"],
                rank=row["ours_best_rank"] if row["ours_best_rank"] is not None else "-",
                bought=row["ours_bought"] or "-",
                gold=row["gold_asin"] or "-",
                searches=row["teacher_searches"],
                hit=row["teacher_hit_at_search"] or "-",
            )
        )
    lines.append("")
    return "\n".join(lines)


def load_trajectories(path: Path | None) -> dict[int, dict]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    by_task: dict[int, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row.get("task_id")
            if task_id is not None:
                by_task.setdefault(int(task_id), row)
    return by_task


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="拆解「老师对、我们错」的题")
    parser.add_argument("--teacher-summary", type=Path, required=True)
    parser.add_argument("--teacher-trajectories", type=Path)
    parser.add_argument("--ours-summary", type=Path, required=True)
    parser.add_argument("--ours-trajectories", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    teacher_summary = json.loads(args.teacher_summary.read_text(encoding="utf-8"))
    ours_summary = json.loads(args.ours_summary.read_text(encoding="utf-8"))
    teacher_ok = {int(x) for x in teacher_summary["strict_success_task_ids"]}
    ours_ok = {int(x) for x in ours_summary["strict_success_task_ids"]}
    teacher_only = sorted(teacher_ok - ours_ok)

    teacher_by_task = load_trajectories(
        args.teacher_trajectories or args.teacher_summary.parent / "trajectories.jsonl"
    )
    ours_by_task = load_trajectories(
        args.ours_trajectories or args.ours_summary.parent / "trajectories.jsonl"
    )

    report = build_report(teacher_only, ours_by_task, teacher_by_task)
    output_dir = args.output_dir or args.teacher_summary.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "teacher-gap.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(report)
    (output_dir / "teacher-gap.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    print(f"已写入 {output_dir / 'teacher-gap.json'}")
    print(f"已写入 {output_dir / 'teacher-gap.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
