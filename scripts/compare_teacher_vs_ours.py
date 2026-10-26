#!/usr/bin/env python3
"""对比老师与自有模型在 Final-200 上的逐题结果。

为什么要有这个脚本：仓库里老师的 139/200 跑在**旧** Final-200 上，与当前
Final-200 Clean 不是同一份卷子，两个百分比不能直接相减。重跑之后光看两个
总数也说明不了问题——必须知道「老师做对而我们做错」的到底是哪几题，以及
我们当时到底做了什么动作（搜了几次、开过几个候选、有没有翻页）。

输出三份东西：
- 控制台摘要；
- `<output-dir>/teacher-vs-ours.json`：完整差集与逐题行为；
- `<output-dir>/teacher-vs-ours.md`：可以直接贴进汇报的表格。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

OPEN_TOOL = "open_product"
SEARCH_TOOL = "search_products"
PAGE_TOOL = "next_page"
BUY_TOOL = "buy_now"


def load_json(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def success_ids(summary: Mapping) -> set[int]:
    """取严格成功的 task_id 集合。

    只认 ``strict_success_task_ids``。不要用 ``purchase_successes``：它把
    ``valid_alternative_purchase`` 也算成功，拿来做对比会凭空多出题目。
    """
    ids = summary.get("strict_success_task_ids")
    if ids is None:
        raise ValueError("summary 缺少 strict_success_task_ids，无法对比")
    return {int(task_id) for task_id in ids}


def load_trajectories(path: Path | None) -> dict[int, dict]:
    """task_id -> 第一条轨迹。

    同一题多次尝试时只保留第一条，避免同题重复计数。
    """
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
            if task_id is None:
                continue
            by_task.setdefault(int(task_id), row)
    return by_task


def behaviour(trajectory: Mapping | None) -> dict:
    """把一条轨迹压成几个能解释差集的数。

    ``opened_products`` 数的是**不同 asin**，不是 open_product 调用次数：
    同一个商品反复打开不算探索。
    """
    if not trajectory:
        return {}
    steps = trajectory.get("steps") or []
    opened = {
        str((step.get("parameters") or {}).get("asin"))
        for step in steps
        if step.get("tool_name") == OPEN_TOOL
    }
    opened.discard("None")
    detail = ((trajectory.get("terminal_result") or {}).get("reward_detail")) or {}
    return {
        "steps": len(steps),
        "searches": sum(1 for step in steps if step.get("tool_name") == SEARCH_TOOL),
        "opened_products": len(opened),
        "paged": sum(1 for step in steps if step.get("tool_name") == PAGE_TOOL),
        "bought": any(step.get("tool_name") == BUY_TOOL for step in steps),
        "reward_type": detail.get("reward_type") or "unknown",
        "done": bool(trajectory.get("done")),
    }


def _mean(rows: Sequence[Mapping], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return sum(values) / len(values) if values else 0.0


def _aggregate(rows: Sequence[Mapping]) -> dict:
    return {
        "tasks": len(rows),
        "mean_steps": _mean(rows, "steps"),
        "mean_searches": _mean(rows, "searches"),
        "mean_opened_products": _mean(rows, "opened_products"),
        "mean_paged": _mean(rows, "paged"),
    }


def compare(
    teacher_summary: Mapping,
    ours_summary: Mapping,
    *,
    teacher_trajectories: Mapping[int, Mapping] | None = None,
    ours_trajectories: Mapping[int, Mapping] | None = None,
    teacher_name: str = "teacher",
    ours_name: str = "ours",
) -> dict:
    """计算差集，并给出两套模型在差集题上的行为均值。"""
    teacher_ok = success_ids(teacher_summary)
    ours_ok = success_ids(ours_summary)
    denominator = int(teacher_summary.get("expected_tasks") or 0)
    ours_denominator = int(ours_summary.get("expected_tasks") or 0)
    if denominator != ours_denominator:
        raise ValueError(
            f"两份 summary 的分母不同（{denominator} vs {ours_denominator}），"
            "不是同一份评测集，不能对比"
        )

    teacher_only = sorted(teacher_ok - ours_ok)
    ours_only = sorted(ours_ok - teacher_ok)
    both = sorted(teacher_ok & ours_ok)

    teacher_traj = teacher_trajectories or {}
    ours_traj = ours_trajectories or {}

    detail_rows = []
    for task_id in teacher_only:
        detail_rows.append(
            {
                "task_id": task_id,
                "teacher": behaviour(teacher_traj.get(task_id)),
                "ours": behaviour(ours_traj.get(task_id)),
            }
        )

    ours_side = [row["ours"] for row in detail_rows if row["ours"]]
    teacher_side = [row["teacher"] for row in detail_rows if row["teacher"]]

    return {
        "expected_tasks": denominator,
        "teacher_name": teacher_name,
        "ours_name": ours_name,
        "teacher_successes": len(teacher_ok),
        "ours_successes": len(ours_ok),
        "teacher_success_rate": len(teacher_ok) / denominator if denominator else 0.0,
        "ours_success_rate": len(ours_ok) / denominator if denominator else 0.0,
        "both_successes": len(both),
        "teacher_only_count": len(teacher_only),
        "ours_only_count": len(ours_only),
        "net_gap": len(teacher_only) - len(ours_only),
        "teacher_only": teacher_only,
        "ours_only": ours_only,
        "both": both,
        "teacher_only_detail": detail_rows,
        "teacher_only_ours_behaviour": _aggregate(ours_side),
        "teacher_only_teacher_behaviour": _aggregate(teacher_side),
        "teacher_only_ours_reward_types": dict(
            sorted(Counter(row.get("reward_type", "unknown") for row in ours_side).items())
        ),
    }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown(report: Mapping) -> str:
    teacher = report["teacher_name"]
    ours = report["ours_name"]
    lines = [
        f"# {teacher} vs {ours}：Final-200 逐题对比",
        "",
        "## 总量",
        "",
        "| 项 | 题数 |",
        "|---|---:|",
        f"| 评测题数 | {report['expected_tasks']} |",
        f"| {teacher} 做对 | {report['teacher_successes']}（{_pct(report['teacher_success_rate'])}） |",
        f"| {ours} 做对 | {report['ours_successes']}（{_pct(report['ours_success_rate'])}） |",
        f"| 两边都对 | {report['both_successes']} |",
        f"| **只有 {teacher} 对（我们的缺口）** | **{report['teacher_only_count']}** |",
        f"| 只有 {ours} 对（我们独有） | {report['ours_only_count']} |",
        f"| 净差 | {report['net_gap']:+d} |",
        "",
        "## 缺口题上的行为对照",
        "",
        f"只统计「{teacher} 对、{ours} 错」的 {report['teacher_only_count']} 题。",
        "",
        f"| 指标 | {teacher} | {ours} |",
        "|---|---:|---:|",
    ]
    t = report["teacher_only_teacher_behaviour"]
    o = report["teacher_only_ours_behaviour"]
    if t["tasks"] and o["tasks"]:
        lines += [
            f"| 平均步数 | {t['mean_steps']:.1f} | {o['mean_steps']:.1f} |",
            f"| 平均搜索次数 | {t['mean_searches']:.1f} | {o['mean_searches']:.1f} |",
            f"| 平均打开不同候选 | {t['mean_opened_products']:.1f} | {o['mean_opened_products']:.1f} |",
            f"| 平均翻页次数 | {t['mean_paged']:.1f} | {o['mean_paged']:.1f} |",
        ]
    else:
        lines.append("| （缺少一方轨迹，只统计了可得的一侧） | | |")

    if report["teacher_only_ours_reward_types"]:
        lines += ["", f"## 我们在这 {report['teacher_only_count']} 题上的终局类型", "", "| 类型 | 题数 |", "|---|---:|"]
        for name, count in report["teacher_only_ours_reward_types"].items():
            lines.append(f"| `{name}` | {count} |")

    lines += [
        "",
        "## 缺口题清单",
        "",
        "| task_id | 老师步数 | 老师开候选 | 我们步数 | 我们开候选 | 我们的终局 |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["teacher_only_detail"]:
        tt = row["teacher"]
        oo = row["ours"]
        lines.append(
            "| {task} | {ts} | {to} | {os} | {oo} | `{rt}` |".format(
                task=row["task_id"],
                ts=tt.get("steps", "-"),
                to=tt.get("opened_products", "-"),
                os=oo.get("steps", "-"),
                oo=oo.get("opened_products", "-"),
                rt=oo.get("reward_type", "-"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="对比老师与自有模型的 Final-200 逐题结果")
    parser.add_argument("--teacher-summary", type=Path, required=True)
    parser.add_argument("--teacher-trajectories", type=Path)
    parser.add_argument("--ours-summary", type=Path, required=True)
    parser.add_argument("--ours-trajectories", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--teacher-name", default="teacher")
    parser.add_argument("--ours-name", default="ours")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    teacher_traj_path = args.teacher_trajectories or args.teacher_summary.parent / "trajectories.jsonl"
    ours_traj_path = args.ours_trajectories or args.ours_summary.parent / "trajectories.jsonl"
    report = compare(
        load_json(args.teacher_summary),
        load_json(args.ours_summary),
        teacher_trajectories=load_trajectories(teacher_traj_path),
        ours_trajectories=load_trajectories(ours_traj_path),
        teacher_name=args.teacher_name,
        ours_name=args.ours_name,
    )
    output_dir = args.output_dir or args.teacher_summary.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "teacher-vs-ours.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(report)
    (output_dir / "teacher-vs-ours.md").write_text(markdown + "\n", encoding="utf-8")

    print(markdown)
    print(f"已写入 {output_dir / 'teacher-vs-ours.json'}")
    print(f"已写入 {output_dir / 'teacher-vs-ours.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
