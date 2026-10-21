#!/usr/bin/env python3
"""把老师的真实采集轨迹筛成「有目标动作 + 买对」的 SFT 增量。

为什么需要它
------------
`collect_sft_data.py` 的去重规则是「**每题留一条通过验收的轨迹**」，它不看有没有
目标动作。探针实测（2026-10-21）：换词重搜题上老师 90.9% 的轨迹会换词，但同一题
4 次里只要有一次没换词也买对了，那条就可能被留成「代表」——增量里于是混进
「一次搜到就买」的轨迹，正好是我们想纠正的行为。

所以这里按 task 逐题挑：**只在做了目标动作且过验收的轨迹里选**，再每题留一条。

口径与 `scripts/analyze_teacher_action_rate.py` **完全一致**（直接 import 它的
`extract_trajectory`），避免「探针报 24.2%、筛出来的却不是那批」。

用法::

    .venv/bin/python scripts/select_teacher_pattern_batch.py \\
        outputs/m10-requery/raw.jsonl \\
        --scenario requery \\
        --out-dir outputs/m10-requery

产物：``kept.jsonl``（4 键 SFT 行，可直接给 ``run_m10r_build.sh``）、
``selection_report.json``（选了多少、丢了多少、为什么丢、名次画像）。
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts.analyze_teacher_action_rate import (  # noqa: E402
    SCENARIO_ACTION,
    SCENARIO_LABEL,
    extract_trajectory,
)
from scripts.build_m10r_pool import describe_row, rank_bucket  # noqa: E402
from scripts.prepare_sft_curriculum import _validate_row  # noqa: E402
from shopping_grpo.collection.sft import acceptance_reasons, build_sft_row  # noqa: E402

SCENARIOS = tuple(SCENARIO_ACTION)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def task_ids_from(path: Path | None) -> set[int]:
    if path is None:
        return set()
    return {int(row["task_id"]) for row in read_jsonl(path)}


def pick_best(candidates: list[dict]) -> dict:
    """同一题多条合格轨迹时留哪条。

    先看步数（越短越干净：老师走完仪式会带一堆无关步骤），再看 attempt_index
    （小的是先采到的，复现时更稳定），最后用 trajectory_id 兜底保证确定性。
    """
    return min(
        candidates,
        key=lambda row: (
            len(row.get("steps") or []),
            int(row.get("attempt_index") or 0),
            str(row.get("trajectory_id") or ""),
        ),
    )


def select(
    rows: list[dict],
    *,
    scenario: str,
    targets: set[int] | None = None,
    excluded: set[int] | None = None,
    max_steps: int | None = None,
) -> tuple[list[dict], dict]:
    """返回 (选中的原始轨迹, 报告)。"""
    action = SCENARIO_ACTION[scenario]
    targets = targets or set()
    excluded = excluded or set()

    stats = collections.Counter(
        {
            "total": 0,
            "skipped_not_in_targets": 0,
            "skipped_excluded": 0,
            "rejected_not_accepted": 0,
            "rejected_no_target_action": 0,
            "rejected_too_long": 0,
            "tasks_with_candidates": 0,
            "picked": 0,
            "multi_candidate_tasks": 0,
        }
    )
    rejections: list[dict] = []
    eligible: dict[int, list[dict]] = collections.defaultdict(list)

    for row in rows:
        task_id = int(row.get("task_id") or 0)
        if targets and task_id not in targets:
            stats["skipped_not_in_targets"] += 1
            continue
        if task_id in excluded:
            stats["skipped_excluded"] += 1
            continue
        stats["total"] += 1

        summary = extract_trajectory(row)
        if not summary["accepted"]:
            stats["rejected_not_accepted"] += 1
            rejections.append(
                {
                    "task_id": task_id,
                    "attempt_index": summary["attempt_index"],
                    "reason": "not_accepted",
                    "detail": summary["reject_reasons"],
                }
            )
            continue
        if not summary[action]:
            stats["rejected_no_target_action"] += 1
            rejections.append(
                {
                    "task_id": task_id,
                    "attempt_index": summary["attempt_index"],
                    "reason": "no_target_action",
                }
            )
            continue
        steps = len(row.get("steps") or [])
        if max_steps is not None and steps > max_steps:
            stats["rejected_too_long"] += 1
            rejections.append(
                {
                    "task_id": task_id,
                    "attempt_index": summary["attempt_index"],
                    "reason": "too_long",
                    "steps": steps,
                }
            )
            continue
        eligible[task_id].append(row)

    chosen = [pick_best(candidates) for _, candidates in sorted(eligible.items())]
    stats["tasks_with_candidates"] = len(eligible)
    stats["picked"] = len(chosen)
    stats["multi_candidate_tasks"] = sum(
        1 for candidates in eligible.values() if len(candidates) > 1
    )
    report = {
        "scenario": scenario,
        "action": action,
        "counts": dict(sorted(stats.items())),
        "rejections": rejections,
    }
    return chosen, report


def to_sft_rows(trajectories: list[dict]) -> tuple[list[dict], list[dict]]:
    """轨迹 → 4 键 SFT 行；顺便跑一遍课程清单的校验，别把炸点留到训练期。"""
    kept: list[dict] = []
    dropped: list[dict] = []
    for trajectory in trajectories:
        row = build_sft_row(trajectory)
        try:
            _validate_row(row)
        except Exception as exc:  # noqa: BLE001 - 逐条记录，不吞
            dropped.append({"task_id": int(row["task_id"]), "error": str(exc)})
            continue
        kept.append(row)
    return kept, dropped


def profile(rows: list[dict]) -> dict:
    """名次画像：这批增量在「答案在第几名」上是什么样。"""
    buckets = collections.Counter()
    ranks: list[int] = []
    steps: list[int] = []
    for row in rows:
        info = describe_row(row)
        buckets[rank_bucket(info["rank"])] += 1
        if info["rank"] is not None:
            ranks.append(int(info["rank"]))
        steps.append(int(info["steps"]))
    return {
        "rank_buckets": dict(sorted(buckets.items())),
        "rank_1": sum(1 for value in ranks if value == 1),
        "rank_ge2": sum(1 for value in ranks if value >= 2),
        "rank_missing": buckets.get("未出现", 0),
        "mean_steps": round(sum(steps) / len(steps), 2) if steps else 0.0,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, help="采集产物 raw.jsonl")
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--targets",
        type=Path,
        default=None,
        help="只收这份清单里的 task_id（防 raw 里混了别的批次）",
    )
    parser.add_argument(
        "--exclude-pool",
        type=Path,
        default=ROOT / "data" / "sft_pure_v4" / "all.jsonl",
        help="现有 SFT 池：同 task_id 不能进增量（建池脚本会直接报错）",
    )
    parser.add_argument(
        "--exclude-evaluation",
        type=Path,
        default=ROOT / "data" / "evaluation" / "tasks.jsonl",
        help="评测集：一条都不能进",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="超过这么多工具步的不要（默认不限；池子 p90 是 11 步）",
    )
    args = parser.parse_args(argv)

    rows = read_jsonl(args.raw)
    if not rows:
        print(f"{args.raw} 是空的，先跑采集。", file=sys.stderr)
        return 1

    excluded = task_ids_from(args.exclude_pool) | task_ids_from(args.exclude_evaluation)
    chosen, report = select(
        rows,
        scenario=args.scenario,
        targets=task_ids_from(args.targets),
        excluded=excluded,
        max_steps=args.max_steps,
    )
    if not chosen:
        print("一条都没选中。看 report 里的 counts / rejections。", file=sys.stderr)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "selection_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return 1

    kept, dropped = to_sft_rows(chosen)
    write_jsonl(args.out_dir / "kept.jsonl", kept)

    report["schema_dropped"] = dropped
    report["kept"] = len(kept)
    report["profile"] = profile(kept)
    (args.out_dir / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    counts = report["counts"]
    print(f"场景 {SCENARIO_LABEL[args.scenario]}（{args.scenario}）")
    print(
        f"  读到 {counts.get('total', 0)} 条 · 过验收 {counts.get('total', 0) - counts.get('rejected_not_accepted', 0)}"
        f" · 有目标动作 {counts.get('tasks_with_candidates', 0)} 题 · 选中 {len(kept)} 题"
    )
    if dropped:
        print(f"  ⚠️ {len(dropped)} 条过不了课程校验，已丢弃（见 selection_report.json）")
    print(f"  → {args.out_dir / 'kept.jsonl'}")
    print(f"  名次画像：{report['profile']['rank_buckets']}（第 1 名 {report['profile']['rank_1']} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
