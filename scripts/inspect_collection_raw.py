"""盘点采集 raw.jsonl：已经跑了多少、丢了没有、能不能直接续跑。

背景（2026-10-21/22）：compare 批跑到 469/640 撞 HTTP 402 余额不足，
requery 批跑到 380/480 又断。``raw.jsonl`` 是**账本**，采集脚本按
``(task_id, attempt_index)`` 跳过已完成的尝试，所以断点续跑不重花钱。
但有两种情况会让「已完成的」被误判：

1. **错误行被当成已完成** —— 402/429 那种没跑到终态的行也写了进去，
   续跑时会被跳过，等于永久丢一条。必须先 ``purge_failed_collection.py``。
2. **末行被截断** —— 进程被 kill 时最后一行可能只写了一半，读的时候
   会 JSON 解析失败（分析/筛选脚本已容错跳过，但这里要报出来）。

这个脚本只读不写，纯 CPU、不调模型，用来回答「已经跑的在不在」。

用法::

    .venv/bin/python scripts/inspect_collection_raw.py outputs/m10-requery/raw.jsonl
    .venv/bin/python scripts/inspect_collection_raw.py outputs/m10-requery/raw.jsonl \\
        --targets data/collection/m10_requery_targets.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

# 这些 status 说明轨迹跑到了终态（能进验收判定）。
TERMINAL_STATUSES = {"done", "assistant_final", "max_steps", "invalid_action_limit"}


def read_rows(path: Path) -> tuple[list[dict], list[int]]:
    """返回 (可解析行, 解析失败的行号列表)。容忍末行截断。"""
    rows: list[dict] = []
    broken: list[int] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            broken.append(lineno)
    return rows, broken


def needs_purge(row: dict) -> bool:
    """与 purge_failed_collection.py 同一判定：有 error / release_error 就没到终态。"""
    return bool(row.get("error") or row.get("release_error"))


def inspect(rows: list[dict], targets: list[int] | None = None) -> dict:
    statuses = collections.Counter(str(row.get("status")) for row in rows)
    keys = collections.Counter(
        (row.get("task_id"), row.get("attempt_index")) for row in rows
    )
    per_task = collections.defaultdict(set)
    for row in rows:
        if not needs_purge(row):
            per_task[row.get("task_id")].add(row.get("attempt_index"))

    attempts = [idx for idxs in per_task.values() for idx in idxs]
    report = {
        "rows": len(rows),
        "statuses": dict(sorted(statuses.items(), key=lambda kv: -kv[1])),
        "purgeable": sum(1 for row in rows if needs_purge(row)),
        "terminal": sum(1 for row in rows if str(row.get("status")) in TERMINAL_STATUSES),
        "duplicate_keys": sum(count - 1 for count in keys.values() if count > 1),
        "tasks": len(per_task),
        "attempts_per_task": (len(attempts) / len(per_task)) if per_task else 0.0,
        "max_attempt_index": max(attempts) if attempts else -1,
    }
    if targets:
        target_set = set(targets)
        done = {task for task in per_task if task in target_set}
        report["targets"] = len(target_set)
        report["targets_done"] = len(done)
        report["targets_pending"] = len(target_set) - len(done)
        # 剩余条数按「每题计划次数」估，不用均值——断点时均值会偏小。
        planned = report["max_attempt_index"] + 1
        report["planned_attempts"] = planned
        report["remaining"] = sum(
            max(0, planned - len(per_task[task])) for task in target_set if task in per_task
        ) + report["targets_pending"] * planned
    return report


def task_ids_from(path: Path) -> list[int]:
    ids: list[int] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.append(int(json.loads(line)["task_id"]))
    return ids


def render(path: Path, report: dict, broken: list[int]) -> str:
    size_mb = Path(path).stat().st_size / 1024 / 1024
    lines = [
        f"文件：{path}（{size_mb:.1f} MB）",
        f"可解析 {report['rows']} 行 · 跑完终态 {report['terminal']} 条"
        f" · 需清理 {report['purgeable']} 条",
        f"覆盖 {report['tasks']} 道题 · 每题平均 {report['attempts_per_task']:.2f} 次"
        f" · 最大 attempt_index {report['max_attempt_index']}",
    ]
    if broken:
        lines.append(f"⚠️ {len(broken)} 行 JSON 解析失败（行号 {broken[:5]}）——多半是进程被 kill 时写了半行")
    if report["duplicate_keys"]:
        lines.append(f"⚠️ {report['duplicate_keys']} 个 (task, attempt) 重复（不影响续跑，续跑按集合跳过）")
    if report["purgeable"]:
        lines.append(
            "⚠️ 有需清理的行：**不清理就会被当成已完成而永久跳过**，续跑前先跑 "
            "`scripts/purge_failed_collection.py --raw <本文件>`"
        )
    if "targets" in report:
        lines.append("")
        lines.append(
            f"对照清单：{report['targets']} 题 · 已完成 {report['targets_done']} 题"
            f" · 未开始 {report['targets_pending']} 题"
        )
        lines.append(
            f"还要补 **{report['remaining']} 条**"
            f"（按每题 {report['planned_attempts']} 次算）"
        )
    lines.append("")
    lines.append("status 分布：")
    for status, count in report["statuses"].items():
        lines.append(f"  {status}: {count}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, help="采集产物 raw.jsonl")
    parser.add_argument(
        "--targets",
        type=Path,
        default=None,
        help="本批清单（jsonl，含 task_id），用来算「还剩多少条」",
    )
    args = parser.parse_args()

    if not args.raw.exists():
        print(f"文件不存在：{args.raw}", file=sys.stderr)
        return 2

    rows, broken = read_rows(args.raw)
    if not rows:
        print(f"{args.raw} 里没有可解析的行——这批没采到任何东西。", file=sys.stderr)
        return 1
    targets = task_ids_from(args.targets) if args.targets else None
    report = inspect(rows, targets)
    print(render(args.raw, report, broken))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
