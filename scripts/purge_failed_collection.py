#!/usr/bin/env python3
"""Purge infrastructure-failed rows from a collection ``raw.jsonl``.

``completed_task_attempts`` treats *every* row in ``raw.jsonl`` as a completed
attempt, so a run that died half-way (SSH drop, env server crash) bakes its
failures in and a re-run would skip them. This script removes exactly those
rows -- the ones that never reached a terminal environment state -- so the next
run re-attempts them.

Rows that reached a terminal state are **kept** even when the teacher did not
win: those are legitimate attempts, not infrastructure failures.

A ``.bak`` copy of the input is written next to it before anything is modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

VALID_REWARD_VERSIONS = {
    "shopsimulator-reward-v3",
    "shopsimulator-reward-v4",
    "shopsimulator-reward-v5",
}


def is_infrastructure_failed(row: dict) -> tuple[bool, list[str]]:
    """True when the rollout never reached a terminal environment state."""
    reasons: list[str] = []
    terminal = row.get("terminal_result") or {}
    reward = terminal.get("reward_detail") or {}
    if row.get("error"):
        reasons.append("has_error")
    if row.get("release_error"):
        reasons.append("release_error")
    if row.get("status") != "done":
        reasons.append("status_not_done")
    if row.get("done") is not True:
        reasons.append("trajectory_not_done")
    if terminal.get("done") is not True or terminal.get("over") is not True:
        reasons.append("environment_not_done")
    if reward.get("reward_version") not in VALID_REWARD_VERSIONS:
        reasons.append("reward_version_missing")
    return bool(reasons), reasons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("outputs/m2-collection/raw.jsonl"),
        help="collection raw.jsonl (a .bak copy is written next to it)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report counts without rewriting"
    )
    args = parser.parse_args()

    raw = Path(args.raw)
    if not raw.is_file():
        print(f"not found: {raw}", file=sys.stderr)
        return 1

    lines = [line for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    keep: list[str] = []
    dropped = 0
    reason_counts: Counter = Counter()
    accepted_kept = 0
    for line in lines:
        row = json.loads(line)
        failed, reasons = is_infrastructure_failed(row)
        if failed:
            dropped += 1
            reason_counts.update(reasons)
            continue
        keep.append(line)
        if not any(
            step.get("tool_name") == "buy_now"
            or step.get("env_action") == "click[Buy Now]"
            for step in row.get("steps") or []
        ):
            continue

    # accepted count uses the same rule as the collection pipeline
    from shopping_grpo.collection.sft import acceptance_reasons  # noqa: E402

    for line in keep:
        if acceptance_reasons(json.loads(line))[0]:
            accepted_kept += 1

    print(f"输入 {len(lines)} 行 | 保留 {len(keep)} | 剔除 {dropped}")
    print("剔除原因分布:", dict(sorted(reason_counts.items())))
    print(f"保留行中 accepted（严格 gold）: {accepted_kept}")

    if args.dry_run:
        print("(dry-run，未改写文件)")
        return 0

    backup = raw.with_suffix(".jsonl.bak")
    shutil.copy2(raw, backup)
    raw.write_text(
        "".join(line + "\n" for line in keep), encoding="utf-8", newline="\n"
    )
    print(f"已备份原文件到 {backup}")
    print(f"已改写 {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
