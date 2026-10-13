#!/usr/bin/env python3
"""Merge the M2 collection increment into the SFT pool (M4-8).

The M2 increment (``data/collection/accepted.jsonl``) passed the collection
pipeline's *terminal* acceptance (strict ``gold_purchase``), but that check only
looks at the final reward. 41 of the 100 trajectories additionally contain one or
more **action-guard rejections**: the teacher emitted an illegal action (for
example calling ``search_products`` while the current page reports
``搜索功能是否可用: False``), the environment refused to execute it and returned
an observation containing ``未执行`` (not executed).

Those illegal assistant turns are still assistant turns, so the assistant-turn
SFT loss would supervise them.  Because every rejection observation states the
action was **not executed**, the environment state is unchanged and the
``(assistant tool-call, tool observation)`` pair can be deleted without altering
the rest of the execution path.  The repaired trajectory is a legal path that
still terminates in ``gold_purchase``.

This script therefore:

1. reads the increment and repairs guard rejections (delete the illegal pair);
2. attaches the pool's uniform ``tools`` schema and normalizes every row to the
   SFT row schema ``{trajectory_id, task_id, messages, tools}``;
3. drops any row that still fails ``prepare_sft_curriculum._validate_row``;
4. merges into the SFT pool by ``task_id`` (idempotent: increment task_ids
   already present in the pool are replaced, never duplicated);
5. writes the repaired increment and a machine-readable repair report.

It never touches ``data/evaluation/**`` (frozen) or ``data/sft/**`` (legacy).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_sft_curriculum import GUARD_MARKERS, _validate_row  # noqa: E402

SFT = ROOT / "data" / "sft_pure_v4"
COLLECTION = ROOT / "data" / "collection"
SFT_ROW_KEYS = ("trajectory_id", "task_id", "messages", "tools")
NOT_EXECUTED_MARKER = "未执行"
REPORT_SCHEMA = "shopping-m4-increment-merge-v1"


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def write_json(path: Path, payload: dict) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256_text(path: Path) -> str:
    """Hash a text artifact with CRLF folded to LF (repo convention)."""
    payload = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def portable(path: Path) -> str:
    """Repo-relative POSIX path when possible, absolute otherwise."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _guard_reason(content: str) -> str:
    """Extract the parenthesised reason code from a guard-rejection observation."""
    for opener, closer in (("（", "）"), ("(", ")")):
        if opener in content:
            inner = content.split(opener, 1)[1]
            if closer in inner:
                return inner.split(closer, 1)[0].strip() or "unknown"
    return "unknown"


def guard_pairs(messages: list[dict]) -> list[tuple[int | None, int, str, bool]]:
    """Locate every guard rejection.

    Returns ``(assistant_index, tool_index, reason, not_executed)`` tuples, where
    ``assistant_index`` is the assistant message owning the rejected tool call
    (``None`` if the call cannot be traced back).
    """
    owner: dict[str, int] = {}
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if call.get("id"):
                owner[str(call["id"])] = index

    pairs: list[tuple[int | None, int, str, bool]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        content = str(message.get("content") or "")
        folded = content.casefold()
        if not any(marker in folded for marker in GUARD_MARKERS):
            continue
        pairs.append(
            (
                owner.get(str(message.get("tool_call_id"))),
                index,
                _guard_reason(content),
                NOT_EXECUTED_MARKER in content,
            )
        )
    return pairs


def repair_messages(messages: list[dict]) -> tuple[list[dict], list[tuple]]:
    """Delete every ``(illegal assistant turn, rejection observation)`` pair."""
    pairs = guard_pairs(messages)
    if not pairs:
        return list(messages), []
    drop: set[int] = set()
    for assistant_index, tool_index, _, _ in pairs:
        drop.add(tool_index)
        if assistant_index is not None:
            drop.add(assistant_index)
    return [message for index, message in enumerate(messages) if index not in drop], pairs


def normalize_row(row: dict, tools: list[dict]) -> tuple[dict, list[tuple]]:
    """Repair and reduce a collection row to the SFT row schema."""
    messages = row.get("messages") or []
    repaired, pairs = repair_messages(messages)
    normalized = {
        "trajectory_id": str(row["trajectory_id"]),
        "task_id": int(row["task_id"]),
        "messages": repaired,
        "tools": copy.deepcopy(tools),
    }
    return normalized, pairs


def merge_rows(pool_rows: list[dict], increment_rows: list[dict]) -> tuple[list[dict], dict]:
    """Merge by ``task_id``; increment rows replace any pre-existing twin."""
    increment_ids = [int(row["task_id"]) for row in increment_rows]
    if len(set(increment_ids)) != len(increment_ids):
        raise ValueError("increment contains duplicate task_ids")
    increment_set = set(increment_ids)

    kept = [row for row in pool_rows if int(row["task_id"]) not in increment_set]
    replaced = len(pool_rows) - len(kept)
    merged = sorted(kept + increment_rows, key=lambda row: int(row["task_id"]))

    merged_ids = [int(row["task_id"]) for row in merged]
    if len(set(merged_ids)) != len(merged_ids):
        raise ValueError("merge produced duplicate task_ids")
    return merged, {
        "pool_rows": len(pool_rows),
        "replaced_existing": replaced,
        "increment_rows": len(increment_rows),
        "merged_rows": len(merged),
    }


def uniform_tools(pool_rows: list[dict]) -> list[dict]:
    """Return the pool's single ``tools`` schema, refusing an ambiguous pool."""
    signatures = {
        hashlib.sha256(
            json.dumps(row["tools"], ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        for row in pool_rows
    }
    if len(signatures) != 1:
        raise ValueError(f"pool has {len(signatures)} distinct tools schemas; expected 1")
    return pool_rows[0]["tools"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=SFT / "all.jsonl")
    parser.add_argument("--increment", type=Path, default=COLLECTION / "accepted.jsonl")
    parser.add_argument("--output", type=Path, default=SFT / "all.jsonl")
    parser.add_argument(
        "--repaired-out", type=Path, default=COLLECTION / "accepted_repaired.jsonl"
    )
    parser.add_argument(
        "--report", type=Path, default=COLLECTION / "increment_repair_report.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pool_rows = read_jsonl(args.pool)
    increment_rows = read_jsonl(args.increment)
    tools = uniform_tools(pool_rows)

    normalized: list[dict] = []
    detail: list[dict] = []
    reason_histogram: Counter = Counter()
    removed_turns = 0
    repaired_rows = 0
    invalid: list[dict] = []
    not_executed_violations: list[int] = []

    for row in increment_rows:
        candidate, pairs = normalize_row(row, tools)
        if pairs:
            repaired_rows += 1
            removed_turns += len(pairs)
            for _, _, reason, not_executed in pairs:
                reason_histogram[reason] += 1
                if not not_executed:
                    not_executed_violations.append(candidate["task_id"])
            detail.append(
                {
                    "task_id": candidate["task_id"],
                    "removed_pairs": len(pairs),
                    "reasons": sorted({reason for _, _, reason, _ in pairs}),
                }
            )
        try:
            _validate_row(candidate)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            invalid.append({"task_id": candidate["task_id"], "error": str(exc)})
            continue
        normalized.append(candidate)

    merged, merge_stats = merge_rows(pool_rows, normalized)
    write_jsonl(args.output, merged)
    write_jsonl(args.repaired_out, normalized)

    report = {
        "schema_version": REPORT_SCHEMA,
        "source_increment": portable(args.increment),
        "output_pool": portable(args.output),
        "repaired_increment": portable(args.repaired_out),
        "counts": {
            "increment_rows": len(increment_rows),
            "repaired_rows": repaired_rows,
            "removed_guard_pairs": removed_turns,
            "invalid_after_repair": len(invalid),
            "admitted_increment_rows": len(normalized),
            **merge_stats,
        },
        "guard_reason_histogram": dict(sorted(reason_histogram.items())),
        "guard_pairs_without_not_executed_marker": sorted(set(not_executed_violations)),
        "invalid_after_repair": invalid,
        "repaired_detail": detail,
        "sha256": {
            "increment": sha256_text(args.increment),
            "repaired_increment": sha256_text(args.repaired_out),
            "merged_pool": sha256_text(args.output),
        },
    }
    write_json(args.report, report)

    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print("guard reason histogram:", report["guard_reason_histogram"])
    print("report:", portable(args.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
