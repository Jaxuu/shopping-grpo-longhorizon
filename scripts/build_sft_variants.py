#!/usr/bin/env python3
"""Build the two SFT data variants required by M4-6 (D_full / D_toolonly).

``D_full`` keeps the teacher's assistant natural-language content alongside the
tool calls, so the assistant-turn loss supervises both. ``D_toolonly`` blanks the
assistant ``content`` and keeps only the tool calls, so the same assistant-turn
loss now covers tool-call tokens alone. That is the data-level half of the M5
A/B; the code-level loss-scope switch is M5-2's job.

Both variants share identical ``task_id`` sets and row counts: every assistant
turn in the pool carries a tool call (verified), so blanking ``content`` never
leaves an empty assistant turn and never drops a row. Only ``content`` differs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "data" / "sft_pure_v4"
REPORT_SCHEMA = "shopping-sft-loss-variants-v1"
SFT_ROW_KEYS = {"trajectory_id", "task_id", "messages", "tools"}


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


def sha256_text(path: Path) -> str:
    return hashlib.sha256(
        Path(path).read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()


def strip_assistant_text(messages: list[dict]) -> tuple[list[dict], int, int]:
    """Blank assistant ``content``; return the new messages plus (turns, chars)."""
    stripped_turns = 0
    stripped_chars = 0
    result = copy.deepcopy(messages)
    for message in result:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content:
            stripped_turns += 1
            stripped_chars += len(content)
            message["content"] = ""
    return result, stripped_turns, stripped_chars


def build_variants(rows: list[dict]) -> tuple[list[dict], dict]:
    toolonly: list[dict] = []
    turns = 0
    chars = 0
    for row in rows:
        if set(row) != SFT_ROW_KEYS:
            raise ValueError(f"task {row.get('task_id')}: unexpected row keys {sorted(row)}")
        messages, row_turns, row_chars = strip_assistant_text(row["messages"])
        if not any(m.get("role") == "assistant" for m in messages):
            raise ValueError(f"task {row['task_id']}: no assistant turn after stripping")
        if not any(
            m.get("role") == "assistant" and m.get("tool_calls") for m in messages
        ):
            raise ValueError(f"task {row['task_id']}: no tool call after stripping")
        turns += row_turns
        chars += row_chars
        toolonly.append({**row, "messages": messages})

    stats = {
        "rows": len(rows),
        "toolonly_rows": len(toolonly),
        "rows_with_assistant_text": sum(
            1
            for row in rows
            if any(
                m.get("role") == "assistant" and (m.get("content") or "")
                for m in row["messages"]
            )
        ),
        "assistant_turns_stripped": turns,
        "assistant_chars_stripped": chars,
    }
    if stats["rows"] != stats["toolonly_rows"]:
        raise ValueError("D_full and D_toolonly row counts diverged")
    return toolonly, stats


def portable(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, default=SFT / "all.jsonl")
    parser.add_argument("--toolonly", type=Path, default=SFT / "all_toolonly.jsonl")
    parser.add_argument("--report", type=Path, default=SFT / "variants_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.full)
    toolonly, stats = build_variants(rows)
    write_jsonl(args.toolonly, toolonly)

    report = {
        "schema_version": REPORT_SCHEMA,
        "full": {"path": portable(args.full), "sha256": sha256_text(args.full)},
        "toolonly": {"path": portable(args.toolonly), "sha256": sha256_text(args.toolonly)},
        "counts": stats,
        "note": (
            "D_toolonly blanks assistant content only; task_id sets and row counts are "
            "identical, so the M5 A/B differs solely in the loss scope."
        ),
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print("report:", portable(args.report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
