#!/usr/bin/env python3
"""把补好理由的轨迹合回完整 SFT 池，并给出「翻页轨迹带思考率」的前后对照。

为什么需要单独一步：``augment_trajectory_reasoning.py`` 只输出**被选中的**那部分
轨迹（M10-R v3 是 144 条含 next_page 的），它不负责拼池子。而 SFT 池必须是完整的
1,210 条——少了 1,066 条模板轨迹，模型会把「每条都翻页」当成常态。

顺带做两件不能省的事：

1. **验证**：补写后每条翻页轨迹的 assistant 回合都必须有 content。缺一条就意味着
   那一步在 tool-call-only 口径下没有监督信号，等于白补。
2. **打标**（``--tag-loss-scope assistant``）：给这些轨迹的每个 assistant 回合写上
   ``loss_scope``，让训练器只在这里打开完整监督。不加这个标记，理由会被
   tool-call-only 的掩码吃掉——**这是整件事最容易静默失败的地方**。

用法：
    python scripts/merge_reasoned_trajectories.py \
        --source data/sft_m10r/capped/all.jsonl \
        --reasoned data/sft_m10r/capped/all_reasoned_paging.jsonl \
        --output data/sft_m10r/capped/v3/all.jsonl \
        --tag-loss-scope assistant
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ASSISTANT = "assistant"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no} 不是合法 JSON：{exc}") from exc
    return rows


def assistant_turns(row: dict) -> list[dict]:
    return [m for m in row.get("messages") or [] if m.get("role") == ASSISTANT]


def action_turns(row: dict) -> list[dict]:
    """只算带 tool_calls 的 assistant 回合：纯文本回合本来就不产生动作。"""
    return [m for m in assistant_turns(row) if m.get("tool_calls")]


def paging_rows(rows: list[dict]) -> list[dict]:
    selected = []
    for row in rows:
        if any(
            (call.get("function") or {}).get("name") == "next_page"
            for message in action_turns(row)
            for call in message.get("tool_calls") or []
        ):
            selected.append(row)
    return selected


def coverage(rows: list[dict]) -> tuple[int, int]:
    """返回 (有内容的动作回合数, 动作回合总数)。"""
    filled = total = 0
    for row in rows:
        for message in action_turns(row):
            total += 1
            if str(message.get("content") or "").strip():
                filled += 1
    return filled, total


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, required=True, help="完整池（未补思考）")
    parser.add_argument("--reasoned", type=Path, required=True, help="补好理由的子集")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tag-loss-scope",
        choices=("assistant", "tool-call-only"),
        default=None,
        help="给被替换轨迹的每个 assistant 回合写上该口径（推荐 assistant）",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="允许补写后仍有空 content 的回合（默认视为失败，因为那等于白补）",
    )
    args = parser.parse_args()

    source = read_jsonl(args.source)
    reasoned = read_jsonl(args.reasoned)
    if not reasoned:
        raise SystemExit(f"{args.reasoned} 是空的：补思考那一步没产出任何轨迹")

    by_id: dict[str, dict] = {}
    for row in reasoned:
        key = str(row.get("trajectory_id"))
        if key in by_id:
            raise SystemExit(
                f"{args.reasoned} 里 trajectory_id={key} 出现了两次（断点续跑时重复写入了？）"
            )
        by_id[key] = row

    source_ids = {str(row.get("trajectory_id")) for row in source}
    unknown = sorted(set(by_id) - source_ids)
    if unknown:
        raise SystemExit(
            f"补写结果里有 {len(unknown)} 条不在源池里，拒绝合并：{unknown[:5]}"
        )

    # 只认源池里真的有 next_page 的那些，防止 --paging-only 忘了加导致整池被覆盖。
    expected_ids = {str(row.get("trajectory_id")) for row in paging_rows(source)}
    extra = sorted(set(by_id) - expected_ids)
    if extra:
        raise SystemExit(
            f"补写结果里有 {len(extra)} 条不含 next_page，与 --paging-only 的口径不符：{extra[:5]}"
        )

    before_filled, before_total = coverage(paging_rows(source))
    merged = []
    replaced = 0
    for row in source:
        key = str(row.get("trajectory_id"))
        replacement = by_id.get(key)
        if replacement is None:
            merged.append(row)
            continue
        if args.tag_loss_scope:
            for message in assistant_turns(replacement):
                message["loss_scope"] = args.tag_loss_scope
        merged.append(replacement)
        replaced += 1

    # 用合并后的池子统计「补写后」，而不是只看补写结果——缺口可能来自没被补写的轨迹。
    after_filled, after_total = coverage(paging_rows(merged))

    missing = sorted(expected_ids - set(by_id))
    if missing:
        print(
            f"[warn] 源池里 {len(missing)} 条翻页轨迹没被补写（{missing[:5]}…），"
            "它们会保持原样进池。",
            file=sys.stderr,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as sink:
        for row in merged:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"源池 {len(source)} 条，补写结果 {len(reasoned)} 条，替换 {replaced} 条")
    print(f"输出 {len(merged)} 条 → {args.output}")
    print()
    print("== 翻页轨迹的「动作回合带思考」覆盖率")
    print(f"   补写前 {before_filled}/{before_total} = {before_filled / max(before_total, 1):.1%}")
    print(f"   补写后 {after_filled}/{after_total} = {after_filled / max(after_total, 1):.1%}")
    if args.tag_loss_scope:
        print(f"   已给这些轨迹的每个 assistant 回合打标：loss_scope={args.tag_loss_scope}")

    if after_filled < after_total and not args.allow_partial:
        print(
            f"\n失败：还有 {after_total - after_filled} 个动作回合没有 content。"
            "\n补思考时这些回合可能全部调用失败（看 augment 的 [warn] 行）。"
            "\n重跑 augment（按 trajectory_id 断点续跑）会补上，别带着缺口去训练。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
