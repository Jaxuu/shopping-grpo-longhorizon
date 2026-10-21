#!/usr/bin/env python3
"""按来源优先级合并 M10-R 的增量批次，**同题只留一条**。

为什么需要它
------------
`build_m10r_pool.py` 有一条硬契约：增量里 task_id 不能重复（重复会让同一道题
在池子里出现两次，训练时等于给这道题双倍权重），违反直接 ``SystemExit``。
而各批清单是从**同一个语料**按不同口径抽的，天然重叠：

* `m10_paging_targets`（199 题，gold 名次 21–60）是 `m10_rank2_targets`
  （5,000 题，gold 名次 ≥2）的子集；
* 探针的 40 道翻页题又是 `m10_paging_targets` 的子集 ⇒ `paging_requery`
  与 `paging` 必然撞题。

2026-10-21 首次跑建池就死在这里（「增量里 task_id 有重复」）。

去重规则（按顺序，先到先得）
--------------------------
1. ``paging_requery`` —— 真实采集。同一现象下**真实老师轨迹优先于脚本合成**。
2. ``paging``         —— 合成，但它是 ``next_page`` 这个动作的唯一来源（底池只有 11 条）。
3. ``compare``        —— 真实采集，教「先开第 1 名再对比」。
4. ``requery``        —— 真实采集，教「换个说法再搜」。
5. ``rank2``          —— 面最广但**不教任何探索动作**（只是 gold 不在第 1 名），所以垫底。

换句话说：**同一道题被多批覆盖时，留那条"教了更稀缺动作"的轨迹。**

用法::

    .venv/bin/python scripts/merge_m10r_increments.py \\
        --out data/sft_m10r/increment_merged.jsonl \\
        outputs/m10-paging-requery/kept.jsonl \\
        outputs/m10-paging/kept.jsonl \\
        outputs/m10-compare/kept.jsonl \\
        outputs/m10-requery/kept.jsonl \\
        outputs/m10-rank2/kept.jsonl

命令行里**位置参数的顺序就是优先级**。不存在的来源会被跳过并打印，不算错。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def merge(sources: list[Path]) -> tuple[list[dict], list[dict]]:
    """按顺序合并并去重。返回 (合并后的行, 每批的来源报告)。"""
    seen: dict[int, str] = {}
    merged: list[dict] = []
    report: list[dict] = []
    for source in sources:
        label = source.name if source.name else str(source)
        if not source.exists():
            report.append({"source": str(source), "status": "missing", "kept": 0, "duplicate": 0})
            continue
        kept = duplicate = 0
        collide_with: dict[str, int] = {}
        for row in read_jsonl(source):
            task_id = int(row["task_id"])
            if task_id in seen:
                duplicate += 1
                owner = seen[task_id]
                collide_with[owner] = collide_with.get(owner, 0) + 1
                continue
            seen[task_id] = label
            merged.append(row)
            kept += 1
        report.append(
            {
                "source": str(source),
                "status": "merged",
                "rows": kept + duplicate,
                "kept": kept,
                "duplicate": duplicate,
                # 撞题时是被谁占掉的——不看这个就分不清「这批本来就没多少」和「被前一批吃掉了」
                "duplicate_with": dict(sorted(collide_with.items())),
            }
        )
    return merged, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources", nargs="+", type=Path, help="按优先级从高到低排列的 kept.jsonl"
    )
    parser.add_argument("--out", type=Path, required=True, help="合并后的 jsonl")
    parser.add_argument(
        "--report", type=Path, default=None, help="合并报告（默认与 --out 同目录）"
    )
    args = parser.parse_args(argv)

    merged, report = merge(args.sources)
    if not merged:
        print("所有来源都不存在或都是空的，没有可合并的增量。", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report_path = args.report or args.out.with_name("increment_merge_report.json")
    report_path.write_text(
        json.dumps(
            {
                "out": str(args.out),
                "total_rows": len(merged),
                "sources": report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("| 来源 | 原始行 | 并入 | 因重复丢弃 | 被谁占掉 |")
    print("|---|---|---|---|---|")
    for item in report:
        if item["status"] == "missing":
            print(f"| {item['source']} | — | 跳过（不存在） | — | — |")
            continue
        who = "、".join(f"{name} {count}" for name, count in item["duplicate_with"].items())
        print(
            f"| {item['source']} | {item['rows']} | {item['kept']} "
            f"| {item['duplicate']} | {who or '—'} |"
        )
    print(f"\n合并后 **{len(merged)} 行** → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
