"""按「行为」砍掉多余的模板轨迹，把池子压到目标规模。

为什么要砍
----------
2026-10-22 实测：1,912 条池子里 **1,350 条（71%）是「搜一次、开第一个、买」**，
真正含探索动作（翻页 / 换词 / 开≥2 个）的只有 508 条（27%）。旧的三段课程又把
模板练了 2–3 遍，曝光比到 **5 : 1**——模型只会更确定地「开第一个」，而 GRPO
需要的组内方差正好被压没了。

**砍模板不是丢数据，是改配权。** 探索轨迹一条不动；模板只留到 ``--keep-template``
条，其余按固定 seed 丢掉，可复现。

产物
----
* ``<out-dir>/all.jsonl`` —— 砍后的池子（4 键 schema 原样保留）
* ``<out-dir>/difficulty_labels.jsonl`` —— 跟着筛（行必须一一对应）
* ``<out-dir>/downsample_report.json`` —— 砍了多少、为什么、曝光比变化

用法::

    .venv/bin/python scripts/downsample_sft_pool.py \\
        --pool data/sft_m10r/all.jsonl \\
        --labels data/sft_m10r/difficulty_labels.jsonl \\
        --keep-template 700 --out-dir data/sft_m10r/capped
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_sft_curriculum import behavior_bucket, behavior_profile  # noqa: E402

SCHEMA_VERSION = "shopping-sft-downsample-v1"


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _order_key(task_id: int, seed: int) -> str:
    """稳定排序键：同一个 seed + task_id 永远给出同一个顺序。"""
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()


def select_template(
    template_ids: list[int], keep: int | None, seed: int
) -> tuple[list[int], list[int]]:
    """返回 (保留的模板 task_id, 丢掉的)。``keep=None`` 表示一条不丢。"""
    if keep is None or keep >= len(template_ids):
        return sorted(template_ids), []
    ordered = sorted(template_ids, key=lambda task_id: _order_key(task_id, seed))
    kept = sorted(ordered[:keep])
    dropped = sorted(ordered[keep:])
    return kept, dropped


def downsample(
    rows: list[dict], *, keep_template: int | None, seed: int
) -> tuple[list[dict], dict]:
    profiles = {}
    buckets: dict[str, list[int]] = collections.defaultdict(list)
    for row in rows:
        task_id = int(row["task_id"])
        profile = behavior_profile(row)
        profiles[task_id] = profile
        buckets[behavior_bucket(profile)].append(task_id)

    kept_template, dropped_template = select_template(
        buckets["template"], keep_template, seed
    )
    keep_ids = set(buckets["exploratory"]) | set(buckets["other"]) | set(kept_template)
    kept_rows = [row for row in rows if int(row["task_id"]) in keep_ids]

    def _exposure(ids: list[int]) -> dict:
        explored = sum(1 for task_id in ids if profiles[task_id]["explored"])
        return {"rows": len(ids), "exploratory": explored, "template": len(ids) - explored}

    before = _exposure([int(row["task_id"]) for row in rows])
    after = _exposure([int(row["task_id"]) for row in kept_rows])
    report = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "keep_template": keep_template,
        "buckets_before": {name: len(ids) for name, ids in sorted(buckets.items())},
        "kept_template": len(kept_template),
        "dropped_template": len(dropped_template),
        "before": before,
        "after": after,
        "dropped_task_ids": dropped_template,
    }
    return kept_rows, report


def render(report: dict) -> str:
    before, after = report["before"], report["after"]
    lines = [
        f"砍前 {before['rows']} 行：探索 {before['exploratory']}"
        f" · 非探索 {before['template']}",
        f"砍后 {after['rows']} 行：探索 {after['exploratory']}"
        f" · 非探索 {after['template']}",
        f"模板 {report['buckets_before'].get('template', 0)} → {report['kept_template']}"
        f"（丢弃 {report['dropped_template']} 条，seed {report['seed']} 可复现）",
    ]
    if after["rows"]:
        share = after["exploratory"] / after["rows"]
        lines.append(f"探索占比：{before['exploratory'] / before['rows']:.1%} → {share:.1%}")
    lines.append(
        "桶分布（砍前）："
        + "，".join(f"{name} {count}" for name, count in report["buckets_before"].items())
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pool", type=Path, required=True, help="SFT 池 all.jsonl")
    parser.add_argument("--labels", type=Path, default=None, help="难度标签，跟着一起筛")
    parser.add_argument(
        "--keep-template",
        type=int,
        default=700,
        help="模板轨迹保留条数（不给就全留；0 表示全砍，不推荐）",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args(argv)

    if args.keep_template is not None and args.keep_template < 0:
        raise SystemExit("--keep-template 不能为负")
    rows = read_jsonl(args.pool)
    kept_rows, report = downsample(
        rows, keep_template=args.keep_template, seed=args.seed
    )
    if not kept_rows:
        raise SystemExit("砍完一条都不剩，检查 --keep-template")

    write_jsonl(args.out_dir / "all.jsonl", kept_rows)
    keep_ids = {int(row["task_id"]) for row in kept_rows}
    if args.labels and args.labels.is_file():
        labels = read_jsonl(args.labels)
        kept_labels = [row for row in labels if int(row["task_id"]) in keep_ids]
        write_jsonl(args.out_dir / "difficulty_labels.jsonl", kept_labels)
        missing = len(keep_ids) - len(kept_labels)
        if missing:
            print(
                f"⚠️ 有 {missing} 行没有难度标签（新题的标签还没补），"
                "先跑 label_sft_difficulty_offline.py",
                file=sys.stderr,
            )
    (args.out_dir / "downsample_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(render(report))
    print(f"写入 {args.out_dir}/all.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
