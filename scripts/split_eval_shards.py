#!/usr/bin/env python3
"""把评测集切成 N 份，供多进程并发跑（采集循环本身是串行的）。

切法是**轮转分配**（第 i 题给第 i%N 片），不是按顺序切块：顺序切块会让某一片
集中某一类任务，8 片跑完的耗时会严重不均，而且中途出错时残缺的是整段题号。
轮转分配下每片题量最多差 1 题。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_task_ids(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_rows(rows: list[dict], shards: int) -> list[list[dict]]:
    if shards < 1:
        raise ValueError("shards must be at least 1")
    buckets: list[list[dict]] = [[] for _ in range(shards)]
    for index, row in enumerate(rows):
        buckets[index % shards].append(row)
    return buckets


def write_shards(rows: list[dict], shard_dir: Path, shards: int) -> list[Path]:
    shard_dir = Path(shard_dir)
    paths = []
    for index, bucket in enumerate(split_rows(rows, shards)):
        shard = shard_dir / f"shard-{index:02d}"
        shard.mkdir(parents=True, exist_ok=True)
        path = shard / "tasks.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in bucket),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="把评测集切成 N 份")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=8)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = read_task_ids(args.benchmark)
    paths = write_shards(rows, args.shard_dir, args.shards)
    for path in paths:
        count = sum(1 for line in path.open(encoding="utf-8") if line.strip())
        print(f"  {path.parent.name}: {count} 题")
    print(f"  合计 {len(rows)} 题")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
