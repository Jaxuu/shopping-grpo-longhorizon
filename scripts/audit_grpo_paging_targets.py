"""GRPO 训练池里有多少题「需要翻页」——决定 GRPO 到底能不能教出翻页。

背景：v3 证明补文本无效（模型推理时 0 条动作回合带内容，见 diagnose_v3_collapse.py 的 C 部分），
剩下唯一能改行为的机制是奖励。但奖励只在训练题上生效：如果 906 道训练题里没有
「gold 在第 20 名之外」的题，翻页这个动作在 GRPO 里永远拿不到奖励差，跑多久都是空转
（M7 Gate 3 的 0.3% 梯度份额就是这么来的）。

本脚本离线复现「指令原文 + 类目 + 属性」三种忠实查询的 BM25 名次（与 M10-R 选题同口径），
统计训练池按名次分桶的题数。纯 CPU、只读，不需要环境服务。

    .venv/bin/python scripts/audit_grpo_paging_targets.py \
        --train data/grpo/train.parquet --index .workbuddy-ai/cache/products_local.sqlite3

判读：
  - 「需要翻页」≥50 题 ⇒ GRPO 有靶子，可以开；
  - 10–50 题 ⇒ 靶子太少，先考虑把 A 桶题并进训练池；
  - <10 题 ⇒ GRPO 教不出翻页，别花 3 小时。
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import select_m10_collection_targets as selection  # noqa: E402
from generate_m10_paging import CORPUS, best_candidate, pages_needed  # noqa: E402

DEFAULT_INDEX = ROOT / ".workbuddy-ai" / "cache" / "products_local.sqlite3"
BUCKETS = ((1, 20), (21, 40), (41, 60), (61, 120), (121, 10**6))


def load_task_ids(path: Path) -> list[int]:
    """从 JSONL 或 GRPO parquet（extra_info 列）里取 task_id。"""
    if path.suffix == ".jsonl":
        ids = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "task_id" in row:
                ids.append(int(row["task_id"]))
            elif isinstance(row.get("extra_info"), dict):
                ids.append(int(row["extra_info"]["task_id"]))
        return ids

    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 环境缺 pyarrow
        raise SystemExit(f"需要 pyarrow 读 parquet：{exc}") from exc
    table = pq.read_table(path, columns=["extra_info"])
    return [int(value["task_id"]) for value in table.column("extra_info").to_pylist()]


def bucket_of(rank: int | None) -> str:
    if rank is None:
        return "名次>150（翻页也看不到）"
    for low, high in BUCKETS:
        if low <= rank <= high:
            return f"{low}-{high}" if high < 10**6 else f"{low}+"
    return "未知"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "grpo" / "train.parquet")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--limit", type=int, default=0, help="只扫前 N 题（调试用）")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.train.is_file():
        raise SystemExit(f"找不到训练池：{args.train}")
    task_ids = load_task_ids(args.train)
    if args.limit:
        task_ids = task_ids[: args.limit]
    print(f"训练池 {len(task_ids)} 题；索引 {args.index}")

    with gzip.open(CORPUS, "rt", encoding="utf-8") as handle:
        products = json.load(handle)
    searcher = selection.load_searcher(args.index)

    histogram: collections.Counter = collections.Counter()
    details: list[dict] = []
    for task_id in task_ids:
        if not (0 <= task_id < len(products)):
            histogram["task_id 越界"] += 1
            continue
        item = products[task_id]
        instructions = item.get("instructions") or []
        if not instructions:
            histogram["无指令"] += 1
            continue
        instruction = instructions[0]
        asin = str(item.get("asin"))
        picked = best_candidate(
            [(query, _rank_of(searcher, query, asin)) for query in selection.build_queries(item, instruction)]
        )
        rank = picked[1] if picked else None
        histogram[bucket_of(rank)] += 1
        if rank is not None and rank > 20:
            details.append(
                {
                    "task_id": task_id,
                    "rank": rank,
                    "pages": pages_needed(rank),
                    "asin": asin,
                }
            )

    order = [f"{low}-{high}" if high < 10**6 else f"{low}+" for low, high in BUCKETS]
    order.append("名次>150（翻页也看不到）")
    print("\n== 按 gold 名次分桶")
    for name in order:
        count = histogram.get(name, 0)
        if count:
            share = count / max(len(task_ids), 1)
            print(f"   {name:>16} | {count:>4} 题 | {share:5.1%}")

    need_paging = sum(1 for row in details if row["rank"] <= 150)
    print(f"\n== 结论：需要翻页（21–150 名）的题 {need_paging} 道")
    if need_paging >= 50:
        print("   GRPO 有靶子：这些题在组内会因「翻不翻」产生奖励差。")
    elif need_paging >= 10:
        print("   靶子偏少：先考虑把 A 桶题并进训练池，否则大部分 batch 组内全错、被 dynamic_sampling 跳过。")
    else:
        print("   靶子不足：GRPO 教不出翻页，别花 3 小时。")

    if args.json:
        print(json.dumps({"total": len(task_ids), "buckets": histogram, "details": details}, ensure_ascii=False))
    return 0


def _rank_of(searcher, query: str, asin: str) -> int | None:
    try:
        hits = searcher.search(query, k=150)
    except Exception:  # noqa: BLE001 - 空查询会抛，跳过
        return None
    for hit in hits:
        if str(hit.asin) == asin:
            return int(hit.rank)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
