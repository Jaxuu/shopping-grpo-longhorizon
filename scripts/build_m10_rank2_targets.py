#!/usr/bin/env python3
"""M10-R：生成「答案不在第 1 名」采样用的任务清单。

为什么换了一套选题口径
----------------------
2026-10-20 实测（`plans/m10-step2-closing.md`）：

* SFT 池 1,246 条可解析轨迹里，模型打开的 ASIN 有 **65.9% 就是搜索结果的第 1 名**；
* 评测 200 题按 gold 名次切开：**第 1 名对 → 86.8%，不对 → 34.0%**；
* ⇒ 模型学到的规则是「开第一个」，而不是「核对后挑对」。

修法：把训练集里「正确答案不在第 1 名」的样本从 34.1% 推到 50–70%，
让「开第一个」变成亏本策略。

为什么**不**按检索名次选题
--------------------------
Step 1 用「三个忠实查询取最好名次」选题，被 Step 2 实测证伪：
同一批题里老师真实 query 下 **14 题 gold 进前 20、10 题掉出 150 名**。
⇒ 离线预测老师会看到什么不可行（`docs/lessons-learned.md` §2.7）。

所以这里**均匀抽未使用的 task_id**，不做任何名次预判；跑完再用老师真实轨迹筛
（`scripts/filter_m10_rank2.py`）。命中率按评测集估约 **16%**（47% 的题 gold 不在第 1 名
× 其中 34% 老师买对），即跑 4,000 题约得 640 条。

用法::

    .venv/Scripts/python.exe scripts/build_m10_rank2_targets.py --limit 4000
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent


def _siblings():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import select_m10_collection_targets as selection  # noqa: PLC0415

    return selection


def pick_task_ids(total: int, used: set[int], limit: int, seed: int) -> list[int]:
    """从 0..total-1 里剔除已用 id，打乱后取前 limit 个（返回时排序）。纯函数，便于测试。"""
    unused = [task_id for task_id in range(total) if task_id not in used]
    if limit > len(unused):
        raise ValueError(f"limit {limit} 超过可选数量 {len(unused)}")
    random.Random(seed).shuffle(unused)
    return sorted(unused[:limit])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--limit", type=int, default=4000, help="抽多少题")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "collection" / "m10_rank2_targets.jsonl",
    )
    args = parser.parse_args(argv)

    selection = _siblings()
    used = selection.load_used_ids()
    with gzip.open(selection.CORPUS, "rt", encoding="utf-8") as handle:
        products = json.load(handle)

    total = len(products)
    print(f"语料 {total} 条；已用 task_id {len(used)} 个；可选 {total - len(used)} 条")
    try:
        picked = pick_task_ids(total, used, args.limit, args.seed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for task_id in picked:
            handle.write(json.dumps({"task_id": task_id}, ensure_ascii=False) + "\n")

    print(f"写入 {args.out}：{len(picked)} 题（seed={args.seed}）")
    print(f"task_id 范围 {picked[0]}–{picked[-1]}；与四池零重叠（已排除 {len(used)} 个）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
