#!/usr/bin/env python3
"""失败是在第 1 步的「查询词」上决定的吗？——比查询质量，不比查询文字。

为什么不能用「是否和老师逐字一致」判
------------------------------------
`diagnose_imitation.py` 显示：121 道失败里 114 道在第 1 步就分叉，看起来像
「照做都做不到」。但**179 道成功的题里也有 151 道在第 1 步分叉**（84%）。
第 1 步是自由文本查询，模型几乎从不逐字照抄（300 道里只有 10 道完全一致），
所以「分叉」测的是「有没有照抄」，不是「会不会」。那条判据因此不成立。

换成比「查询质量」：同一道题，老师的第一句查询和模型的第一句查询，各自
把 gold 顶到第几名。检索是确定的，所以这是可复现的客观量。

判读
----
* 失败的题里，模型自己的查询把 gold 挤出前 20 名的比例 ≥50% ⇒ 失败是**检索词**
  造成的（少写了区分性属性）⇒ 处方是「教它保留区分词」，属于**改分布**，不是加量；
* 两边查询质量差不多、失败却集中 ⇒ 问题在选择环节，另判。

用法::

    .venv/bin/python scripts/diagnose_query_quality.py \\
        --run outputs/evaluation/selftest-300 \\
        --pool data/sft_m10r/capped/all.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments" / "ShopSimulator" / "shop_env"
if str(SHOP_ENV) not in sys.path:
    sys.path.append(str(SHOP_ENV))

DEFAULT_POOL = ROOT / "data" / "sft_m10r" / "capped" / "all.jsonl"
PAGE = 20  # 渲染一页的行数；进不了前 20 名就等于「这一页看不到」


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load("build_probe_sets", ROOT / "scripts" / "build_probe_sets.py")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def is_success(record: Mapping) -> bool:
    detail = ((record.get("terminal_result") or {}).get("reward_detail")) or {}
    return detail.get("purchase_success") is True or detail.get("reward_type") == "gold_purchase"


def first_query_teacher(row: Mapping) -> str | None:
    for message in row.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") != "search_products":
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return None
            if isinstance(arguments, Mapping):
                query = arguments.get("query")
                return str(query) if query else None
    return None


def first_query_model(record: Mapping) -> str | None:
    for step in record.get("steps") or []:
        if step.get("tool_name") != "search_products":
            continue
        parameters = step.get("parameters") or {}
        if isinstance(parameters, Mapping):
            query = parameters.get("query")
            return str(query) if query else None
    return None


def rank_for(searcher, query: str | None, gold: str, depth: int = 150) -> int | None:
    if not query:
        return None
    for hit in searcher.search(query, k=depth):
        if hit.asin == gold:
            return hit.rank
    return None


def top_page(rank: int | None) -> bool:
    return rank is not None and rank <= PAGE


def summarize(ranks: Sequence[int | None]) -> dict[str, object]:
    found = sorted(rank for rank in ranks if rank is not None)
    return {
        "n": len(ranks),
        "in_page": sum(1 for rank in ranks if top_page(rank)),
        "median": found[len(found) // 2] if found else None,
        "missing": sum(1 for rank in ranks if rank is None),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--products", type=Path, default=probe.audit.DEFAULT_PRODUCTS)
    parser.add_argument("--index", type=Path, default=probe.DEFAULT_INDEX)
    parser.add_argument("--show", type=int, default=4)
    args = parser.parse_args(argv)

    goals = probe.audit.build_goals(probe.audit.load_products(args.products))
    pool = {int(row["task_id"]): row for row in read_jsonl(args.pool)}
    records = read_jsonl(Path(args.run) / "trajectories.jsonl")
    if not pool or not records:
        raise SystemExit("池或轨迹为空，检查 --pool / --run")

    searcher = probe.load_searcher(args.index)

    rows = []
    for record in records:
        task_id = int(record["task_id"])
        pool_row = pool.get(task_id)
        if pool_row is None:
            continue
        goal = goals[task_id]
        gold = str(goal.get("asin"))
        t_query = first_query_teacher(pool_row)
        m_query = first_query_model(record)
        rows.append(
            {
                "task_id": task_id,
                "success": is_success(record),
                "t_query": t_query,
                "m_query": m_query,
                "t_rank": rank_for(searcher, t_query, gold),
                "m_rank": rank_for(searcher, m_query, gold),
            }
        )

    if not rows:
        raise SystemExit("没有一道题能同时对上池和轨迹")

    print("# 第 1 步的查询质量：老师 vs 模型")
    print()
    print(f"能对齐的题 {len(rows)} 道；gold 进前 {PAGE} 名 = 「这一页能看到」")
    print()

    t_stats = summarize([row["t_rank"] for row in rows])
    m_stats = summarize([row["m_rank"] for row in rows])
    print("| 查询来自 | 进前 20 名 | 名次中位 | 150 名内搜不到 |")
    print("|---|---:|---:|---:|")
    for label, stats in (("老师", t_stats), ("模型", m_stats)):
        print(
            f"| {label} | {stats['in_page']}/{stats['n']}"
            f"（{stats['in_page'] / stats['n']:.0%}） | {stats['median']} | {stats['missing']} |"
        )
    print()

    fails = [row for row in rows if not row["success"]]
    oks = [row for row in rows if row["success"]]
    print("## 按结果拆开（只看模型自己的查询）")
    print()
    print("| 分组 | 题数 | 模型查询进前 20 名 | 老师查询进前 20 名 |")
    print("|---|---:|---:|---:|")
    for label, group in (("做错", fails), ("做对", oks)):
        if not group:
            continue
        m_in = sum(1 for row in group if top_page(row["m_rank"]))
        t_in = sum(1 for row in group if top_page(row["t_rank"]))
        print(f"| {label} | {len(group)} | {m_in}（{m_in / len(group):.0%}） | {t_in}（{t_in / len(group):.0%}） |")
    print()

    worse = [
        row for row in rows
        if row["t_rank"] is not None and (row["m_rank"] is None or row["m_rank"] > row["t_rank"])
    ]
    better = [
        row for row in rows
        if row["m_rank"] is not None and (row["t_rank"] is None or row["m_rank"] < row["t_rank"])
    ]
    print("## 逐题对比")
    print()
    print(f"- 模型的查询把 gold 排得**更靠后**（或直接搜不到）：{len(worse)} 道")
    print(f"- 模型的查询把 gold 排得**更靠前**：{len(better)} 道")
    print()

    examples = [row for row in fails if not top_page(row["m_rank"])][: args.show]
    if examples:
        print("## 样例（做错、且模型自己的查询把 gold 挤出前 20 名）")
        print()
        print("| task | 老师查询（名次） | 模型查询（名次） |")
        print("|---|---|---|")
        for row in examples:
            print(
                f"| {row['task_id']} | `{row['t_query']}`（{row['t_rank']}） | "
                f"`{row['m_query']}`（{row['m_rank'] or '搜不到'}） |"
            )
        print()

    print("## 判定")
    print()
    if not fails:
        print("没有失败题，无法判定。")
    else:
        miss = sum(1 for row in fails if not top_page(row["m_rank"]))
        share = miss / len(fails)
        if share >= 0.5:
            print(
                f"**失败是检索词造成的**：{miss}/{len(fails)}（{share:.0%}）的失败题里，"
                "模型自己那句查询根本没能把 gold 顶进前 20 名。⇒ 处方是「教它保留区分性属性」，"
                "属于**改分布**，不是加量。"
            )
        else:
            print(
                f"**失败不是检索词造成的**：只有 {miss}/{len(fails)}（{share:.0%}）的失败题"
                "搜不到 gold。⇒ 问题在选择环节（看见了却没挑对），另判。"
            )
    print()
    print("> 检索是确定的，所以同一句查询的名次可复现；这里比的是查询质量，不是查询文字。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
