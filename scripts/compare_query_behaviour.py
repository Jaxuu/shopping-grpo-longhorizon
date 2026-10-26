#!/usr/bin/env python3
"""对比两次评测里模型**实际写出的搜索查询**。

分数一样有两种完全不同的解释，只看总分分不出来：

* 提示词被忽略了 —— 查询压根没变，那这次实验等于没做；
* 查询真的变好了，但救不回分数 —— 那「写全 query」这条线该关掉。

这个脚本只看中间变量：查询文本改没改、词数变多变少、改对的题和改错的题
分别是什么情况。它不读 gold，也不改任何评测产物。

    .venv/bin/python scripts/compare_query_behaviour.py \
      --base outputs/evaluation/m10r-v2/trajectories.jsonl \
      --variant outputs/evaluation/prompt-query-attrs/trajectories.jsonl \
      --show 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SEARCH_TOOL = "search_products"


def load_trajectories(path):
    """task_id -> 轨迹；同一 task 出现多次时取最后一条（与合并口径一致）。"""
    records = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row.get("task_id")
            if task_id is not None:
                records[int(task_id)] = row
    return records


def search_queries(record):
    """按出现顺序取出这条轨迹里所有搜索查询。"""
    queries = []
    for step in record.get("steps") or []:
        if step.get("tool_name") != SEARCH_TOOL:
            continue
        query = (step.get("parameters") or {}).get("query")
        if isinstance(query, str) and query.strip():
            queries.append(query.strip())
    return queries


def term_count(query):
    """BM25 的查询词数：这类查询用空格分词，所以空格切分就是词数。"""
    return len(query.split())


def task_terms(record):
    """这条轨迹里出现过的所有查询词（去重），用来算重合度。"""
    terms = set()
    for query in search_queries(record):
        terms.update(query.split())
    return terms


def is_success(record):
    detail = ((record.get("terminal_result") or {}).get("reward_detail")) or {}
    return detail.get("reward_type") == "gold_purchase"


def _jaccard(left, right):
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def compare_records(base, variant):
    """逐题对比查询行为，返回汇总 + 明细。"""
    shared = sorted(set(base) & set(variant))
    rows = []
    for task_id in shared:
        base_record, variant_record = base[task_id], variant[task_id]
        base_queries = search_queries(base_record)
        variant_queries = search_queries(variant_record)
        base_terms = task_terms(base_record)
        variant_terms = task_terms(variant_record)
        base_first = base_queries[0] if base_queries else ""
        variant_first = variant_queries[0] if variant_queries else ""
        base_ok, variant_ok = is_success(base_record), is_success(variant_record)
        rows.append(
            {
                "task_id": task_id,
                "base_queries": base_queries,
                "variant_queries": variant_queries,
                "changed": base_queries != variant_queries,
                "first_term_delta": term_count(variant_first) - term_count(base_first),
                "jaccard": _jaccard(base_terms, variant_terms),
                "base_searches": len(base_queries),
                "variant_searches": len(variant_queries),
                "base_success": base_ok,
                "variant_success": variant_ok,
                "outcome": (
                    "fixed"
                    if variant_ok and not base_ok
                    else "broke"
                    if base_ok and not variant_ok
                    else "same"
                ),
            }
        )

    def mean(values):
        values = list(values)
        return sum(values) / len(values) if values else 0.0

    def mean_first_terms(records):
        values = [term_count(search_queries(record)[0]) for record in records if search_queries(record)]
        return mean(values)

    def mean_terms(records):
        values = [term_count(query) for record in records for query in search_queries(record)]
        return mean(values)

    shared_ids = shared
    outcome_counts = {"fixed": 0, "broke": 0, "same": 0}
    for row in rows:
        outcome_counts[row["outcome"]] += 1

    summary = {
        "tasks": len(rows),
        "tasks_query_changed": sum(1 for row in rows if row["changed"]),
        "tasks_first_query_longer": sum(1 for row in rows if row["first_term_delta"] > 0),
        "tasks_first_query_shorter": sum(1 for row in rows if row["first_term_delta"] < 0),
        "mean_first_query_terms": {
            "base": round(mean_first_terms([base[t] for t in shared_ids]), 2),
            "variant": round(mean_first_terms([variant[t] for t in shared_ids]), 2),
        },
        "mean_all_query_terms": {
            "base": round(mean_terms([base[t] for t in shared_ids]), 2),
            "variant": round(mean_terms([variant[t] for t in shared_ids]), 2),
        },
        "mean_searches": {
            "base": round(mean([row["base_searches"] for row in rows]), 2),
            "variant": round(mean([row["variant_searches"] for row in rows]), 2),
        },
        "mean_term_jaccard": round(mean([row["jaccard"] for row in rows]), 3),
        "outcome": outcome_counts,
        "rows": rows,
    }
    return summary


def _query_block(queries, limit=4):
    if not queries:
        return "（没搜索）"
    shown = queries[:limit]
    text = " / ".join(f"`{q}`" for q in shown)
    if len(queries) > limit:
        text += f" 等 {len(queries)} 次"
    return text


def render_markdown(summary, show=10):
    lines = ["# 两次评测的查询行为对比", "", "## 总量", "", "| 项 | 值 |", "|---|---:|"]
    lines += [
        f"| 对比题数 | {summary['tasks']} |",
        f"| 查询文本有变化的题 | {summary['tasks_query_changed']}"
        f"（{summary['tasks_query_changed'] / summary['tasks'] * 100:.1f}%） |"
        if summary["tasks"]
        else "| 查询文本有变化的题 | 0 |",
        f"| 首次查询变长 / 变短 | {summary['tasks_first_query_longer']} / "
        f"{summary['tasks_first_query_shorter']} |",
        f"| 首次查询平均词数 | {summary['mean_first_query_terms']['base']} → "
        f"{summary['mean_first_query_terms']['variant']} |",
        f"| 全部查询平均词数 | {summary['mean_all_query_terms']['base']} → "
        f"{summary['mean_all_query_terms']['variant']} |",
        f"| 平均搜索次数 | {summary['mean_searches']['base']} → "
        f"{summary['mean_searches']['variant']} |",
        f"| 查询词集合平均重合度 | {summary['mean_term_jaccard']} |",
        "",
        "## 结果",
        "",
        "| 项 | 题数 |",
        "|---|---:|",
        f"| 变体做对、基线做错 | {summary['outcome']['fixed']} |",
        f"| 基线做对、变体做错 | {summary['outcome']['broke']} |",
        f"| 两边一样 | {summary['outcome']['same']} |",
        "",
        "## 判读",
        "",
    ]
    changed_rate = summary["tasks_query_changed"] / summary["tasks"] if summary["tasks"] else 0.0
    if changed_rate < 0.1:
        lines.append(
            "查询几乎没变 ⇒ **这次实验没有测到提示词**：模型没照着改，"
            "不能据此说「写全 query 没用」。要验证只能换更强的约束（少样本示范或训练）。"
        )
    elif summary["mean_term_jaccard"] > 0.8 and summary["mean_first_query_terms"]["variant"] <= summary[
        "mean_first_query_terms"
    ]["base"]:
        lines.append(
            "查询改了但词数没变多 ⇒ 提示词被理解了但没执行到「补全属性」这一步，"
            "属于执行力度问题，不是知识问题。"
        )
    else:
        lines.append(
            "查询确实变长/换词了 ⇒ 提示词生效。若分数仍不变，说明瓶颈不在查询质量，"
            "而在「拿到候选之后选不选得对」。"
        )

    interesting = [row for row in summary["rows"] if row["outcome"] != "same" and row["changed"]]
    if interesting and show:
        lines += ["", f"## 改对的 {summary['outcome']['fixed']} 题 / 改错的 "
                  f"{summary['outcome']['broke']} 题（最多列 {show} 条）", "",
                  "| task_id | 结果 | 基线查询 | 变体查询 |", "|---:|:--:|---|---|"]
        for row in interesting[:show]:
            lines.append(
                f"| {row['task_id']} | {row['outcome']} | {_query_block(row['base_queries'])} | "
                f"{_query_block(row['variant_queries'])} |"
            )
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="对比两次评测的搜索查询行为")
    parser.add_argument("--base", type=Path, required=True, help="基线轨迹 JSONL")
    parser.add_argument("--variant", type=Path, required=True, help="变体轨迹 JSONL")
    parser.add_argument("--output", type=Path, default=None, help="markdown 输出路径")
    parser.add_argument("--json", type=Path, default=None, help="明细 JSON 输出路径")
    parser.add_argument("--show", type=int, default=10, help="明细表最多列几行")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = compare_records(load_trajectories(args.base), load_trajectories(args.variant))
    text = render_markdown(summary, show=args.show)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"已写入 {args.output}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
