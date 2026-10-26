#!/usr/bin/env python3
"""把 N 片并发评测的轨迹合并成一次完整评测，并重算 summary。

为什么不能直接把各片的 summary 相加：分母会变成「各片题数之和」，missing 判断也
只在片内成立。这里统一以**完整评测集**为分母重算，缺题会出现在 missing_tasks 里，
而不是被悄悄算成失败。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopping_grpo.evaluation.summary import summarize_trajectories


def read_expected_ids(benchmark: Path) -> list[int]:
    ids = []
    with Path(benchmark).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.append(int(json.loads(line)["task_id"]))
    return ids


def read_shard_trajectories(shard_dir: Path, shards: int) -> dict[int, dict]:
    """task_id -> 轨迹。分片之间任务互斥，重复出现时后者覆盖（便于单题重跑）。"""
    merged: dict[int, dict] = {}
    for index in range(shards):
        path = Path(shard_dir) / f"shard-{index:02d}" / "trajectories.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                task_id = row.get("task_id")
                if task_id is None:
                    continue
                merged[int(task_id)] = row
    return merged


def merge(
    benchmark: Path,
    shard_dir: Path,
    shards: int,
    output_dir: Path,
    *,
    model: str,
    max_steps: int = 35,
    observation_token_budget: int = 1536,
    observation_tokenizer: str = "local-hf-tokenizer",
) -> dict:
    expected = read_expected_ids(benchmark)
    merged = read_shard_trajectories(shard_dir, shards)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for task_id in sorted(merged):
            handle.write(json.dumps(merged[task_id], ensure_ascii=False) + "\n")

    summary = summarize_trajectories(expected, [merged[task_id] for task_id in sorted(merged)])
    summary["protocol"] = {
        "benchmark": str(benchmark),
        "model": model,
        "reward_contract": "shopsimulator-reward-v3",
        "max_steps": max_steps,
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
        # 远端 API 没有 chat 级 /tokenize，所以关掉上下文预算；observation 截断仍按
        # 本地 tokenizer 执行，保证与本地模型的截断口径一致。
        "context_window": 0,
        "observation_token_budget": observation_token_budget,
        "observation_tokenizer": observation_tokenizer,
        "shards": shards,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="合并分片评测结果")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="unknown")
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--observation-token-budget", type=int, default=1536)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = merge(
        args.benchmark,
        args.shard_dir,
        args.shards,
        args.output_dir,
        model=args.model,
        max_steps=args.max_steps,
        observation_token_budget=args.observation_token_budget,
    )
    print(
        f"  完成 {summary['completed_tasks']}/{summary['expected_tasks']}，"
        f"缺失 {len(summary['missing_tasks'])} 题"
    )
    print(
        f"  严格成功 {summary['strict_successes']}/{summary['expected_tasks']}"
        f"（{summary['strict_success_rate'] * 100:.1f}%）"
    )
    if summary["missing_tasks"]:
        print(f"  缺失 task_id：{summary['missing_tasks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
