"""切片与合并的单测。

两个容易错的地方：
1. 切片必须是**轮转**而不是顺序切块，否则每片的题目类型集中、耗时严重不均；
2. 合并必须以**完整评测集**为分母重算，不能把各片 summary 相加——那样缺题会被
   算成失败，分数被压低且看不出缺了哪些题。
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


split_mod = _load("split_eval_shards", "scripts/split_eval_shards.py")
merge_mod = _load("merge_eval_shards", "scripts/merge_eval_shards.py")


def _success(task_id: int) -> dict:
    return {
        "task_id": task_id,
        "status": "done",
        "done": True,
        "steps": [],
        "terminal_result": {
            "done": True,
            "over": True,
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v4",
                "reward_type": "gold_purchase",
                "reward_valid": True,
                "purchase_success": True,
                "termination_reason": "gold_purchase",
            },
        },
    }


def _failure(task_id: int) -> dict:
    return {
        "task_id": task_id,
        "status": "done",
        "done": True,
        "steps": [],
        "terminal_result": {
            "done": True,
            "over": True,
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v4",
                "reward_type": "wrong_purchase",
                "reward_valid": True,
                "purchase_success": False,
                "termination_reason": "wrong_purchase",
            },
        },
    }


def _write_benchmark(path: pathlib.Path, task_ids) -> pathlib.Path:
    path.write_text(
        "".join(json.dumps({"task_id": task_id}) + "\n" for task_id in task_ids),
        encoding="utf-8",
    )
    return path


def test_split_rows_is_round_robin() -> None:
    rows = [{"task_id": index} for index in range(10)]
    buckets = split_mod.split_rows(rows, 3)
    assert [len(bucket) for bucket in buckets] == [4, 3, 3]
    # 轮转：第 0 片拿到 0/3/6/9，而不是 0/1/2/3。
    assert [row["task_id"] for row in buckets[0]] == [0, 3, 6, 9]


def test_split_covers_every_task_exactly_once(tmp_path) -> None:
    ids = list(range(200))
    paths = split_mod.write_shards(
        [{"task_id": task_id} for task_id in ids], tmp_path / "shards", 8
    )
    assert len(paths) == 8
    seen = []
    for path in paths:
        seen += [json.loads(line)["task_id"] for line in path.open(encoding="utf-8") if line.strip()]
    assert sorted(seen) == ids
    assert len(seen) == len(set(seen))


def test_split_rejects_non_positive_shards() -> None:
    with pytest.raises(ValueError):
        split_mod.split_rows([{"task_id": 1}], 0)


def test_merge_recomputes_with_full_denominator(tmp_path) -> None:
    ids = list(range(200))
    benchmark = _write_benchmark(tmp_path / "tasks.jsonl", ids)
    shard_dir = tmp_path / "shards"
    split_mod.write_shards([{"task_id": task_id} for task_id in ids], shard_dir, 8)

    # 前 124 题成功，其余失败，按片写入轨迹。
    for index in range(8):
        shard = shard_dir / f"shard-{index:02d}"
        rows = [
            _success(task_id) if task_id < 124 else _failure(task_id)
            for task_id in range(index, 200, 8)
        ]
        (shard / "trajectories.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    summary = merge_mod.merge(
        benchmark, shard_dir, 8, tmp_path / "out", model="deepseek-v4-flash"
    )
    assert summary["expected_tasks"] == 200
    assert summary["completed_tasks"] == 200
    assert summary["strict_successes"] == 124
    assert summary["strict_success_rate"] == pytest.approx(0.62)
    assert summary["missing_tasks"] == []
    assert summary["protocol"]["model"] == "deepseek-v4-flash"
    assert summary["protocol"]["context_window"] == 0


def test_merge_reports_missing_tasks(tmp_path) -> None:
    ids = list(range(20))
    benchmark = _write_benchmark(tmp_path / "tasks.jsonl", ids)
    shard_dir = tmp_path / "shards"
    split_mod.write_shards([{"task_id": task_id} for task_id in ids], shard_dir, 2)

    # 只写第 0 片，第 1 片完全缺失。
    (shard_dir / "shard-00" / "trajectories.jsonl").write_text(
        "".join(json.dumps(_success(task_id)) + "\n" for task_id in range(0, 20, 2)),
        encoding="utf-8",
    )
    summary = merge_mod.merge(benchmark, shard_dir, 2, tmp_path / "out", model="m")
    assert summary["completed_tasks"] == 10
    assert summary["missing_tasks"] == list(range(1, 20, 2))
    # 分母仍是 20：缺的题算失败，不会被悄悄缩小分母。
    assert summary["strict_success_rate"] == pytest.approx(0.5)

    merged_rows = (tmp_path / "out" / "trajectories.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(merged_rows) == 10
