"""`scripts/analyze_hard_pool_progress.py` 的纯 CPU 单元测试（不依赖 torch / verl）。"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_hard_pool_progress import (
    collect_groups,
    detect_uid_kind,
    group_stats,
    load_pool_ids,
    split_by_hardness,
    summarize_val,
    window_rows,
)


def make_group(uid: int, success: bool, utility: float | None = None) -> dict:
    utils = [utility if utility is not None else (1.0 if success else -0.85)] * 4
    return {
        "uid": uid,
        "terminal_utilities": utils,
        "purchase_success": [success, False, False, False],
        "kept": success,
        "drop_reason": None if success else "constant_reward",
    }


def write_pool(tmp_path: Path) -> tuple[Path, Path]:
    rows = [{"extra_info": {"task_id": i, "index": i}} for i in range(3)]
    rows += [{"extra_info": {"task_id": 9000 + j, "index": 3 + j}} for j in range(2)]
    pool = tmp_path / "pool.jsonl"
    pool.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    hard = tmp_path / "hard.jsonl"
    hard.write_text(
        "".join(json.dumps({"task_id": 9000 + j}) + "\n" for j in range(2)),
        encoding="utf-8",
    )
    return pool, hard


def test_load_pool_ids_separates_hard_by_both_keys(tmp_path: Path) -> None:
    pool, hard = write_pool(tmp_path)
    sets, n_rows = load_pool_ids(pool, hard)
    assert n_rows == 5
    assert sets["task_id"] == {9000, 9001}
    assert sets["index"] == {3, 4}


def test_detect_uid_kind_prefers_task_id_when_values_exceed_pool_size() -> None:
    hard_sets = {"task_id": {9000, 9001}, "index": {3, 4}}
    assert detect_uid_kind([9000, 1, 9001], hard_sets, 5) == "task_id"
    assert detect_uid_kind([3, 1, 4], hard_sets, 5) == "index"
    assert detect_uid_kind(["a", "b"], hard_sets, 5) == "unknown"


def test_group_stats_counts_any_success_and_utility() -> None:
    stats = group_stats([make_group(1, True), make_group(2, False)])
    assert stats["groups"] == 2
    assert stats["kept"] == 1
    assert stats["any_success"] == 1
    assert stats["rollouts"] == 8
    assert stats["utility_mean"] == (1.0 * 4 + -0.85 * 4) / 8


def test_split_by_hardness_routes_by_uid() -> None:
    per_step = {1: [make_group(3, False), make_group(0, True)]}
    stats = split_by_hardness(per_step, {3, 4}, "index")
    assert stats[1]["hard"]["groups"] == 1
    assert stats[1]["easy"]["groups"] == 1
    assert stats[1]["hard"]["any_success_rate"] == 0.0
    assert stats[1]["easy"]["any_success_rate"] == 1.0


def test_window_rows_weights_utility_by_rollouts() -> None:
    per_step = {
        1: [make_group(3, False)],
        2: [make_group(3, True), make_group(3, True)],
    }
    stats = split_by_hardness(per_step, {3}, "index")
    rows = window_rows(stats, 10)
    assert len(rows) == 1
    assert rows[0]["hard"]["groups"] == 3
    assert rows[0]["hard"]["any_success_rate"] == 2 / 3
    # 12 条 rollout：8 条 1.0（step 2）+ 4 条 -0.85（step 1）
    assert rows[0]["hard"]["utility_mean"] == (8 * 1.0 + 4 * -0.85) / 12


def test_collect_groups_merges_multiple_batches_per_step() -> None:
    records = [
        {"event": "generation_batch", "global_step": 2, "groups": [make_group(1, False)]},
        {"event": "generation_batch", "global_step": 2, "groups": [make_group(2, True)]},
        {"event": "optimizer_step", "global_step": 2, "metrics": {}},
    ]
    per_step = collect_groups(records)
    assert len(per_step[2]) == 2


def test_summarize_val_keeps_only_val_prefixed_metrics() -> None:
    records = [
        {
            "event": "optimizer_step",
            "global_step": 50,
            "metrics": {
                "val-core/shopsimulator/reward/mean@1": 0.41,
                "actor/ppo_kl": 0.002,
                "val-aux/num_turns": 6.0,
            },
        },
        {"event": "optimizer_step", "global_step": 51, "metrics": {"actor/ppo_kl": 0.003}},
    ]
    rows = summarize_val(records)
    assert len(rows) == 1
    assert rows[0]["step"] == 50
    assert set(rows[0]["metrics"]) == {
        "val-core/shopsimulator/reward/mean@1",
        "val-aux/num_turns",
    }
