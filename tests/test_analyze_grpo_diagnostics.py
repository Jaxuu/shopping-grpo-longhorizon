"""`scripts/analyze_grpo_diagnostics.py` 的纯 CPU 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_grpo_diagnostics import (
    analyse,
    compare_with_probe,
    read_records,
    render,
    summarize_metrics,
)


def group(kept: bool, accepted: bool = False, reason: str | None = None) -> dict:
    return {"uid": "u", "kept": kept, "accepted": accepted, "drop_reason": reason}


def generation_batch(step: int, index: int, groups: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "event": "generation_batch",
        "global_step": step,
        "generation_batch": index,
        "groups": groups,
        "target_prompts": 4,
    }


def write(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    return path


def test_skipped_update_is_a_retry_round_not_a_lost_step(tmp_path):
    """核心口径：skipped_update 是重试轮次；总步数只看 optimizer_step 的不同 global_step。"""
    records = [
        # step 1：第一个 batch 就凑满 4 个 kept ⇒ 更新
        generation_batch(1, 1, [group(True, True) for _ in range(4)]),
        {"event": "optimizer_step", "global_step": 1, "metrics": {"actor/ppo_kl": 0.002}},
        # step 2 第 1 轮：两个 batch 共 8 个组里只有 2 个 kept ⇒ 凑不满 ⇒ 跳过（重试）
        generation_batch(
            2,
            1,
            [
                group(True, True),
                group(False, reason="constant_reward"),
                group(False, reason="constant_reward"),
                group(False, reason="constant_reward"),
            ],
        ),
        generation_batch(
            2,
            2,
            [
                group(True, True),
                group(False, reason="constant_reward"),
                group(False, reason="constant_reward"),
                group(False, reason="sampling_invalid"),
            ],
        ),
        {"event": "skipped_update", "global_step": 2, "metrics": {}},
        # step 2 第 2 轮：凑满 ⇒ 这一步照样更新了
        generation_batch(2, 3, [group(True, True) for _ in range(4)]),
        {"event": "optimizer_step", "global_step": 2, "metrics": {"actor/ppo_kl": 0.004}},
    ]
    path = write(tmp_path / "d.jsonl", records)

    summary = analyse(read_records(path))

    assert summary["records"] == 7
    # 总步数 = optimizer_step 的不同 step 数，而不是「事件总数 - 跳过数」
    assert summary["steps"]["total_steps"] == 2
    assert summary["steps"]["optimizer_updated_steps"] == 2
    assert summary["steps"]["skipped_rounds"] == 1
    assert summary["steps"]["distinct_steps_with_retry"] == 1
    assert summary["steps"]["retry_rounds_per_step"] == pytest.approx(0.5)
    assert summary["steps"]["steps_with_retry_that_still_updated"] == 1
    assert summary["steps"]["steps_skipped_without_any_update"] == 0
    assert summary["steps"]["first_skipped_step"] == 2
    assert summary["steps"]["last_skipped_step"] == 2

    assert summary["groups"]["seen"] == 16
    assert summary["groups"]["kept"] == 10
    assert summary["groups"]["kept_rate"] == pytest.approx(0.625)
    assert summary["groups"]["accepted"] == 10
    assert summary["groups"]["drop_reason_counts"]["constant_reward"] == 5
    assert summary["groups"]["drop_reason_counts"]["sampling_invalid"] == 1
    assert summary["generation_batches"]["total"] == 4
    # per_step_counts 是「每步 batch 数 -> 出现该批数的步数」
    assert summary["generation_batches"]["per_step_counts"] == {1: 1, 3: 1}
    assert summary["generation_batches"]["max_per_step"] == 3
    assert summary["optimizer_metrics"]["num_steps_with_metrics"] == 2


def test_step_without_any_update_is_flagged(tmp_path):
    """极端情况：某步重试到中止都没更新（连续跳过上限），应被单独计数。"""
    records = [
        generation_batch(1, 1, [group(False, reason="constant_reward") for _ in range(4)]),
        {"event": "skipped_update", "global_step": 1, "metrics": {}},
    ]
    path = write(tmp_path / "d.jsonl", records)
    summary = analyse(read_records(path))

    assert summary["steps"]["total_steps"] == 0
    assert summary["steps"]["skipped_rounds"] == 1
    assert summary["steps"]["steps_skipped_without_any_update"] == 1
    # 没有更新步时不应除零
    assert summary["steps"]["retry_rounds_per_step"] == 0.0


def test_summarize_metrics_picks_interesting_keys(tmp_path):
    per_step = {
        1: {"actor/ppo_kl": 0.001, "actor/pg_clipfrac": 0.0, "noise/x": 5.0},
        2: {"actor/ppo_kl": 0.003, "actor/pg_clipfrac": 0.0005, "noise/x": 7.0},
        3: {"actor/ppo_kl": 0.002, "actor/pg_clipfrac": 0.0002, "noise/x": 6.0},
    }
    summary = summarize_metrics(per_step)

    assert summary["num_steps_with_metrics"] == 3
    kl = summary["picked"]["actor/ppo_kl"]
    assert kl["first_step"] == 1
    assert kl["first"] == pytest.approx(0.001)
    assert kl["last_step"] == 3
    assert kl["last"] == pytest.approx(0.002)
    assert kl["min"] == pytest.approx(0.001)
    assert kl["max"] == pytest.approx(0.003)
    assert kl["mean"] == pytest.approx(0.002)
    # 不在白名单里的键被过滤掉
    assert "noise/x" not in summary["picked"]


def test_render_reports_zero_lost_steps_and_points_at_lr(tmp_path):
    """渲染结果应明确「0 步丢失」，并在 KL 贴 0 时把矛头指向 lr/步数/n 而非数据集。"""
    records = [
        generation_batch(1, 1, [group(True, True) for _ in range(4)]),
        {"event": "optimizer_step", "global_step": 1, "metrics": {"actor/ppo_kl": 0.0001}},
        generation_batch(2, 1, [group(True, True) for _ in range(4)]),
        {"event": "optimizer_step", "global_step": 2, "metrics": {"actor/ppo_kl": 0.0002}},
    ]
    summary = analyse(read_records(write(tmp_path / "d.jsonl", records)))
    text = render(summary, probe=None, max_gen_batches=3)

    assert "完全没有任何更新的步" in text
    assert "0" in text
    assert "lr" in text
    assert "不该改数据集" in text


def test_compare_with_probe_computes_rates(tmp_path):
    labels = tmp_path / "difficulty_labels.jsonl"
    write(
        labels,
        [
            {"task_id": 1, "reward_varying": True, "eligible": True},
            {"task_id": 2, "reward_varying": True, "eligible": False},
            {"task_id": 3, "reward_varying": False, "eligible": False},
            {"task_id": 4, "reward_varying": False, "eligible": False},
        ],
    )
    summary = {"groups": {"kept_rate": 0.4}}
    probe = compare_with_probe(summary, labels)
    assert probe["probe_tasks"] == 4
    assert probe["probe_reward_varying"] == 2
    assert probe["probe_reward_varying_rate"] == pytest.approx(0.5)
    assert probe["probe_eligible"] == 1
    assert probe["probe_eligible_rate"] == pytest.approx(0.25)
    assert probe["training_kept_rate"] == pytest.approx(0.4)


def test_read_records_rejects_missing_and_malformed(tmp_path):
    with pytest.raises(SystemExit, match="不存在"):
        read_records(tmp_path / "nope.jsonl")

    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="不是合法 JSON"):
        read_records(bad)
