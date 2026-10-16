"""`scripts/analyze_grpo_diagnostics.py` 的纯 CPU 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_grpo_diagnostics import analyse, compare_with_probe, read_records


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


def test_analyse_counts_updates_skips_and_kept_rate(tmp_path):
    records = [
        # step 1：第一个 batch 就凑满 4 个 kept ⇒ 更新
        generation_batch(1, 1, [group(True, True) for _ in range(4)]),
        {"event": "optimizer_step", "global_step": 1},
        # step 2：两个 batch 共 8 个组里只有 2 个 kept ⇒ 凑不满 ⇒ 跳过
        generation_batch(
            2, 1, [group(True, True), group(False, reason="constant_reward"), group(False, reason="constant_reward"), group(False, reason="constant_reward")]
        ),
        generation_batch(
            2, 2, [group(True, True), group(False, reason="constant_reward"), group(False, reason="sampling_invalid"), group(False, reason="constant_reward")]
        ),
        {"event": "skipped_update", "global_step": 2, "metrics": {}},
    ]
    path = write(tmp_path / "d.jsonl", records)

    summary = analyse(read_records(path))

    assert summary["records"] == 5
    assert summary["steps"]["optimizer_updated_steps"] == 1
    assert summary["steps"]["skipped_steps"] == 1
    assert summary["steps"]["skip_rate"] == pytest.approx(0.5)
    assert summary["groups"]["seen"] == 12
    assert summary["groups"]["kept"] == 6
    assert summary["groups"]["kept_rate"] == pytest.approx(0.5)
    assert summary["groups"]["drop_reason_counts"]["constant_reward"] == 5
    assert summary["groups"]["drop_reason_counts"]["sampling_invalid"] == 1
    assert summary["generation_batches"]["total"] == 3
    # per_step_counts 是「每步 batch 数 -> 出现该批数的步数」
    assert summary["generation_batches"]["per_step_counts"] == {1: 1, 2: 1}
    assert summary["generation_batches"]["max_per_step"] == 2


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
