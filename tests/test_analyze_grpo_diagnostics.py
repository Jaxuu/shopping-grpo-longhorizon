"""`scripts/analyze_grpo_diagnostics.py` 的纯 CPU 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_grpo_diagnostics import (
    analyse,
    compare_with_probe,
    estimate_filter_saving,
    read_records,
    render,
    summarize_metrics,
    summarize_timing,
)


def group(kept: bool, accepted: bool = False, reason: str | None = None) -> dict:
    return {"uid": "u", "kept": kept, "accepted": accepted, "drop_reason": reason}


def generation_batch(
    step: int,
    index: int,
    groups: list[dict],
    target_prompts: int = 4,
    accepted_after: int | None = None,
) -> dict:
    record = {
        "schema_version": 1,
        "event": "generation_batch",
        "global_step": step,
        "generation_batch": index,
        "groups": groups,
        "target_prompts": target_prompts,
    }
    if accepted_after is not None:
        record["accepted_prompts_after_batch"] = accepted_after
    return record


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
        generation_batch(1, 1, [group(True, True) for _ in range(4)], accepted_after=4),
        {"event": "optimizer_step", "global_step": 1, "metrics": {"actor/ppo_kl": 0.0001}},
        generation_batch(2, 1, [group(True, True) for _ in range(4)], accepted_after=4),
        {"event": "optimizer_step", "global_step": 2, "metrics": {"actor/ppo_kl": 0.0002}},
    ]
    summary = analyse(read_records(write(tmp_path / "d.jsonl", records)))
    text = render(summary, probe=None, max_gen_batches=3)

    assert "完全没有任何更新的步" in text
    assert "每一步都收满了" in text
    assert "lr" in text
    assert "不能指望改数据集带来提升" in text


def test_render_includes_cost_breakdown_and_filter_saving(tmp_path):
    """成本结构章节应给出 rollout 占比与过滤的省时估算。"""
    records = [
        generation_batch(1, 1, [group(True, True)] * 4, accepted_after=4),
        {
            "event": "optimizer_step",
            "global_step": 1,
            "metrics": {"actor/ppo_kl": 0.005, "timing_s/step": 140.0, "timing_s/gen": 40.0},
        },
        generation_batch(2, 1, [group(True, True)] * 4, accepted_after=4),
        {
            "event": "optimizer_step",
            "global_step": 2,
            "metrics": {"actor/ppo_kl": 0.006, "timing_s/step": 140.0, "timing_s/gen": 40.0},
        },
    ]
    summary = analyse(read_records(write(tmp_path / "d.jsonl", records)))
    text = render(summary, probe=None, max_gen_batches=3)

    assert "成本结构" in text
    assert "timing_s/gen" in text
    assert "28.6%" in text  # 40 / 140
    assert "分钟" in text


def test_batch_fill_reports_whether_every_step_reached_target(tmp_path):
    """batch_fill 回答「模型每步到底拿到几个有梯度的组」——过滤数据集有没有信号可捡的决定性证据。"""
    records = [
        # step 1：两个 batch 才凑满（第一个 batch 只收到 1 个）
        generation_batch(1, 1, [group(True, True)] + [group(False)] * 3, accepted_after=1),
        generation_batch(1, 2, [group(True, True)] * 3 + [group(False)], accepted_after=4),
        {"event": "optimizer_step", "global_step": 1, "metrics": {}},
        # step 2：一个 batch 就收满
        generation_batch(2, 1, [group(True, True)] * 4, accepted_after=4),
        {"event": "optimizer_step", "global_step": 2, "metrics": {}},
    ]
    summary = analyse(read_records(write(tmp_path / "d.jsonl", records)))
    fill = summary["batch_fill"]

    assert fill["target_prompts"] == 4
    assert fill["steps_with_fill_info"] == 2
    assert fill["steps_reached_target"] == 2
    assert fill["steps_below_target"] == 0
    assert fill["mean_filled_per_step"] == pytest.approx(4.0)
    assert fill["min_filled_per_step"] == 4


def test_batch_fill_flags_steps_that_never_reached_target(tmp_path):
    """没被收满的步必须被点名 —— 那才是"过滤/扩容才有意义"的场景。"""
    records = [
        generation_batch(1, 1, [group(True, True)] + [group(False)] * 3, accepted_after=1),
        generation_batch(1, 2, [group(False)] * 4, accepted_after=1),
        {"event": "skipped_update", "global_step": 1, "metrics": {}},
        generation_batch(2, 1, [group(True, True)] * 4, accepted_after=4),
        {"event": "optimizer_step", "global_step": 2, "metrics": {}},
    ]
    summary = analyse(read_records(write(tmp_path / "d.jsonl", records)))
    fill = summary["batch_fill"]

    assert fill["steps_with_fill_info"] == 2
    assert fill["steps_reached_target"] == 1
    assert fill["steps_below_target"] == 1
    assert fill["below_target_steps"] == [1]
    assert fill["min_filled_per_step"] == 1


def test_batch_fill_is_absent_when_fields_missing(tmp_path):
    """旧诊断文件没有这两个字段时不能崩，只应报告 0 个可比较步。"""
    records = [
        generation_batch(1, 1, [group(True, True)] * 4),
        {"event": "optimizer_step", "global_step": 1, "metrics": {}},
    ]
    # generation_batch 默认带 target_prompts，但这里显式去掉以模拟旧文件
    for record in records:
        record.pop("target_prompts", None)
    summary = analyse(read_records(write(tmp_path / "d.jsonl", records)))
    assert summary["batch_fill"]["steps_with_fill_info"] == 0
    assert summary["batch_fill"]["target_prompts"] is None


def test_summarize_timing_computes_step_shares():
    per_step = {
        1: {"timing_s/step": 140.0, "timing_s/gen": 42.0, "timing_s/update_actor": 80.0},
        2: {"timing_s/step": 160.0, "timing_s/gen": 38.0, "timing_s/update_actor": 100.0},
    }
    timing = summarize_timing(per_step)

    assert timing["available"] is True
    assert timing["step_seconds"] == pytest.approx(150.0)
    assert timing["mean_seconds"]["timing_s/gen"] == pytest.approx(40.0)
    # 40 / 150 ≈ 26.7%
    assert timing["shares"]["timing_s/gen"] == pytest.approx(40.0 / 150.0)
    # mean_seconds 按耗时降序：step(150) > update_actor(90) > gen(40)
    assert list(timing["mean_seconds"]) == [
        "timing_s/step",
        "timing_s/update_actor",
        "timing_s/gen",
    ]


def test_summarize_timing_falls_back_when_step_key_missing():
    timing = summarize_timing({1: {"timing_s/gen": 40.0, "timing_s/update_actor": 90.0}})
    assert timing["available"] is True
    # 没有 timing_s/step 时用最大分项兜底，并在来源里标注
    assert timing["step_seconds"] == pytest.approx(90.0)
    assert "max(" in timing["step_seconds_source"]


def test_summarize_timing_ignores_token_length_keys():
    """`timing_s/agent_loop/slowest/response_length` 是 token 数，不是秒。

    M6-R 第 5 步的真实数据里它是 6270.8，被当成秒会得出 4729% 的单步占比。
    """
    per_step = {
        1: {
            "timing_s/step": 132.6,
            "timing_s/gen": 15.8,
            "timing_s/update_actor": 108.4,
            "timing_s/agent_loop/slowest/response_length": 6270.8,
            "timing_s/agent_loop/slowest/prompt_length": 1856.6,
        },
    }
    timing = summarize_timing(per_step)

    assert "timing_s/agent_loop/slowest/response_length" not in timing["mean_seconds"]
    assert "timing_s/agent_loop/slowest/prompt_length" not in timing["mean_seconds"]
    assert timing["step_seconds"] == pytest.approx(132.6)
    # 所有占比都必须 ≤ 1（此前 response_length 会算出 47.3 倍）
    assert max(timing["shares"].values()) <= 1.0
    assert timing["shares"]["timing_s/gen"] == pytest.approx(15.8 / 132.6)


def test_summarize_timing_fallback_not_polluted_by_token_length():
    """兜底分支同样不能被 token 长度污染：否则单步秒数会被算成 6270.8。"""
    timing = summarize_timing(
        {
            1: {
                "timing_s/gen": 15.8,
                "timing_s/update_actor": 108.4,
                "timing_s/agent_loop/slowest/response_length": 6270.8,
            },
        }
    )
    assert timing["step_seconds"] == pytest.approx(108.4)
    assert "response_length" not in timing["step_seconds_source"]


def test_estimate_filter_saving_uses_measured_batches_and_rollout_share(tmp_path):
    """过滤省时估算 = rollout 平均耗时 × generation batch 的下降比例。"""
    records = []
    for step in range(1, 5):
        # 每步 2 个 batch（模拟重试），共 8 个 batch
        for index in (1, 2):
            records.append(
                generation_batch(
                    step,
                    index,
                    [group(True, True)] * 4,
                    accepted_after=4,
                )
            )
        records.append(
            {
                "event": "optimizer_step",
                "global_step": step,
                "metrics": {"timing_s/step": 100.0, "timing_s/gen": 40.0},
            }
        )
    summary = analyse(read_records(write(tmp_path / "d.jsonl", records)))

    saving = summary["filter_saving_estimate"]
    assert saving is not None
    assert saving["rollout_key"] == "timing_s/gen"
    assert saving["rollout_share"] == pytest.approx(0.4)
    assert saving["batches_before"] == 8
    # 过滤后：4 步 × 1.2 = 4.8 个 batch ⇒ 下降 40%
    assert saving["batches_after_estimate"] == pytest.approx(4.8)
    assert saving["batch_reduction"] == pytest.approx(1 - 4.8 / 8)
    assert saving["saved_seconds_per_step"] == pytest.approx(40.0 * (1 - 4.8 / 8))
    # 4 步 × 每步省下的秒数 / 60
    assert saving["saved_minutes_total"] == pytest.approx(
        40.0 * (1 - 4.8 / 8) * 4 / 60.0
    )


def test_no_timing_keys_means_no_saving_estimate(tmp_path):
    records = [
        generation_batch(1, 1, [group(True, True)] * 4, accepted_after=4),
        {"event": "optimizer_step", "global_step": 1, "metrics": {"actor/ppo_kl": 0.01}},
    ]
    summary = analyse(read_records(write(tmp_path / "d.jsonl", records)))
    assert summary["timing"]["available"] is False
    assert summary["filter_saving_estimate"] is None


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
