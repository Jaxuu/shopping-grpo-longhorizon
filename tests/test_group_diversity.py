"""`shopping_grpo.training.grpo.group_diversity` 的纯 CPU 单元测试。"""

from __future__ import annotations

import pytest

from shopping_grpo.training.grpo.group_diversity import (
    aggregate_groups,
    behavior_fingerprint,
    compare_n,
    group_trajectories,
    summarize_group,
)


def make_trajectory(
    task_id: int,
    *,
    reward: float = 1.0,
    reward_type: str = "gold_purchase",
    reward_valid: bool = True,
    opened: tuple[str, ...] = ("A1",),
    queries: tuple[str, ...] = ("q1",),
    purchased: str | None = "A1",
    error: dict | None = None,
) -> dict:
    steps = [{"step_index": 0, "tool_name": "search_products", "parameters": {"query": q}}
             for q in queries]
    steps += [
        {"step_index": 1, "tool_name": "open_product", "parameters": {"asin": asin}}
        for asin in opened
    ]
    if purchased is not None:
        steps.append(
            {"step_index": 2, "tool_name": "buy_now", "parameters": {"asin": purchased}}
        )
    return {
        "trajectory_id": f"t-{task_id}-{reward}-{opened}-{queries}",
        "task_id": task_id,
        "attempt_index": 0,
        "steps": steps,
        "blocked_tool_calls": [],
        "tool_call_truncations": [],
        "context_compactions": [],
        "final_reward": reward,
        "done": True,
        "status": "done",
        "error": error,
        "terminal_result": {
            "reward_type": reward_type,
            "termination_reason": reward_type,
            "reward_valid": reward_valid,
            "reward": reward,
            "purchase": {"asin": purchased} if purchased else None,
        },
    }


def test_constant_reward_group_is_dropped():
    group = [make_trajectory(1, reward=1.0) for _ in range(4)]
    row = summarize_group(group)
    assert row["reward_varying"] is False
    assert row["drop_reason"] == "constant_reward"
    assert row["kept"] is False


def test_varying_reward_group_is_kept():
    group = [
        make_trajectory(1, reward=1.0),
        make_trajectory(1, reward=1.0),
        make_trajectory(1, reward=-0.65, reward_type="repeat_loop"),
        make_trajectory(1, reward=0.25, reward_type="partial_alternative_purchase"),
    ]
    row = summarize_group(group)
    assert row["reward_varying"] is True
    assert row["drop_reason"] is None
    assert row["kept"] is True
    assert row["utility_min"] == pytest.approx(-0.65)
    assert row["utility_max"] == pytest.approx(1.0)


def test_sampling_invalid_group_is_dropped_even_when_reward_varies():
    group = [
        make_trajectory(1, reward=1.0),
        make_trajectory(1, reward=-0.85, reward_type="wrong_purchase"),
        make_trajectory(1, reward=0.0, reward_type="reward_unverifiable", reward_valid=False),
        make_trajectory(1, reward=1.0),
    ]
    row = summarize_group(group)
    assert row["reward_varying"] is True
    assert row["sampling_invalid_reasons"] == ["reward_unverifiable"]
    assert row["drop_reason"] == "sampling_invalid"
    assert row["kept"] is False


def test_error_trajectory_counts_as_sampling_invalid():
    group = [make_trajectory(1), make_trajectory(1, reward=-0.5, error={"type": "URLError"})]
    row = summarize_group(group)
    assert "error" in row["sampling_invalid_reasons"]
    assert row["kept"] is False


def test_tolerance_boundary_treats_tiny_difference_as_constant():
    group = [make_trajectory(1, reward=1.0), make_trajectory(1, reward=1.0 + 1e-12)]
    assert summarize_group(group, tolerance=1.0e-8)["reward_varying"] is False


def test_behavioral_diversity_counts_distinct_fingerprints():
    group = [
        make_trajectory(1, opened=("A1",), purchased="A1"),
        make_trajectory(1, opened=("A2",), purchased="A2"),
        make_trajectory(1, opened=("A2",), purchased="A2"),
        make_trajectory(1, opened=("A3", "A4"), purchased="A4", queries=("q1", "q2")),
    ]
    row = summarize_group(group)
    assert row["distinct_open_sets"] == 3
    assert row["distinct_purchases"] == 3
    assert row["distinct_query_sets"] == 2
    assert row["distinct_tool_sequences"] >= 2


def test_behavior_fingerprint_is_order_insensitive_for_opened_products():
    first = make_trajectory(1, opened=("A2", "A1"), purchased="A1")
    second = make_trajectory(1, opened=("A1", "A2"), purchased="A1")
    assert behavior_fingerprint(first)["open_set"] == behavior_fingerprint(second)["open_set"]


def test_compare_n_shows_raising_n_rescues_a_constant_n4_group():
    """前 4 条同分、后 4 条不同分：n=4 该组无效，n=8 有效。"""
    attempts = [make_trajectory(7, reward=0.25, reward_type="partial_alternative_purchase")
                for _ in range(4)]
    attempts += [
        make_trajectory(7, reward=1.0, opened=("GOLD",), purchased="GOLD"),
        make_trajectory(7, reward=-0.85, reward_type="wrong_purchase", purchased="BAD"),
        make_trajectory(7, reward=1.0, opened=("GOLD",), purchased="GOLD"),
        make_trajectory(7, reward=0.25, reward_type="partial_alternative_purchase"),
    ]
    grouped = group_trajectories(attempts)
    assert len(grouped[7]) == 8

    comparison = compare_n(grouped, [4, 8])
    assert comparison["n=4"]["signal"]["kept_group_rate"] == pytest.approx(0.0)
    assert comparison["n=8"]["signal"]["kept_group_rate"] == pytest.approx(1.0)
    assert comparison["n=4"]["num_groups_usable"] == 1
    assert comparison["n=8"]["num_groups_usable"] == 1


def test_compare_n_skips_groups_that_cannot_supply_n_attempts():
    grouped = group_trajectories([make_trajectory(1), make_trajectory(1)])
    comparison = compare_n(grouped, [2, 8])
    assert "n=2" in comparison
    assert "n=8" not in comparison


def test_group_trajectories_keys_by_task_id():
    grouped = group_trajectories([make_trajectory(3), make_trajectory(4), make_trajectory(3)])
    assert sorted(grouped) == [3, 4]
    assert len(grouped[3]) == 2


def test_aggregate_rejects_empty_input():
    with pytest.raises(ValueError):
        aggregate_groups([])


def test_aggregate_reports_expected_rates():
    rows = [
        summarize_group([make_trajectory(1), make_trajectory(1)]),  # constant -> dropped
        summarize_group(
            [
                make_trajectory(2, reward=1.0),
                make_trajectory(2, reward=-0.65, reward_type="repeat_loop"),
            ]
        ),  # varying -> kept
    ]
    report = aggregate_groups(rows)
    assert report["num_groups"] == 2
    assert report["num_trajectories"] == 4
    assert report["signal"]["constant_reward_group_rate"] == pytest.approx(0.5)
    assert report["signal"]["kept_group_rate"] == pytest.approx(0.5)
    assert report["reward"]["reward_type_counts"]["gold_purchase"] == 3
    assert report["reward"]["reward_type_counts"]["repeat_loop"] == 1
