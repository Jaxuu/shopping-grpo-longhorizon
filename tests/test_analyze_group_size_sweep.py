"""`scripts/analyze_group_size_sweep.py` 的纯 CPU 单元测试。

固定四件事：
  1. `load_groups` 默认剔除 sampling-invalid 轨迹（与 `eligible` 同口径）；
  2. `n = 4` 时直接报**实测**比例，不用 bootstrap（bootstrap 在 n=4 会低估）；
  3. `bootstrap` 与 `beta_binomial` 都随 `n` 单调不减；
  4. 每步 rollout 成本随 `n` 上升（这是取舍的另一半）。
"""

from __future__ import annotations

import json

import pytest

from scripts.analyze_group_size_sweep import (
    analyze,
    beta_binomial_mixed_rate,
    bootstrap_mixed_rate,
    load_groups,
    main,
    observed_mixed_rate,
    render,
)


def write_trajectories(tmp_path, rows) -> object:
    path = tmp_path / "trajectories.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def traj(task_id: int, attempt: int, reward: float, **extra) -> dict:
    row = {
        "task_id": task_id,
        "attempt_index": attempt,
        "final_reward": reward,
        "terminal_result": {"reward_detail": {"reward_valid": True, "reward_type": "gold_purchase"}},
    }
    row.update(extra)
    return row


@pytest.fixture()
def groups():
    # 1 题全同（1.0），1 题 2 成功 2 失败，1 题 3 成功 1 失败
    return {
        1: [1.0, 1.0, 1.0, 1.0],
        2: [1.0, 1.0, -0.65, -0.65],
        3: [1.0, 1.0, 1.0, 0.55],
    }


def test_load_groups_drops_sampling_invalid(tmp_path):
    rows = [
        traj(1, 0, 1.0),
        traj(1, 1, 1.0),
        traj(1, 2, 1.0),
        traj(1, 3, 1.0),
        # 第 2 题有一条 error 轨迹 ⇒ 整条被剔除，剩 3 条仍可分组
        traj(2, 0, -0.85),
        traj(2, 1, -0.85, error="boom"),
        traj(2, 2, 1.0),
        traj(2, 3, -0.85),
    ]
    path = write_trajectories(tmp_path, rows)
    groups = load_groups(path)
    assert set(groups) == {1, 2}
    assert groups[1] == [1.0, 1.0, 1.0, 1.0]
    assert len(groups[2]) == 3
    assert 1.0 in groups[2]  # 非 invalid 的那条保留


def test_load_groups_keep_invalid(tmp_path):
    rows = [traj(1, i, 1.0) for i in range(4)]
    rows.append(traj(1, 4, -0.85, error="boom"))
    path = write_trajectories(tmp_path, rows)
    assert len(load_groups(path, keep_invalid=True)[1]) == 4  # 仍只取前 4 条


def test_observed_mixed_rate(groups):
    assert observed_mixed_rate(groups) == pytest.approx(2 / 3)


def test_bootstrap_is_lower_bound_and_monotone(groups):
    rates = [bootstrap_mixed_rate(groups, n, trials=400) for n in (4, 8, 16)]
    assert rates[0] <= rates[1] <= rates[2]
    # 全同的题在 bootstrap 下永远贡献 0 ⇒ 上限是 2/3
    assert rates[2] <= 2 / 3 + 1e-9


def test_beta_binomial_monotone_and_bounds(groups):
    rates = [beta_binomial_mixed_rate(groups, n) for n in (4, 8, 16)]
    assert rates[0] <= rates[1] <= rates[2]
    assert all(0.0 <= rate <= 1.0 for rate in rates)


def test_beta_binomial_handles_all_same():
    # 4/4 全同：Jeffreys 先验下仍给出正概率（假设真实 p 在内部）
    rate = beta_binomial_mixed_rate({1: [1.0, 1.0, 1.0, 1.0]}, 8)
    assert 0.0 < rate < 1.0


def test_analyze_reports_observed_rate_at_baseline(groups):
    summary = analyze(groups, group_sizes=(4, 8), trials=200)
    first = summary["rows"][0]
    assert first["n"] == 4
    # n=4 用实测值，不用 bootstrap
    assert first["mixed_rate_bootstrap"] == pytest.approx(observed_mixed_rate(groups))
    assert summary["rows"][1]["rollouts_per_step"] >= first["rollouts_per_step"]


def test_render_contains_key_sections(groups):
    text = render(analyze(groups, group_sizes=(4, 8), trials=100))
    assert "组大小" in text
    assert "每 1000 条 rollout" in text
    assert "每步训练组数不变" in text


def test_main_writes_report(tmp_path, groups):
    rows = []
    for task_id, rewards in groups.items():
        for attempt, reward in enumerate(rewards):
            rows.append(traj(task_id, attempt, reward))
    path = write_trajectories(tmp_path, rows)
    out = tmp_path / "report.md"
    assert main([str(path), "--group-sizes", "4,8", "--trials", "50", "--out", str(out)]) == 0
    assert out.is_file()
    assert "组大小" in out.read_text(encoding="utf-8")
