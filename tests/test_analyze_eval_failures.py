"""`scripts/analyze_eval_failures.py` 的纯 CPU 单元测试。

固定五件事（都是 2026-10-18 用真实评测数据踩出来的口径）：
  1. `max_run` 数的是**连续相同动作**（`select_option` 死循环是核心靶点）；
  2. `active_dimensions` 只收 `active=True` 的维度 —— 未激活的维度恒为 0，
     直接平均会把「没要求」误读成「没做到」；
  3. 缺陷必须**有区分度**才算可及空间（成功题里同样普遍的缺陷不算）；
  4. `flat_reward_types` 识别「同 reward_type 内奖励恒定」的零梯度块；
  5. 过程形状判定用**中位数 + 长尾占比**，不被个别离群轨迹带偏。
"""

from __future__ import annotations

import json

import pytest

from scripts.analyze_eval_failures import (
    COMPARISON_OPEN_THRESHOLD,
    LOW_EVIDENCE_THRESHOLD,
    REPEAT_RUN_THRESHOLD,
    active_dimensions,
    analyze,
    classify,
    fingerprint,
    flat_reward_types,
    load_rows,
    main,
    max_run,
    render,
)


def step(name: str, **params) -> dict:
    return {"tool_name": name, "parameters": params, "observation": ""}


def row(
    task_id: int,
    *,
    reward: float,
    reward_type: str,
    steps: list[dict],
    coverage: float = 1.0,
    dims: dict | None = None,
) -> dict:
    dimensions = dims or {
        "key_options": {"active": True, "score": 1.0},
        "brand": {"active": False, "score": 0.0},
    }
    return {
        "task_id": task_id,
        "final_reward": reward,
        "steps": steps,
        "terminal_result": {
            "reward_detail": {
                "reward_type": reward_type,
                "reward_valid": True,
                "evidence_coverage": coverage,
                "evidence": {"preference_scoring": {"dimensions": dimensions}},
            }
        },
    }


def short_traj(asin: str = "A") -> list[dict]:
    return [step("search_products", query="q"), step("open_product", asin=asin), step("purchase", asin=asin)]


def test_max_run_counts_consecutive_identical():
    assert max_run([]) == 0
    assert max_run(["a", "a", "a", "b", "a"]) == 3
    assert max_run(["a", "b", "c"]) == 1


def test_active_dimensions_skips_inactive():
    detail = {
        "evidence": {
            "preference_scoring": {
                "dimensions": {
                    "brand": {"active": False, "score": 0.0},
                    "key_options": {"active": True, "score": 0.5},
                }
            }
        }
    }
    assert active_dimensions(detail) == {"key_options": 0.5}


def test_fingerprint_counts_distinct_opens():
    fp = fingerprint(
        row(
            1,
            reward=-0.85,
            reward_type="wrong_purchase",
            steps=[
                step("search_products", query="x"),
                step("open_product", asin="A"),
                step("open_product", asin="A"),
                step("open_product", asin="B"),
                step("purchase", asin="A"),
            ],
        )
    )
    assert fp["distinct_opens"] == 2
    assert fp["searches"] == 1
    assert fp["purchases"] == 1
    assert fp["steps"] == 5


def test_classify_thresholds():
    loop = fingerprint(
        row(
            1,
            reward=-0.65,
            reward_type="repeat_loop",
            steps=[step("select_option", value="v") for _ in range(REPEAT_RUN_THRESHOLD + 2)],
        )
    )
    assert "action_loop" in classify(loop)

    single_open = fingerprint(row(2, reward=-0.85, reward_type="wrong_purchase", steps=short_traj()))
    assert "no_comparison" in classify(single_open)
    assert "action_loop" not in classify(single_open)

    thin = fingerprint(
        row(
            3,
            reward=-0.2,
            reward_type="partial_alternative_purchase",
            steps=[step("open_product", asin="A"), step("open_product", asin="B")],
            coverage=LOW_EVIDENCE_THRESHOLD / 2,
        )
    )
    assert "low_evidence" in classify(thin)
    assert "no_comparison" not in classify(thin)  # opens == COMPARISON_OPEN_THRESHOLD


def test_defect_without_discrimination_is_not_addressable():
    """成功题同样「只打开一个商品」⇒ no_comparison 不算可及空间。"""
    rows = [
        row(1, reward=1.0, reward_type="gold_purchase", steps=short_traj()),
        row(2, reward=1.0, reward_type="gold_purchase", steps=short_traj()),
        row(3, reward=-0.85, reward_type="wrong_purchase", steps=short_traj()),
        row(4, reward=-0.85, reward_type="wrong_purchase", steps=short_traj()),
    ]
    summary = analyze(rows)
    no_comparison = next(item for item in summary["defects"] if item["defect"] == "no_comparison")
    assert no_comparison["failure_rate"] == pytest.approx(1.0)
    assert no_comparison["success_rate"] == pytest.approx(1.0)
    assert no_comparison["discriminating"] is False
    assert summary["addressable"]["discriminating_defects"] == []
    assert summary["addressable"]["with_discriminating_defect"] == 0


def test_discriminating_defect_is_counted():
    """只有失败组才出现动作循环 ⇒ 有区分度，计入可及空间。"""
    loop_steps = [step("select_option", value="v") for _ in range(REPEAT_RUN_THRESHOLD)]
    rows = [
        row(1, reward=1.0, reward_type="gold_purchase", steps=short_traj()),
        row(2, reward=1.0, reward_type="gold_purchase", steps=short_traj()),
        row(3, reward=-0.65, reward_type="repeat_loop", steps=loop_steps),
        row(4, reward=-0.65, reward_type="repeat_loop", steps=loop_steps),
    ]
    summary = analyze(rows)
    loop = next(item for item in summary["defects"] if item["defect"] == "action_loop")
    assert loop["discriminating"] is True
    assert summary["addressable"]["discriminating_defects"] == ["action_loop"]
    assert summary["addressable"]["with_discriminating_defect"] == 2


def test_flat_reward_types_detects_constant_bucket():
    rows = [
        row(1, reward=-0.85, reward_type="wrong_purchase", steps=short_traj()),
        row(2, reward=-0.85, reward_type="wrong_purchase", steps=short_traj()),
        row(3, reward=-0.2, reward_type="partial_alternative_purchase", steps=short_traj()),
        row(4, reward=0.1, reward_type="partial_alternative_purchase", steps=short_traj()),
    ]
    flat = {item["reward_type"]: item["flat"] for item in flat_reward_types([fingerprint(r) for r in rows])}
    assert flat["wrong_purchase"] is True
    assert flat["partial_alternative_purchase"] is False


def test_zero_gradient_mass_counts_only_failures():
    rows = [
        row(1, reward=1.0, reward_type="gold_purchase", steps=short_traj()),
        row(2, reward=-0.85, reward_type="wrong_purchase", steps=short_traj()),
        row(3, reward=-0.85, reward_type="wrong_purchase", steps=short_traj()),
        row(4, reward=-0.2, reward_type="partial_alternative_purchase", steps=short_traj()),
        row(5, reward=0.1, reward_type="partial_alternative_purchase", steps=short_traj()),
    ]
    summary = analyze(rows)
    assert summary["addressable"]["zero_gradient_failures"] == 2
    assert summary["addressable"]["zero_gradient_share_of_failures"] == pytest.approx(2 / 4)


def test_shape_verdict_uses_median_not_outlier_mean():
    """失败组被 1 条 35 步的离群轨迹拉高均值，但中位数相同、长尾占比仅 3% ⇒ 判「无区分度」。"""
    rows = [row(i, reward=1.0, reward_type="gold_purchase", steps=short_traj()) for i in range(1, 31)]
    rows += [row(i, reward=-0.85, reward_type="wrong_purchase", steps=short_traj()) for i in range(31, 61)]
    rows.append(
        row(
            61,
            reward=-0.5,
            reward_type="max_steps",
            steps=[step("search_products", query=f"q{i}") for i in range(35)],
        )
    )
    summary = analyze(rows)
    assert summary["shape"]["failure"]["mean_steps"] > summary["shape"]["success"]["mean_steps"] + 0.5
    assert summary["shape"]["failure"]["median_steps"] == summary["shape"]["success"]["median_steps"]
    assert summary["shape"]["failure"]["long_share"] < 0.05
    assert "没有区分度" in render(summary)


def test_render_contains_sections():
    rows = [row(1, reward=1.0, reward_type="gold_purchase", steps=short_traj())]
    text = render(analyze(rows))
    assert "结局分布" in text
    assert "缺陷的区分度" in text
    assert "零梯度质量" in text


def test_load_rows_reads_jsonl(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    path.write_text(
        json.dumps(row(1, reward=1.0, reward_type="gold_purchase", steps=short_traj())) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text(json.dumps({"expected_tasks": 1}), encoding="utf-8")
    rows, summary = load_rows(tmp_path)
    assert len(rows) == 1
    assert summary["expected_tasks"] == 1


def test_main_writes_report(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row(task_id, reward=reward, reward_type=kind, steps=short_traj()))
            for task_id, reward, kind in ((1, 1.0, "gold_purchase"), (2, -0.85, "wrong_purchase"))
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "report.md"
    assert main([str(tmp_path), "--out", str(out)]) == 0
    assert "失败解剖" in out.read_text(encoding="utf-8")


def test_missing_file_returns_error(tmp_path):
    assert main([str(tmp_path / "nope")]) == 2
