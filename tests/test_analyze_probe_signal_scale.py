"""`scripts/analyze_probe_signal_scale.py` 的纯 CPU 单元测试。

固定三件事，防止以后有人"顺手"改掉关键语义：
  1. `spread = utility_max - utility_min`（与动态采样的准入判据同口径）；
  2. tolerance 扫描的**方向**：阈值越高，存活题越少、接受率越低、越难收满 4 组；
  3. 判读的分档（近似退化率 ≥50% ⇒ 换杠杆；<25% ⇒ 提 lr 是对的）。
"""

from __future__ import annotations

import json

import pytest

from scripts.analyze_probe_signal_scale import (
    DEFAULT_TOLERANCE,
    analyze,
    fill_probability,
    load_labels,
    main,
    quantile,
    render,
    spread,
    verdict,
)


def label(
    task_id: int,
    *,
    utility_min: float,
    utility_max: float,
    utility_std: float | None = None,
    pass_count: int = 2,
    pass_rate: float = 0.5,
    varying: bool | None = None,
    eligible: bool | None = None,
) -> dict:
    """构造一行 `difficulty_labels.jsonl`。"""
    if utility_std is None:
        utility_std = (utility_max - utility_min) / 2.0
    if varying is None:
        varying = (utility_max - utility_min) > DEFAULT_TOLERANCE
    if eligible is None:
        eligible = varying
    return {
        "task_id": task_id,
        "attempts_observed": 4,
        "filter_n": 4,
        "pass_count": pass_count,
        "pass_rate": pass_rate,
        "reward_varying": varying,
        "utility_min": utility_min,
        "utility_max": utility_max,
        "utility_std": utility_std,
        "sampling_invalid_reasons": [],
        "eligible": eligible,
        "filter_reason": None if eligible else "constant_reward",
    }


# ── 基础工具 ────────────────────────────────────────────────────────────


class TestSpreadAndQuantile:
    def test_spread_uses_max_minus_min(self) -> None:
        assert spread({"utility_min": -0.3, "utility_max": 1.0}) == pytest.approx(1.3)

    def test_spread_falls_back_to_twice_std(self) -> None:
        assert spread({"utility_std": 0.25}) == pytest.approx(0.5)

    def test_spread_defaults_to_zero_when_absent(self) -> None:
        assert spread({}) == 0.0

    def test_quantile_endpoints_and_interpolation(self) -> None:
        values = [0.0, 1.0, 2.0, 3.0]
        assert quantile(values, 0.0) == 0.0
        assert quantile(values, 1.0) == 3.0
        assert quantile(values, 0.5) == pytest.approx(1.5)
        assert quantile(values, 0.25) == pytest.approx(0.75)

    def test_quantile_of_empty_is_nan(self) -> None:
        assert quantile([], 0.5) != quantile([], 0.5)  # NaN != NaN


class TestFillProbability:
    def test_certain_when_accept_rate_is_one(self) -> None:
        result = fill_probability(1.0)
        assert result["success_probability"] == pytest.approx(1.0)
        assert result["expected_batches"] == pytest.approx(1.0)

    def test_impossible_when_accept_rate_is_zero(self) -> None:
        result = fill_probability(0.0)
        assert result["success_probability"] == pytest.approx(0.0)
        assert result["expected_batches"] == pytest.approx(3.0)

    def test_accept_rate_0366_is_marginal(self) -> None:
        """探针口径接受率 36.6% ⇒ 3 批内收满只有约 70%（**不是**"几乎必然"）。"""
        result = fill_probability(0.366)
        assert result["success_probability"] == pytest.approx(0.6953, abs=1e-3)
        assert 2.5 < result["expected_batches"] < 2.8

    def test_predicted_attempts_reproduce_the_observed_82(self) -> None:
        """交叉验证（重要）：训练侧实测接受率 33.8% ⇒ 预测 50 步需 80.6 次尝试。

        第一轮 `training_diagnostics.jsonl` 实测 = 50 次更新 + 32 次重试 = **82 次尝试**。
        误差 1.7% ⇒ 这个二项模型与观测吻合，可用来预算"提高 tolerance 要多付多少 rollout"。
        """
        q = fill_probability(0.338)["success_probability"]
        assert 50 / q == pytest.approx(80.6, abs=1.0)

    def test_tightening_tolerance_doubles_rollout_cost(self) -> None:
        """接受率 33.8% → 25% 时，每步尝试数 1.61 → 2.85（近乎翻倍）。"""
        assert 1 / fill_probability(0.338)["success_probability"] == pytest.approx(1.61, abs=0.02)
        assert 1 / fill_probability(0.25)["success_probability"] == pytest.approx(2.85, abs=0.05)

    def test_probability_decreases_with_tolerance_style_threshold(self) -> None:
        probs = [fill_probability(p)["success_probability"] for p in (0.5, 0.4, 0.3, 0.2)]
        assert probs == sorted(probs, reverse=True)

    def test_rejects_out_of_range_and_bad_shapes(self) -> None:
        with pytest.raises(ValueError):
            fill_probability(1.5)
        with pytest.raises(ValueError):
            fill_probability(0.5, prompts_per_batch=0)
        with pytest.raises(ValueError):
            fill_probability(0.5, max_batches=0)
        with pytest.raises(ValueError):
            fill_probability(0.5, target=0)

    def test_unreachable_target_is_zero(self) -> None:
        result = fill_probability(1.0, prompts_per_batch=1, max_batches=2, target=5)
        assert result["success_probability"] == 0.0


# ── load_labels ────────────────────────────────────────────────────────


class TestLoadLabels:
    def test_reads_jsonl_skipping_blank_lines(self, tmp_path) -> None:
        path = tmp_path / "labels.jsonl"
        rows = [label(1, utility_min=0.0, utility_max=1.0), label(2, utility_min=0.5, utility_max=0.5)]
        path.write_text(
            "\n".join([json.dumps(rows[0]), "", json.dumps(rows[1])]) + "\n",
            encoding="utf-8",
        )
        loaded = load_labels(path)
        assert [row["task_id"] for row in loaded] == [1, 2]

    def test_rejects_bad_json(self, tmp_path) -> None:
        path = tmp_path / "labels.jsonl"
        path.write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="不是合法 JSON"):
            load_labels(path)

    def test_rejects_non_object_line(self, tmp_path) -> None:
        path = tmp_path / "labels.jsonl"
        path.write_text("[1, 2]\n", encoding="utf-8")
        with pytest.raises(ValueError, match="不是 JSON 对象"):
            load_labels(path)

    def test_rejects_empty_file(self, tmp_path) -> None:
        path = tmp_path / "labels.jsonl"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="没有任何标签行"):
            load_labels(path)


# ── analyze ────────────────────────────────────────────────────────────


def _mixed_pool() -> list[dict]:
    """一个可控的混合池：大 spread / 小 spread / 零方差，各若干。"""
    rows: list[dict] = []
    tid = 0
    # 6 道大 spread（1 成功 + 3 失败 ⇒ spread ≈ 1）
    for _ in range(6):
        tid += 1
        rows.append(label(tid, utility_min=-0.85, utility_max=1.0, pass_count=1, pass_rate=0.25))
    # 2 道小 spread（近似退化：spread = 0.02 ⇒ std = 0.01）
    for _ in range(2):
        tid += 1
        rows.append(label(tid, utility_min=0.19, utility_max=0.21, utility_std=0.01))
    # 4 道零方差（constant_reward，未保留）
    for _ in range(4):
        tid += 1
        rows.append(
            label(
                tid,
                utility_min=1.0,
                utility_max=1.0,
                utility_std=0.0,
                pass_count=4,
                pass_rate=1.0,
                varying=False,
                eligible=False,
            )
        )
    return rows


class TestAnalyze:
    def test_totals_and_rates(self) -> None:
        summary = analyze(_mixed_pool())
        assert summary["totals"]["tasks"] == 12
        assert summary["totals"]["varying"] == 8
        assert summary["totals"]["kept"] == 8
        assert summary["totals"]["varying_rate"] == pytest.approx(8 / 12)

    def test_near_degenerate_split(self) -> None:
        summary = analyze(_mixed_pool(), near_degenerate_std=0.05)
        near = summary["near_degenerate"]
        assert near["tasks"] == 2
        assert near["rate_of_varying"] == pytest.approx(2 / 8)
        assert near["meaningful_tasks"] == 6
        assert near["meaningful_rate_of_varying"] == pytest.approx(6 / 8)

    def test_spread_quantiles_only_over_varying(self) -> None:
        summary = analyze(_mixed_pool())
        # 零方差题被排除 ⇒ 最小 spread 应是 0.02 而不是 0
        assert summary["spread_quantiles"]["p00"] == pytest.approx(0.02)
        assert summary["spread_quantiles"]["p100"] == pytest.approx(1.85)

    def test_tolerance_sweep_is_monotone_in_survivors(self) -> None:
        summary = analyze(_mixed_pool())
        survivors = [row["surviving_tasks"] for row in summary["tolerance_sweep"]]
        assert survivors == sorted(survivors, reverse=True)
        # 1e-8 时 8 道都活着；0.05 时只剩 6 道大 spread
        assert summary["tolerance_sweep"][0]["surviving_tasks"] == 8
        at_005 = next(row for row in summary["tolerance_sweep"] if row["tolerance"] == 0.05)
        assert at_005["surviving_tasks"] == 6

    def test_sweep_reports_fill_cost(self) -> None:
        summary = analyze(_mixed_pool())
        at_005 = next(row for row in summary["tolerance_sweep"] if row["tolerance"] == 0.05)
        # 6/12 = 50% 接受率 ⇒ 3 批内收满 4 组的概率应明显低于 1
        assert at_005["accept_rate"] == pytest.approx(0.5)
        assert at_005["success_probability"] == pytest.approx(fill_probability(0.5)["success_probability"])

    def test_by_pass_count_groups_by_successes(self) -> None:
        summary = analyze(_mixed_pool())
        assert summary["by_pass_count"]["1"]["tasks"] == 6
        assert summary["by_pass_count"]["1"]["varying_rate"] == pytest.approx(1.0)
        # pass_count=4 的组是零方差 ⇒ varying 率 0
        assert summary["by_pass_count"]["4"]["varying_rate"] == pytest.approx(0.0)

    def test_pass_bands_count_eligible(self) -> None:
        summary = analyze(_mixed_pool())
        full = summary["pass_bands"][0]
        assert full["band"] == [0.0, 1.0]
        assert full["tasks"] == 12
        assert full["eligible_tasks"] == 8

    def test_current_fill_uses_overall_varying_rate(self) -> None:
        summary = analyze(_mixed_pool())
        assert summary["current_fill"]["success_probability"] == pytest.approx(
            fill_probability(8 / 12)["success_probability"]
        )

    def test_rejects_empty_labels(self) -> None:
        with pytest.raises(ValueError, match="labels 为空"):
            analyze([])


# ── verdict / render ───────────────────────────────────────────────────


class TestVerdict:
    def test_mostly_degenerate_says_switch_lever(self) -> None:
        rows = [label(i, utility_min=0.19, utility_max=0.21, utility_std=0.01) for i in range(1, 11)]
        head, lines = verdict(analyze(rows, near_degenerate_std=0.05))
        assert head == "换杠杆"
        assert any("std 归一化" in line for line in lines)

    def test_mostly_meaningful_says_raise_lr(self) -> None:
        rows = [label(i, utility_min=-0.85, utility_max=1.0, pass_count=1, pass_rate=0.25) for i in range(1, 11)]
        head, lines = verdict(analyze(rows, near_degenerate_std=0.05))
        assert head == "提 lr 是对的"
        assert any("×4" in line for line in lines)

    def test_middle_ground_mentions_both(self) -> None:
        rows = [label(i, utility_min=-0.85, utility_max=1.0) for i in range(1, 7)]
        rows += [label(i, utility_min=0.19, utility_max=0.21, utility_std=0.01) for i in range(7, 10)]
        head, lines = verdict(analyze(rows, near_degenerate_std=0.05))
        assert head == "两者都值得做，但先提 lr"
        assert any("std 归一化" in line for line in lines)

    def test_warns_about_tolerance_cost(self) -> None:
        rows = [label(i, utility_min=-0.85, utility_max=1.0) for i in range(1, 13)]
        rows += [label(i, utility_min=1.0, utility_max=1.0, varying=False, eligible=False) for i in range(13, 25)]
        _, lines = verdict(analyze(rows))
        assert any("提高 tolerance 的代价" in line for line in lines)


class TestRender:
    def test_contains_all_sections(self) -> None:
        text = render(analyze(_mixed_pool()))
        for heading in (
            "# 探针信号尺度分析",
            "## 1. 总量",
            "## 2. 有效组的 spread / std 分布（关键）",
            "## 3. 判读",
            "## 4. tolerance 扫描（提高准入阈值的代价）",
            "## 5. 按组内成功条数看 spread",
            "## 6. 静态 pass_rate 带（只作「换训练集」的备选口径）",
        ):
            assert heading in text

    def test_states_the_static_filter_is_not_a_recommendation(self) -> None:
        text = render(analyze(_mixed_pool()))
        assert "不是**建议改训练集" in text

    def test_marks_the_current_tolerance_row(self) -> None:
        text = render(analyze(_mixed_pool()))
        assert "←当前" in text


# ── CLI ────────────────────────────────────────────────────────────────


class TestMain:
    def test_writes_report_and_returns_zero(self, tmp_path, capsys) -> None:
        labels_path = tmp_path / "labels.jsonl"
        labels_path.write_text(
            "\n".join(json.dumps(row) for row in _mixed_pool()) + "\n", encoding="utf-8"
        )
        out_path = tmp_path / "nested" / "signal_scale.md"
        code = main([str(labels_path), "--out", str(out_path)])
        assert code == 0
        assert "探针信号尺度分析" in capsys.readouterr().out
        assert out_path.is_file()
        assert "## 3. 判读" in out_path.read_text(encoding="utf-8")

    def test_json_mode_emits_parseable_json(self, tmp_path, capsys) -> None:
        labels_path = tmp_path / "labels.jsonl"
        labels_path.write_text(
            "\n".join(json.dumps(row) for row in _mixed_pool()) + "\n", encoding="utf-8"
        )
        code = main([str(labels_path), "--json"])
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["totals"]["tasks"] == 12

    def test_missing_file_returns_two(self, tmp_path, capsys) -> None:
        code = main([str(tmp_path / "nope.jsonl")])
        assert code == 2
        assert "读取失败" in capsys.readouterr().err
