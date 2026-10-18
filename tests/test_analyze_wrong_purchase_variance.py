"""`scripts/analyze_wrong_purchase_variance.py` 的语义固定。

钉死四件事（改动前先看这里）：

1. **`ms = 0` 必须退化为 v4 的 −0.85**——v5 只许「往好里分」，不许动最坏情形。
2. **`cap` 是带顶（用 `min`）**，与 `partial_alternative_purchase` 同形。
   写成 `max` 会把所有题压到 cap 上，分辨率归零（本脚本第一版就踩过）。
3. **`budget_proximity` 在预算门 PASS 时必须等于 1.0**，
   否则「没超预算的失败」会被无端打折。
4. **Gate 2 的判据是组内标准差的「中位数」**，且单采样输入必须报
   `not_measured` 而不是悄悄按 0 判 fail。
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import analyze_wrong_purchase_variance as module  # noqa: E402


def wrong_row(
    task_id: object,
    match_score: float,
    *,
    required: object = 100.0,
    actual: object = 120.0,
    category: str = "pass",
    budget: str = "fail",
    asin: bool = False,
    nested: bool = False,
) -> dict:
    detail = {
        "reward_type": "wrong_purchase",
        "reward_version": "shopsimulator-reward-v4",
        "weighted_score": match_score,
        "terminal_utility": -0.85,
        "target_asin_match": asin,
        "hard_gates": {
            "category": {"status": category, "passed": category == "pass"},
            "budget": {
                "status": budget,
                "passed": budget == "pass",
                "required": required,
                "actual": actual,
            },
        },
    }
    if nested:
        return {"task_id": task_id, "terminal_result": {"reward_detail": detail}}
    return {"task_id": task_id, "reward_detail": detail}


class BudgetProximityTest(unittest.TestCase):
    def test_over_budget_is_discounted(self):
        # 超支 20% ⇒ 折扣到 1/1.2
        self.assertAlmostEqual(module.budget_proximity(100.0, 120.0), 100 / 120)

    def test_within_budget_is_neutral(self):
        # 预算门 PASS（actual < required）⇒ 1.0，不产生任何影响
        self.assertEqual(module.budget_proximity(100.0, 20.0), 1.0)

    def test_missing_or_degenerate_inputs_are_neutral(self):
        for required, actual in (
            (None, 120.0),
            (100.0, None),
            ("100", 120.0),
            (100.0, 0.0),
            (100.0, -5.0),
            (float("nan"), 120.0),
        ):
            with self.subTest(required=required, actual=actual):
                self.assertEqual(module.budget_proximity(required, actual), 1.0)


class WrongPurchaseRewardTest(unittest.TestCase):
    def test_zero_driver_degrades_to_v4_constant(self):
        for axis in ("match", "product"):
            with self.subTest(axis=axis):
                value = module.wrong_purchase_reward(
                    0.0, 0.5, scale=0.20, cap=-0.66, axis=axis
                )
                self.assertAlmostEqual(value, module.WRONG_FLOOR)

    def test_cap_is_an_upper_bound_using_min(self):
        # driver = 1.0 ⇒ −0.85 + 0.20 = −0.65 > cap(−0.66) ⇒ 必须被压回 −0.66
        value = module.wrong_purchase_reward(
            1.0, 1.0, scale=0.20, cap=-0.66, axis="product"
        )
        self.assertAlmostEqual(value, -0.66)

    def test_product_axis_multiplies_the_two_signals(self):
        by_match = module.wrong_purchase_reward(
            0.8, 0.25, scale=0.20, cap=-0.66, axis="match"
        )
        by_product = module.wrong_purchase_reward(
            0.8, 0.25, scale=0.20, cap=-0.66, axis="product"
        )
        # 同一 ms 下，超支越严重得分越低（−0.85 + 0.20×0.8×0.25 < −0.85 + 0.20×0.8）
        self.assertLess(by_product, by_match)

    def test_reward_never_above_floor_and_never_below_band(self):
        for axis in ("match", "product"):
            for ms in (0.0, 0.25, 0.5, 0.75, 1.0):
                for prox in (0.1, 0.5, 1.0):
                    value = module.wrong_purchase_reward(
                        ms, prox, scale=0.20, cap=-0.66, axis=axis
                    )
                    with self.subTest(axis=axis, ms=ms, prox=prox):
                        self.assertLessEqual(value, -0.66 + 1e-12)
                        self.assertGreaterEqual(value, module.WRONG_FLOOR - 1e-12)

    def test_unknown_axis_is_rejected(self):
        with self.assertRaises(ValueError):
            module.wrong_purchase_reward(0.5, 1.0, scale=0.2, cap=-0.66, axis="price")


class ExtractTest(unittest.TestCase):
    def test_reads_both_trajectory_shapes(self):
        rows = [wrong_row(1, 0.4), wrong_row(2, 0.6, nested=True)]
        records = module.extract_wrong_purchases(rows)
        self.assertEqual([r["task_id"] for r in records], ["1", "2"])
        self.assertEqual([round(r["match_score"], 2) for r in records], [0.4, 0.6])

    def test_ignores_other_reward_types(self):
        rows = [
            {"task_id": 1, "reward_detail": {"reward_type": "gold_purchase"}},
            {"task_id": 2},
            wrong_row(3, 0.5),
        ]
        self.assertEqual(len(module.extract_wrong_purchases(rows)), 1)

    def test_missing_budget_gate_falls_back_to_neutral_proximity(self):
        row = {
            "task_id": 9,
            "reward_detail": {
                "reward_type": "wrong_purchase",
                "weighted_score": 0.3,
                "hard_gates": {"category": {"status": "fail"}},
            },
        }
        record = module.extract_wrong_purchases([row])[0]
        self.assertEqual(record["proximity"], 1.0)
        self.assertEqual(record["budget_status"], None)

    def test_reward_type_counts_covers_unknown(self):
        counts = module.reward_type_counts(
            [
                {"reward_detail": {"reward_type": "gold_purchase"}},
                {"terminal_result": {}},
                {"reward_detail": {"reward_type": "wrong_purchase"}},
            ]
        )
        self.assertEqual(counts["gold_purchase"], 1)
        self.assertEqual(counts["unknown"], 1)
        self.assertEqual(counts["wrong_purchase"], 1)


class GroupSpreadTest(unittest.TestCase):
    def test_single_sample_input_has_no_groups(self):
        records = module.extract_wrong_purchases([wrong_row(i, 0.4) for i in range(5)])
        spread = module.group_spread(records, "match")
        self.assertEqual(spread["groups"], 0)
        self.assertIsNone(spread["median"])

    def test_identical_samples_give_zero_spread(self):
        records = module.extract_wrong_purchases(
            [wrong_row(7, 0.5) for _ in range(4)]
        )
        spread = module.group_spread(records, "match")
        self.assertEqual(spread["groups"], 1)
        self.assertAlmostEqual(spread["median"], 0.0)
        self.assertAlmostEqual(spread["zero_share"], 1.0)

    def test_population_stdev_is_used(self):
        # 4 个采样 0/0/0/1 ⇒ pstdev = 0.4330（样本标准差会是 0.5，这里不要那个）
        records = module.extract_wrong_purchases(
            [wrong_row(7, 0.0), wrong_row(7, 0.0), wrong_row(7, 0.0), wrong_row(7, 1.0)]
        )
        spread = module.group_spread(records, "match")
        self.assertAlmostEqual(spread["median"], 0.4330127, places=6)

    def test_product_axis_uses_match_times_proximity(self):
        # 同一 ms、不同超支幅度 ⇒ match 轴零方差，product 轴有方差
        rows = [
            wrong_row(7, 0.6, required=100.0, actual=100.0),
            wrong_row(7, 0.6, required=100.0, actual=300.0),
        ]
        records = module.extract_wrong_purchases(rows)
        self.assertAlmostEqual(module.group_spread(records, "match")["median"], 0.0)
        self.assertGreater(module.group_spread(records, "product")["median"], 0.15)


class Gate2VerdictTest(unittest.TestCase):
    def _verdict(self, median, groups=10):
        spread = {"axis": "product", "groups": groups, "median": median}
        return module.gate2_verdict(spread)["status"]

    def test_thresholds(self):
        self.assertEqual(self._verdict(0.06), "pass")
        self.assertEqual(self._verdict(0.05), "pass")
        self.assertEqual(self._verdict(0.049), "marginal")
        self.assertEqual(self._verdict(0.02), "marginal")
        self.assertEqual(self._verdict(0.019), "fail")
        self.assertEqual(self._verdict(0.0), "fail")

    def test_no_groups_is_not_measured(self):
        verdict = module.gate2_verdict({"axis": "match", "groups": 0, "median": None})
        self.assertEqual(verdict["status"], "not_measured")
        self.assertIn("探针", verdict["reason"])


class FormulaSummaryTest(unittest.TestCase):
    def test_wide_candidate_breaks_the_ladder(self):
        records = module.extract_wrong_purchases(
            [wrong_row(1, 0.0), wrong_row(2, 1.0, actual=100.0)]
        )
        wide = module.formula_summary(records, "A_ms_only_wide")
        self.assertIn("repeat_loop", wide["violations"])
        self.assertIn("max_steps", wide["violations"])

    def test_safe_candidates_preserve_the_ladder(self):
        records = module.extract_wrong_purchases(
            [wrong_row(1, 0.0), wrong_row(2, 1.0, actual=100.0)]
        )
        for name in ("B_ms_only_safe", "C_ms_x_prox"):
            with self.subTest(candidate=name):
                self.assertEqual(module.formula_summary(records, name)["violations"], [])

    def test_product_axis_has_more_resolution_than_match_axis(self):
        # 同一 ms 的不同超支幅度，product 轴才分得开
        rows = [
            wrong_row(1, 0.6, actual=110.0),
            wrong_row(2, 0.6, actual=300.0),
        ]
        records = module.extract_wrong_purchases(rows)
        self.assertEqual(module.formula_summary(records, "B_ms_only_safe")["distinct"], 1)
        self.assertEqual(module.formula_summary(records, "C_ms_x_prox")["distinct"], 2)

    def test_empty_input_is_safe(self):
        summary = module.formula_summary([], "C_ms_x_prox")
        self.assertEqual(summary["distinct"], 0)
        self.assertIsNone(summary["top"])


class RenderAndBuildTest(unittest.TestCase):
    def _write(self, rows) -> Path:
        directory = tempfile.mkdtemp()
        path = Path(directory) / "trajectories.jsonl"
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        return path

    def test_report_contains_all_sections(self):
        path = self._write([wrong_row(1, 0.0), wrong_row(2, 0.6), wrong_row(3, 0.8)])
        report = module.build_report(path)
        text = module.render_report(
            module.Counter(report["counts"]),
            report["records"],
            report["spreads"],
            report["verdicts"],
            report["formulas"],
            source=report["source"],
        )
        for section in ("## 1. 总览", "## 2. 内部解剖", "## 3. Gate 2", "## 4. v5", "## 5. 裁定"):
            self.assertIn(section, text)
        self.assertIn("not_measured", text)
        self.assertIn("C_ms_x_prox", text)
        # 单采样输入不得被误判成 fail
        self.assertNotIn("Gate 2（`ms×prox` 轴）= fail", text)

    def test_malformed_lines_are_skipped(self):
        directory = tempfile.mkdtemp()
        path = Path(directory) / "trajectories.jsonl"
        path.write_text(
            json.dumps(wrong_row(1, 0.5)) + "\nnot json\n\n[1,2,3]\n",
            encoding="utf-8",
        )
        report = module.build_report(path)
        self.assertEqual(report["rows"], 1)
        self.assertEqual(len(report["records"]), 1)

    def test_json_output_is_serialisable(self):
        path = self._write([wrong_row(1, 0.5)])
        report = module.build_report(path)
        payload = json.dumps(
            {k: v for k, v in report.items() if k != "records"}, default=str
        )
        self.assertIn("verdicts", payload)


if __name__ == "__main__":
    unittest.main()
