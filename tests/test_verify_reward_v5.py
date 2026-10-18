"""`scripts/verify_reward_v5.py` 的语义固定。

钉死四件事（改动前先看这里）：

1. **非 `wrong_purchase` 的行必须逐位不变**——v5 只许改一个分支。
2. **变化行数必须恰等于 `wrong_purchase` 行数**，多一条就是误伤。
3. **`driver = 0` 必须退化到 −0.85**（v4 常数），否则最坏情形变了。
4. **strict success 与 `purchase_success` 必须前后一致**——
   这是「改奖励不破坏评测可比性」的唯一凭据。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import verify_reward_v5 as module  # noqa: E402


def row(
    task_id: object,
    reward_type: str,
    reward: float,
    *,
    match_score: float = 0.0,
    required: object = None,
    actual: object = None,
    purchase_success: bool | None = None,
    nested: bool = True,
) -> dict:
    if purchase_success is None:
        purchase_success = reward_type in {
            "gold_purchase",
            "valid_alternative_purchase",
        }
    detail = {
        "reward_type": reward_type,
        "reward_version": "shopsimulator-reward-v4",
        "reward_valid": reward_type != "reward_unverifiable",
        "purchase_success": purchase_success,
        "terminal_utility": reward,
        "weighted_score": match_score,
        "hard_gates": {
            "category": {"status": "fail", "required": "x", "actual": "y"},
            "budget": {"status": "fail", "required": required, "actual": actual},
        },
    }
    if nested:
        return {"task_id": task_id, "terminal_result": {"reward_detail": detail}}
    return {"task_id": task_id, "reward_detail": detail}


class ShapedRewardTest(unittest.TestCase):
    def test_zero_driver_is_the_v4_floor(self):
        self.assertEqual(module.shaped_reward("wrong_purchase", 0.0, 1.0), -0.85)

    def test_full_driver_is_capped(self):
        self.assertEqual(module.shaped_reward("wrong_purchase", 1.0, 1.0), -0.66)

    def test_midpoint_is_linear(self):
        self.assertAlmostEqual(
            module.shaped_reward("wrong_purchase", 0.5, 1.0), -0.85 + 0.20 * 0.5
        )

    def test_proximity_multiplies(self):
        self.assertAlmostEqual(
            module.shaped_reward("wrong_purchase", 1.0, 0.5), -0.85 + 0.20 * 0.5
        )


class ReplayTest(unittest.TestCase):
    def test_only_wrong_purchase_rows_change(self):
        rows = [
            row(1, "gold_purchase", 1.0, match_score=1.0),
            row(2, "partial_alternative_purchase", -0.1, match_score=0.36),
            row(3, "repeat_loop", -0.65),
            row(4, "wrong_purchase", -0.85, match_score=0.0),
            row(5, "wrong_purchase", -0.85, match_score=0.5),
        ]
        report = module.replay(rows)
        self.assertEqual(report["rows"], 5)
        self.assertEqual(report["wrong_rows"], 2)
        self.assertEqual(report["changed_rows"], 1)  # 只有 ms=0.5 那条变了
        self.assertEqual(report["untouched_wrong_rows"], 1)  # ms=0 退化为 -0.85
        self.assertEqual(report["violations"], [])
        self.assertEqual(report["success_before"], 1)
        self.assertEqual(report["success_after"], 1)
        self.assertEqual(report["strict_before"], 1)
        self.assertEqual(report["strict_after"], 1)

    def test_floor_row_is_untouched(self):
        report = module.replay([row(4, "wrong_purchase", -0.85, match_score=0.0)])
        record = report["wrong_records"][0]
        self.assertEqual(record["shaped"], -0.85)
        self.assertEqual(record["delta"], 0.0)

    def test_proximity_discounts_overspend(self):
        report = module.replay(
            [row(6, "wrong_purchase", -0.85, match_score=1.0, required=100.0, actual=200.0)]
        )
        record = report["wrong_records"][0]
        self.assertAlmostEqual(record["proximity"], 0.5)
        self.assertAlmostEqual(record["shaped"], -0.85 + 0.20 * 0.5)

    def test_band_violation_is_detected(self):
        original = module.CAP
        module.CAP = -0.95  # 人为把带顶压到地板以下
        try:
            report = module.replay([row(7, "wrong_purchase", -0.85, match_score=1.0)])
        finally:
            module.CAP = original
        self.assertTrue(any("越出 band" in item for item in report["violations"]))

    def test_flat_input_has_no_violation(self):
        """全 v4 落盘值（−0.85）时，重放必须报 0 变化。"""
        rows = [row(i, "wrong_purchase", -0.85, match_score=0.0) for i in range(4)]
        report = module.replay(rows)
        self.assertEqual(report["changed_rows"], 0)
        self.assertEqual(report["violations"], [])

    def test_nested_and_top_level_detail_both_work(self):
        rows = [
            row(1, "wrong_purchase", -0.85, match_score=0.5, nested=True),
            row(2, "wrong_purchase", -0.85, match_score=0.5, nested=False),
        ]
        report = module.replay(rows)
        self.assertEqual(report["rows"], 2)
        self.assertEqual(len({r["shaped"] for r in report["wrong_records"]}), 1)

    def test_mean_reward_moves_up(self):
        rows = [
            row(1, "gold_purchase", 1.0, match_score=1.0),
            row(2, "wrong_purchase", -0.85, match_score=0.8),
        ]
        report = module.replay(rows)
        self.assertGreater(
            report["mean_reward_after"], report["mean_reward_before"]
        )


class RenderTest(unittest.TestCase):
    def test_report_states_the_invariant(self):
        report = module.replay(
            [
                row(1, "gold_purchase", 1.0, match_score=1.0),
                row(2, "wrong_purchase", -0.85, match_score=0.4),
            ]
        )
        text = module.render(report, source="synthetic.jsonl")
        self.assertIn("Reward v5 数据重放", text)
        self.assertIn("strict success", text)
        self.assertIn("机械上移", text)
        self.assertIn("组内", text)  # 必须提醒「题间分辨率不产生梯度」

    def test_report_flags_violation(self):
        original = module.CAP
        module.CAP = -0.95
        try:
            report = module.replay([row(1, "wrong_purchase", -0.85, match_score=1.0)])
        finally:
            module.CAP = original
        text = module.render(report, source="synthetic.jsonl")
        self.assertIn("不通过", text)


class CliTest(unittest.TestCase):
    def test_exit_code_zero_on_clean_replay(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traj.jsonl"
            path.write_text(
                json.dumps(row(1, "wrong_purchase", -0.85, match_score=0.0)) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(module.main([str(path), "--json"]), 0)


if __name__ == "__main__":
    unittest.main()
