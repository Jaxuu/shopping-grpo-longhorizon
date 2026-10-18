"""`scripts/analyze_v5_gradient_share.py` 的语义固定。

钉死四件事：

1. **全 wrong 且 ms 全同的组不产生梯度**（v4 与 v5 都会被 dynamic_sampling 丢弃）。
2. **全 wrong 且 ms 有差异的组进入「回收组」**，并计入份额分子。
3. **混合组只进分母**（v5 不改它们的 advantage）。
4. **份额 = 回收组权重 / 全部 kept 组权重**，与手算一致。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import analyze_v5_gradient_share as module  # noqa: E402


def row(
    task_id: object,
    reward_type: str,
    *,
    match_score: float = 0.0,
    reward: float | None = None,
    required: object = None,
    actual: object = None,
    length: float | None = None,
) -> dict:
    if reward is None:
        reward = 1.0 if reward_type == "gold_purchase" else -0.85
    payload = {
        "task_id": task_id,
        "terminal_result": {
            "reward_detail": {
                "reward_type": reward_type,
                "weighted_score": match_score,
                "terminal_utility": reward,
                "hard_gates": {
                    "budget": {"status": "fail", "required": required, "actual": actual}
                },
            }
        },
    }
    if length is not None:
        payload["response_tokens"] = length
    return payload


class CollectGroupsTest(unittest.TestCase):
    def test_single_sample_tasks_are_skipped(self):
        groups = module.collect_groups(
            [row(1, "wrong_purchase", match_score=0.5), row(2, "gold_purchase")]
        )
        self.assertEqual(groups, [])

    def test_groups_by_task_id(self):
        groups = module.collect_groups(
            [
                row(1, "wrong_purchase", match_score=0.1),
                row(1, "wrong_purchase", match_score=0.2),
                row(2, "gold_purchase"),
                row(2, "gold_purchase"),
            ]
        )
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]["samples"]), 2)


class LengthTest(unittest.TestCase):
    def test_explicit_token_field(self):
        self.assertEqual(module._length({"response_tokens": 1234}), 1234.0)

    def test_falls_back_to_step_count(self):
        self.assertEqual(module._length({"steps": [1, 2, 3]}), 3.0)

    def test_falls_back_to_one(self):
        self.assertEqual(module._length({}), 1.0)


class GradientShareTest(unittest.TestCase):
    def test_identical_all_wrong_group_contributes_nothing(self):
        rows = [row(1, "wrong_purchase", match_score=0.5) for _ in range(4)]
        report = module.gradient_share(module.collect_groups(rows), "A_ms_only_wide")
        self.assertEqual(report["kept_groups"], 0)
        self.assertEqual(report["recovered_groups"], 0)
        self.assertEqual(report["share"], 0.0)
        self.assertEqual(report["verdict"], "fail")

    def test_varying_all_wrong_group_is_recovered(self):
        rows = [
            row(1, "wrong_purchase", match_score=0.0),
            row(1, "wrong_purchase", match_score=0.5),
            row(1, "wrong_purchase", match_score=1.0),
            row(1, "wrong_purchase", match_score=0.5),
        ]
        report = module.gradient_share(module.collect_groups(rows), "A_ms_only_wide")
        self.assertEqual(report["recovered_groups"], 1)
        self.assertEqual(report["kept_groups"], 1)
        self.assertAlmostEqual(report["share"], 1.0)
        self.assertEqual(report["verdict"], "pass")
        self.assertGreater(report["median_recovered_spread"], 0.0)

    def test_mixed_group_only_enters_the_denominator(self):
        rows = [
            # 回收组：4 条全 wrong，ms 有差异
            row(1, "wrong_purchase", match_score=0.0),
            row(1, "wrong_purchase", match_score=0.5),
            row(1, "wrong_purchase", match_score=1.0),
            row(1, "wrong_purchase", match_score=0.5),
            # 混合组：2 gold + 2 wrong（ms=0）
            row(2, "gold_purchase"),
            row(2, "gold_purchase"),
            row(2, "wrong_purchase", match_score=0.0),
            row(2, "wrong_purchase", match_score=0.0),
        ]
        report = module.gradient_share(module.collect_groups(rows), "A_ms_only_wide")
        self.assertEqual(report["recovered_groups"], 1)
        self.assertEqual(report["kept_groups"], 2)
        # 手算：回收组 Σ|A| = 0.475；混合组 Σ|A| = 3.7 ⇒ share = 0.475/4.175
        self.assertAlmostEqual(report["share"], 0.475 / 4.175, places=6)
        self.assertEqual(report["verdict"], "pass")

    def test_tighter_cap_reduces_the_share(self):
        """保序候选（cap −0.66）的份额必然低于宽候选（cap −0.45）。"""
        rows = [
            row(1, "wrong_purchase", match_score=0.0),
            row(1, "wrong_purchase", match_score=0.5),
            row(1, "wrong_purchase", match_score=1.0),
            row(1, "wrong_purchase", match_score=0.5),
            row(2, "gold_purchase"),
            row(2, "gold_purchase"),
            row(2, "wrong_purchase", match_score=0.0),
            row(2, "wrong_purchase", match_score=0.0),
        ]
        groups = module.collect_groups(rows)
        wide = module.gradient_share(groups, "A_ms_only_wide")
        safe = module.gradient_share(groups, "B_ms_only_safe")
        self.assertLess(safe["share"], wide["share"])

    def test_overspend_reduces_the_driver(self):
        """同一 ms 下超支越多，回收组的组内跨度越小。"""
        tight = module.gradient_share(
            module.collect_groups(
                [
                    row(1, "wrong_purchase", match_score=0.0, required=100, actual=101),
                    row(1, "wrong_purchase", match_score=1.0, required=100, actual=101),
                ]
            ),
            "A_ms_only_wide",
        )
        loose = module.gradient_share(
            module.collect_groups(
                [
                    row(1, "wrong_purchase", match_score=0.0, required=100, actual=320),
                    row(1, "wrong_purchase", match_score=1.0, required=100, actual=320),
                ]
            ),
            "C_ms_x_prox",
        )
        self.assertGreater(tight["median_recovered_spread"], 0.0)
        self.assertGreater(loose["median_recovered_spread"], 0.0)
        self.assertGreater(
            tight["median_recovered_spread"], loose["median_recovered_spread"]
        )


class RenderTest(unittest.TestCase):
    def test_report_lists_every_candidate_and_the_thresholds(self):
        groups = module.collect_groups(
            [
                row(1, "wrong_purchase", match_score=0.0),
                row(1, "wrong_purchase", match_score=1.0),
            ]
        )
        payload = {
            "source": "synthetic.jsonl",
            "rows": 2,
            "groups": 1,
            "formulas": [module.gradient_share(groups, name) for name in module.CANDIDATES],
        }
        text = module.render(payload)
        self.assertIn("Gate 3", text)
        self.assertIn("梯度份额", text)
        for name in module.CANDIDATES:
            self.assertIn(name, text)
        self.assertIn("长度权重口径", text)


if __name__ == "__main__":
    unittest.main()
