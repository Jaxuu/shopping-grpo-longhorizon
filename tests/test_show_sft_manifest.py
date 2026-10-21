"""`show_sft_manifest.py` 的测试：桶与阶段的呈现口径。

要点：阶段是**累积**的（a ⊂ b ⊂ c），「本阶段新增」不能把累计行数直接当新增；
桶不是按轨迹类型分，而是按难度标签分。
"""

from __future__ import annotations

import unittest

from scripts import show_sft_manifest as module


def manifest() -> dict:
    return {
        "counts": {"rows": 100, "train": 90, "validation": 10, "evaluation_overlap": 0},
        "buckets": {
            "foundation": {"rows": 20, "train_task_ids": [1, 2], "validation_task_ids": [3]},
            "constraints": {"rows": 30, "train_task_ids": [4], "validation_task_ids": []},
            "strategy": {"rows": 50, "train_task_ids": [5, 6, 7], "validation_task_ids": [8]},
        },
        "stages": {
            "a": {"buckets": ["foundation"], "epochs": 1.0, "learning_rate": 1e-4, "train_rows": 2},
            "b": {
                "buckets": ["foundation", "constraints"],
                "epochs": 1.0,
                "learning_rate": 7e-5,
                "train_rows": 3,
            },
            "c": {
                "buckets": ["foundation", "constraints", "strategy"],
                "epochs": 1.0,
                "learning_rate": 5e-5,
                "train_rows": 6,
            },
        },
        "review_flags": {"search_heavy": [1, 2], "trajectory_long": []},
    }


class ShowManifestTest(unittest.TestCase):
    def test_bucket_shares_use_total_rows(self):
        text = module.render(manifest())
        self.assertIn("| 基础（simple） | 20 | 2 | 1 | 20.0% |", text)
        self.assertIn("| 策略（hard 或需换词/对比） | 50 | 3 | 1 | 50.0% |", text)

    def test_stage_increment_is_not_cumulative(self):
        text = module.render(manifest())
        # a 累计 2（新增 2）、b 累计 3（新增 1）、c 累计 6（新增 3）
        self.assertIn("| A 打基础 | foundation | 2 | 2 | 1.0 | 0.0001 |", text)
        self.assertIn("| B 加约束 | foundation + constraints | 3 | 1 | 1.0 | 7e-05 |", text)
        self.assertIn(
            "| C 加策略 | foundation + constraints + strategy | 6 | 3 | 1.0 | 5e-05 |", text
        )

    def test_review_flags_only_print_non_empty(self):
        text = module.render(manifest())
        self.assertIn("search_heavy: 2 题", text)
        self.assertNotIn("trajectory_long", text)


if __name__ == "__main__":
    unittest.main()
