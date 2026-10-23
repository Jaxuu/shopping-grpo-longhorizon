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


def single_manifest() -> dict:
    """单阶段（M10-R v2）：只有一个 full 阶段，标签不能退化成原始的 "full"。"""
    return {
        "counts": {"rows": 100, "train": 90, "validation": 10, "evaluation_overlap": 0},
        "bucket_mode": "behavior",
        "stage_plan": "single",
        "buckets": {
            "exploratory": {"rows": 40, "train_task_ids": [1, 2], "validation_task_ids": [3]},
            "template": {"rows": 55, "train_task_ids": [4], "validation_task_ids": []},
            "other": {"rows": 5, "train_task_ids": [], "validation_task_ids": []},
        },
        "stages": {
            "full": {
                "buckets": ["exploratory", "template", "other"],
                "epochs": 1.0,
                "learning_rate": 5e-5,
                "train_rows": 3,
            }
        },
        "review_flags": {},
    }


class ShowSingleStageManifestTest(unittest.TestCase):
    def test_header_says_single_stage_not_cumulative(self):
        text = module.render(single_manifest())
        self.assertIn("## 阶段（单阶段：全量分布 1 个 epoch）", text)
        # 「累积」的说法在两段/三段课程里成立，单阶段下会让人以为还有第二阶段
        self.assertNotIn("累积：后一阶段包含前一阶段的全部桶", text)

    def test_full_stage_gets_a_human_label(self):
        text = module.render(single_manifest())
        self.assertIn("| 单阶段全量 | exploratory + template + other | 3 | 3 |", text)

    def test_behavior_mode_labels_do_not_leak_into_single_stage(self):
        """stage_plan=single 时标签取自 behavior 表，不能变成「A 学探索」。"""
        text = module.render(single_manifest())
        self.assertNotIn("A 学探索", text)


if __name__ == "__main__":
    unittest.main()
