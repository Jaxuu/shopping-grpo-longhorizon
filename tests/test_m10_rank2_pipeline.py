"""`scripts/filter_m10_rank2.py` 与 `scripts/build_m10_rank2_targets.py` 的口径固定。

钉死四件事：

1. **名次取「所有搜索/翻页里最好的那个」**——只要曾经排过第 1 名，这条样本就教「开第一个」，必须剔掉。
2. **`rank=None`（从没出现）不算命中**，不能当成「名次很大」放进来。
3. **同一题只留一条**，避免一题刷满配额。
4. **选题均匀抽，不按检索名次预判**（离线预测已被 Step 2 证伪）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_m10_rank2_targets as builder  # noqa: E402
import filter_m10_rank2 as module  # noqa: E402


def search_observation(rows: list[tuple[int, str]]) -> str:
    lines = ["[SHOPPING_OBSERVATION_V2]", "page_type: search_results"]
    lines += [f"{rank}|{asin}|9.0|品牌|类目|属性|标题" for rank, asin in rows]
    return "\n".join(lines)


def trajectory(*observations: str) -> dict:
    return {
        "steps": [
            {"tool_name": "search_products", "parameters": {"query": "x"}, "observation": text}
            for text in observations
        ]
    }


class GoldRankTest(unittest.TestCase):
    def test_rank_from_single_page(self) -> None:
        row = trajectory(search_observation([(1, "A1"), (4, "G1")]))
        self.assertEqual(module.gold_rank(row, "G1"), 4)

    def test_best_rank_across_pages(self) -> None:
        row = trajectory(
            search_observation([(21, "G1")]),
            search_observation([(3, "G1")]),
        )
        self.assertEqual(module.gold_rank(row, "G1"), 3)

    def test_missing_gold_returns_none(self) -> None:
        row = trajectory(search_observation([(1, "A1")]))
        self.assertIsNone(module.gold_rank(row, "G1"))

    def test_non_search_steps_ignored(self) -> None:
        row = {"steps": [{"tool_name": "open_product", "parameters": {"asin": "G1"}, "observation": ""}]}
        self.assertIsNone(module.gold_rank(row, "G1"))


class KeepRuleTest(unittest.TestCase):
    def test_rank_two_is_kept(self) -> None:
        self.assertTrue(module.is_kept(2, 2))

    def test_rank_one_is_dropped(self) -> None:
        self.assertFalse(module.is_kept(1, 2))

    def test_none_is_dropped(self) -> None:
        self.assertFalse(module.is_kept(None, 2))

    def test_max_rank_caps(self) -> None:
        self.assertFalse(module.is_kept(25, 2, max_rank=20))
        self.assertTrue(module.is_kept(20, 2, max_rank=20))

    def test_zero_max_rank_means_unlimited(self) -> None:
        self.assertTrue(module.is_kept(99, 2, max_rank=0))


class OpenedAsinsTest(unittest.TestCase):
    def test_ordered_opens(self) -> None:
        row = {
            "steps": [
                {"tool_name": "open_product", "parameters": {"asin": "A1"}},
                {"tool_name": "buy_now", "parameters": {}},
                {"tool_name": "open_product", "parameters": {"asin": "A2"}},
            ]
        }
        self.assertEqual(module.opened_asins(row), ["A1", "A2"])


class TargetPickTest(unittest.TestCase):
    def test_excludes_used_ids(self) -> None:
        picked = builder.pick_task_ids(total=10, used={0, 5, 9}, limit=7, seed=1)
        self.assertEqual(sorted(picked), [1, 2, 3, 4, 6, 7, 8])

    def test_deterministic_for_same_seed(self) -> None:
        first = builder.pick_task_ids(total=100, used=set(), limit=10, seed=7)
        second = builder.pick_task_ids(total=100, used=set(), limit=10, seed=7)
        self.assertEqual(first, second)

    def test_different_seed_changes_pick(self) -> None:
        first = builder.pick_task_ids(total=100, used=set(), limit=10, seed=7)
        second = builder.pick_task_ids(total=100, used=set(), limit=10, seed=8)
        self.assertNotEqual(first, second)

    def test_limit_over_available_raises(self) -> None:
        with self.assertRaises(ValueError):
            builder.pick_task_ids(total=5, used={1, 2, 3}, limit=3, seed=1)


if __name__ == "__main__":
    unittest.main()
