"""`scripts/audit_m10_step2_binding.py` 的「处境成立」判据固定。

钉死四件事：

1. **S1 只在 gold 名次 >20 或不在召回内时才算「必须翻页」**。
2. **S2 只在 gold 名次 2–20 时才算「存在可比较的对手」**；第 1 名不算。
3. **S3 只在第 1 名最低价 > 预算时才算「必须排除」**，区间价用最低价。
4. 拿不到预算 / 拿不到搜索结果时一律判「未成立」，不猜。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import audit_m10_step2_binding as module  # noqa: E402


def rows(*prices: float) -> list[dict]:
    return [
        {"rank": index + 1, "asin": f"A{index:011d}", "low": price}
        for index, price in enumerate(prices)
    ]


class PagingBindingTest(unittest.TestCase):
    def test_rank_beyond_page_one_is_binding(self) -> None:
        binding, _ = module.binding_of("S1_paging", 42, rows(), 100.0)
        self.assertTrue(binding)

    def test_rank_20_is_not_binding(self) -> None:
        binding, _ = module.binding_of("S1_paging", 20, rows(), 100.0)
        self.assertFalse(binding)

    def test_rank_21_is_binding(self) -> None:
        binding, _ = module.binding_of("S1_paging", 21, rows(), 100.0)
        self.assertTrue(binding)

    def test_missing_from_recall_is_binding(self) -> None:
        binding, _ = module.binding_of("S1_paging", None, rows(), 100.0)
        self.assertTrue(binding)


class CompareBindingTest(unittest.TestCase):
    def test_rank_two_to_five_is_binding(self) -> None:
        for rank in (2, 3, 5):
            binding, _ = module.binding_of("S2_compare", rank, rows(), 100.0)
            self.assertTrue(binding, rank)

    def test_rank_one_is_not_binding(self) -> None:
        binding, _ = module.binding_of("S2_compare", 1, rows(), 100.0)
        self.assertFalse(binding)

    def test_rank_beyond_page_one_is_not_compare_problem(self) -> None:
        binding, _ = module.binding_of("S2_compare", 30, rows(), 100.0)
        self.assertFalse(binding)

    def test_missing_from_recall_is_not_compare_problem(self) -> None:
        binding, _ = module.binding_of("S2_compare", None, rows(), 100.0)
        self.assertFalse(binding)


class BudgetBindingTest(unittest.TestCase):
    def test_top1_over_budget_is_binding(self) -> None:
        binding, reason = module.binding_of("S3_budget_exclude", 3, rows(201.0, 90.0), 100.0)
        self.assertTrue(binding)
        self.assertIn("201", reason)

    def test_top1_within_budget_is_not_binding(self) -> None:
        binding, _ = module.binding_of("S3_budget_exclude", 3, rows(90.0, 80.0), 100.0)
        self.assertFalse(binding)

    def test_no_budget_is_not_binding(self) -> None:
        binding, _ = module.binding_of("S3_budget_exclude", 3, rows(201.0), None)
        self.assertFalse(binding)

    def test_no_search_rows_is_not_binding(self) -> None:
        binding, _ = module.binding_of("S3_budget_exclude", 3, [], 100.0)
        self.assertFalse(binding)

    def test_unreadable_price_is_not_binding(self) -> None:
        binding, _ = module.binding_of(
            "S3_budget_exclude", 3, [{"rank": 1, "asin": "A00000000001", "low": None}], 100.0
        )
        self.assertFalse(binding)


class TeacherQueryTest(unittest.TestCase):
    def test_queries_are_collected_in_order(self) -> None:
        row = {
            "steps": [
                {"tool_name": "search_products", "parameters": {"query": "木梳 礼盒"}},
                {"tool_name": "open_product", "parameters": {"asin": "A1"}},
                {"tool_name": "search_products", "parameters": {"query": "红色 梳子"}},
            ]
        }
        self.assertEqual(module.teacher_queries(row), ["木梳 礼盒", "红色 梳子"])

    def test_empty_query_is_skipped(self) -> None:
        row = {"steps": [{"tool_name": "search_products", "parameters": {"query": ""}}]}
        self.assertEqual(module.teacher_queries(row), [])


if __name__ == "__main__":
    unittest.main()
