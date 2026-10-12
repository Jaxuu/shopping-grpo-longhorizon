"""Budget-parser coverage tests (M1-R1 / M1-R2, ``shopping-budget-parser-v4``)."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "environments" / "ShopSimulator" / "shop_env"
if str(SHOP_ENV) not in sys.path:
    sys.path.append(str(SHOP_ENV))

from web_agent_site.engine.constraints import (  # noqa: E402
    BUDGET_PARSER_VERSION,
    explicit_budget_from_instruction,
)

_SPEC = importlib.util.spec_from_file_location(
    "audit_solvability", ROOT / "scripts" / "audit_solvability.py"
)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


class BudgetParserCoverageTest(unittest.TestCase):
    """M1-R1: the new phrasings must parse."""

    def test_version(self):
        self.assertEqual(BUDGET_PARSER_VERSION, "shopping-budget-parser-v4")

    def test_chinese_numerals(self):
        self.assertEqual(explicit_budget_from_instruction("预算七十元"), 70)
        self.assertEqual(explicit_budget_from_instruction("预算五元"), 5)
        self.assertEqual(explicit_budget_from_instruction("预算一千二百元"), 1200)
        self.assertEqual(explicit_budget_from_instruction("预算是四百七十元"), 470)
        # shorthand that omits the trailing unit
        self.assertEqual(explicit_budget_from_instruction("价格两千五左右就好"), 2750)

    def test_colloquial_units(self):
        self.assertEqual(explicit_budget_from_instruction("预算三百块"), 300)
        self.assertEqual(explicit_budget_from_instruction("预算五百块钱"), 500)
        self.assertEqual(explicit_budget_from_instruction("价格不超过70块"), 70)

    def test_prefixless_range(self):
        self.assertEqual(explicit_budget_from_instruction("350 到 400 元之间"), 400)
        self.assertEqual(explicit_budget_from_instruction("价格在40到50之间的"), 50)
        self.assertEqual(explicit_budget_from_instruction("价格 10-20 元"), 20)

    def test_unit_price(self):
        self.assertEqual(explicit_budget_from_instruction("每平方米一千元"), 1000)

    def test_wan_shorthand(self):
        self.assertEqual(explicit_budget_from_instruction("预算在1万2千以内"), 12000)

    def test_non_ascii_whitespace(self):
        self.assertEqual(
            explicit_budget_from_instruction("预算\u00a0在\u00a0300\u00a0元以内"), 300
        )
        self.assertEqual(
            explicit_budget_from_instruction("预算在　300　元以内"), 300
        )


class BudgetParserGuardTest(unittest.TestCase):
    """M1-R2: non-price numbers must never be captured."""

    def test_open_ended_lower_bound(self):
        self.assertIsNone(explicit_budget_from_instruction("预算4k+"))

    def test_lower_bound_only(self):
        self.assertIsNone(explicit_budget_from_instruction("价格不低于2元"))
        self.assertIsNone(explicit_budget_from_instruction("克重不低于四百克"))

    def test_non_price_numbers_are_ignored(self):
        self.assertEqual(
            explicit_budget_from_instruction("要求SPF50+，重量有75克左右，价格在90元以内"),
            90,
        )
        self.assertEqual(
            explicit_budget_from_instruction("长不到60厘米，宽不到40厘米，价格不超过15元"),
            15,
        )
        self.assertIsNone(explicit_budget_from_instruction("显存不能低于40G，有推荐吗？"))

    def test_soft_preference_keeps_v3_tolerance(self):
        self.assertAlmostEqual(
            explicit_budget_from_instruction("价格控制在450元左右"), 495.0
        )


class BudgetParserCoverageRegressionTest(unittest.TestCase):
    """M1-R6: coverage must improve without regressing."""

    @classmethod
    def setUpClass(cls):
        cls.products = audit.load_products(audit.DEFAULT_PRODUCTS)
        cls.instructions = []
        for item in cls.products:
            for instruction in item.get("instructions") or []:
                if instruction.get("attributes"):
                    cls.instructions.append(instruction["instruction"])
        cls.dataset_ids = {
            name: audit.task_ids(path)
            for name, path in audit.DEFAULT_DATASETS.items()
            if path.exists()
        }

    def test_zero_coverage_regression_against_v3(self):
        regressions = []
        for ids in self.dataset_ids.values():
            for task_id in ids:
                text = self.instructions[task_id]
                if (
                    audit.budget_parser_v3(text) is not None
                    and explicit_budget_from_instruction(text) is None
                ):
                    regressions.append(task_id)
        self.assertEqual(regressions, [])

    def test_full_pool_parse_rate(self):
        parsed = total = 0
        for ids in self.dataset_ids.values():
            for task_id in ids:
                total += 1
                if explicit_budget_from_instruction(self.instructions[task_id]) is not None:
                    parsed += 1
        self.assertGreaterEqual(parsed / total, 0.90, f"{parsed}/{total}")

    def test_evaluation_parse_rate_does_not_regress(self):
        ids = self.dataset_ids["evaluation"]
        parsed = sum(
            1
            for task_id in ids
            if explicit_budget_from_instruction(self.instructions[task_id]) is not None
        )
        self.assertGreaterEqual(parsed / len(ids), 0.935)


if __name__ == "__main__":
    unittest.main()
