"""`scripts/analyze_m10_step2_verification.py` 的判定口径固定。

钉死五件事（改动前先看这里）：

1. **「回头买」= 买到的那个在打开序列里出现过两次**（A→B→A 才下单）。
   不能用 `purchased != opened[-1]`——`buy_now` 只能买当前页面那个商品，
   所以该条件**永远为假**，会把 S2/S3 恒判成 0%，进而误关两条路。
2. **「排除超预算」用详情页最低价**：区间 `5.0 to 18.0` 且预算 20 不算超预算。
3. **价格优先读 `raw_observation`**：`observation` 会被投影截断成 `5.0 to…8.0`。
4. **必须拿成交价确认买了预算内的**，否则不算排除。
5. **读 `accepted.jsonl` 直接报错**，不静默给 0。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import analyze_m10_step2_verification as module  # noqa: E402

BUDGET = module._budget_parser()


def detail_observation(asin: str, price: str) -> str:
    return "\n".join(
        [
            "[SHOPPING_OBSERVATION_V2]",
            "page_type: product_detail",
            f"asin: {asin}",
            f"price: {price}",
        ]
    )


def search_observation(rows: list[tuple[int, str, str]]) -> str:
    lines = [
        "[SHOPPING_OBSERVATION_V2]",
        "page_type: search_results",
        "格式: rank|asin|price|brand|category|key_attributes|title",
    ]
    lines += [f"{rank}|{asin}|{price}|品牌|类目|属性|标题" for rank, asin, price in rows]
    return "\n".join(lines)


def open_step(asin: str, price: str, *, raw: str | None = None) -> dict:
    observation = detail_observation(asin, price)
    step = {
        "tool_name": "open_product",
        "parameters": {"asin": asin},
        "observation": observation,
    }
    if raw is not None:
        step["raw_observation"] = raw
    return step


def trajectory(
    *,
    steps: list[dict],
    purchased: str | None,
    purchase_price: float | None = 90.0,
    gold: str = "G00000000001",
    instruction: str = "买一件东西，价格不超过 100 元。",
    task_id: int = 1,
) -> dict:
    return {
        "task_id": task_id,
        "attempt_index": 0,
        "status": "done",
        "steps": steps,
        "initial_result": {"instruction": f"Instruction: {instruction}"},
        "terminal_result": {
            "goal": {"asin": gold},
            "purchase": (
                {"asin": purchased, "price": purchase_price} if purchased else {}
            ),
            "reward_detail": {"reward_type": "gold_purchase" if purchased else "max_steps"},
        },
    }


class PriceParsingTest(unittest.TestCase):
    def test_range_keeps_both_ends(self) -> None:
        self.assertEqual(module.parse_price_range("price: 5.0 to 18.0"), (5.0, 18.0))

    def test_single_value(self) -> None:
        self.assertEqual(module.parse_price_range("price: 9.0"), (9.0, 9.0))

    def test_thousands_separator(self) -> None:
        self.assertEqual(module.parse_price_range("price: 1,299.00"), (1299.0, 1299.0))

    def test_missing_price_line(self) -> None:
        self.assertIsNone(module.parse_price_range("page_type: search_results"))

    def test_raw_observation_wins_over_truncated(self) -> None:
        step = {
            "observation": "price: 5.0 to…8.0",
            "raw_observation": "price: 5.0 to 18.0",
        }
        self.assertIn("18.0", module.observation_text(step))


class BackBuyTest(unittest.TestCase):
    """S2：`buy_now` 只能买当前页面那个商品，所以证据是「它也出现在更早的打开里」。"""

    def test_reopen_then_buy_counts(self) -> None:
        # A → B → A → 买 A。旧口径（purchased != opened[-1]）会判 False。
        record = module.analyze(
            trajectory(
                steps=[
                    open_step("A00000000001", "90.0"),
                    open_step("B00000000002", "80.0"),
                    open_step("A00000000001", "90.0"),
                ],
                purchased="A00000000001",
            ),
            budget_parser=BUDGET,
        )
        self.assertTrue(record["bought_earlier_candidate"])
        self.assertTrue(record["compared_before_buy"])

    def test_buy_last_opened_is_not_back_buy(self) -> None:
        # A → B → 买 B：只是「买最后一个」，不是回头买。
        record = module.analyze(
            trajectory(
                steps=[
                    open_step("A00000000001", "90.0"),
                    open_step("B00000000002", "80.0"),
                ],
                purchased="B00000000002",
            ),
            budget_parser=BUDGET,
        )
        self.assertTrue(record["compared_before_buy"])
        self.assertFalse(record["bought_earlier_candidate"])

    def test_single_open_is_not_back_buy(self) -> None:
        record = module.analyze(
            trajectory(
                steps=[open_step("A00000000001", "90.0")],
                purchased="A00000000001",
            ),
            budget_parser=BUDGET,
        )
        self.assertFalse(record["compared_before_buy"])
        self.assertFalse(record["bought_earlier_candidate"])

    def test_reopening_same_item_is_not_comparison(self) -> None:
        record = module.analyze(
            trajectory(
                steps=[
                    open_step("A00000000001", "90.0"),
                    open_step("A00000000001", "90.0"),
                ],
                purchased="A00000000001",
            ),
            budget_parser=BUDGET,
        )
        self.assertFalse(record["compared_before_buy"])
        self.assertFalse(record["bought_earlier_candidate"])


class BudgetExclusionTest(unittest.TestCase):
    """S3：打开过「最低价都超预算」的候选，最终买了预算内的另一个。"""

    def test_exclude_expensive_then_buy_cheap(self) -> None:
        record = module.analyze(
            trajectory(
                steps=[
                    open_step("X00000000001", "201.0 to 300.0"),
                    open_step("G00000000001", "90.0"),
                ],
                purchased="G00000000001",
                purchase_price=90.0,
            ),
            budget_parser=BUDGET,
        )
        self.assertTrue(record["s3_judgeable"])
        self.assertEqual(record["over_budget_opened"], ["X00000000001"])
        self.assertTrue(record["excluded_over_budget"])

    def test_range_within_budget_is_not_over_budget(self) -> None:
        # 最低价 5.0 ≤ 预算 100 ⇒ 能挑到便宜规格，不算超预算。
        record = module.analyze(
            trajectory(
                steps=[
                    open_step("X00000000001", "5.0 to 300.0"),
                    open_step("G00000000001", "90.0"),
                ],
                purchased="G00000000001",
                purchase_price=90.0,
            ),
            budget_parser=BUDGET,
        )
        self.assertEqual(record["over_budget_opened"], [])
        self.assertFalse(record["excluded_over_budget"])

    def test_buying_the_expensive_one_is_not_exclusion(self) -> None:
        record = module.analyze(
            trajectory(
                steps=[open_step("X00000000001", "201.0 to 300.0")],
                purchased="X00000000001",
                purchase_price=250.0,
            ),
            budget_parser=BUDGET,
        )
        self.assertFalse(record["excluded_over_budget"])

    def test_purchase_over_budget_is_not_exclusion(self) -> None:
        record = module.analyze(
            trajectory(
                steps=[
                    open_step("X00000000001", "201.0 to 300.0"),
                    open_step("Y00000000002", "150.0"),
                ],
                purchased="Y00000000002",
                purchase_price=150.0,
            ),
            budget_parser=BUDGET,
        )
        self.assertFalse(record["bought_in_budget"])
        self.assertFalse(record["excluded_over_budget"])

    def test_no_budget_in_instruction_is_unjudgeable(self) -> None:
        record = module.analyze(
            trajectory(
                steps=[
                    open_step("X00000000001", "201.0 to 300.0"),
                    open_step("G00000000001", "90.0"),
                ],
                purchased="G00000000001",
                instruction="随便买一件红色的东西。",
            ),
            budget_parser=BUDGET,
        )
        self.assertFalse(record["s3_judgeable"])
        self.assertFalse(record["excluded_over_budget"])

    def test_loose_metric_when_top_rank_is_over_budget(self) -> None:
        record = module.analyze(
            trajectory(
                steps=[
                    {
                        "tool_name": "search_products",
                        "parameters": {"query": "x"},
                        "observation": search_observation(
                            [(1, "X00000000001", "201.0 to 300.0"), (2, "G00000000001", "90.0")]
                        ),
                    },
                    open_step("G00000000001", "90.0"),
                ],
                purchased="G00000000001",
                purchase_price=90.0,
            ),
            budget_parser=BUDGET,
        )
        self.assertTrue(record["top_over_budget"])
        self.assertTrue(record["avoided_top_over_budget"])
        # 没有打开贵的那个 ⇒ 严格口径不算「排除」证据
        self.assertFalse(record["excluded_over_budget"])


class PagingTest(unittest.TestCase):
    def test_next_page_detected(self) -> None:
        record = module.analyze(
            trajectory(
                steps=[
                    {"tool_name": "search_products", "parameters": {}, "observation": "x"},
                    {"tool_name": "next_page", "parameters": {}, "observation": "x"},
                    open_step("G00000000001", "90.0"),
                ],
                purchased="G00000000001",
            ),
            budget_parser=BUDGET,
        )
        self.assertTrue(record["used_next_page"])

    def test_many_steps_without_next_page(self) -> None:
        record = module.analyze(
            trajectory(
                steps=[
                    {"tool_name": "search_products", "parameters": {}, "observation": "x"},
                    {"tool_name": "search_products", "parameters": {}, "observation": "x"},
                    open_step("G00000000001", "90.0"),
                ],
                purchased="G00000000001",
            ),
            budget_parser=BUDGET,
        )
        self.assertFalse(record["used_next_page"])


class GateTest(unittest.TestCase):
    def test_s3_gate_uses_judgeable_denominator(self) -> None:
        tiers = {1: "S3_budget_exclude", 2: "S3_budget_exclude"}
        rows = [
            trajectory(
                task_id=1,
                steps=[
                    open_step("X00000000001", "201.0 to 300.0"),
                    open_step("G00000000001", "90.0"),
                ],
                purchased="G00000000001",
                purchase_price=90.0,
            ),
            trajectory(
                task_id=2,
                steps=[open_step("G00000000001", "90.0")],
                purchased="G00000000001",
                instruction="随便买一件。",  # 无预算 ⇒ 不可判定
            ),
        ]
        report = module.summarize(rows, tiers, budget_parser=BUDGET)
        rate, hit, total = module._gate_entry(report, "S3_budget_exclude")
        self.assertEqual((hit, total), (1, 1))
        self.assertEqual(rate, 1.0)

    def test_accepted_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "accepted.jsonl"
            path.write_text(json.dumps({"task_id": 1}) + "\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                module.main([str(path)])


if __name__ == "__main__":
    unittest.main()
