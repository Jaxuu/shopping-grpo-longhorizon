"""`inspect_m10_paging_output.py` 的测试：四条检查各自能抓到该抓的东西。

重点是「不能太宽松」：一个检查若对坏轨迹也说 ok，那冒烟就等于没做。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (str(ROOT / "scripts"), str(ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import inspect_m10_paging_output as module  # noqa: E402

GOLD = "620296036484"
OTHER = "11111111"


def search_page(rows: list[tuple[int, str]], page: int = 1) -> str:
    lines = [
        "[SHOPPING_OBSERVATION_V2]",
        "page_type: search_results",
        f"Page {page} of 8 (Total results: 150)",
    ]
    lines += [f"{rank}|{asin}|9.0|品牌|类目|属性|标题" for rank, asin in rows]
    lines.append('可点击的按钮: ["Next >", "< Prev", "Back to Search"]')
    return "\n".join(lines)


def detail_page(asin: str = GOLD) -> str:
    return (
        "[SHOPPING_OBSERVATION_V2]\npage_type: product_detail\n"
        f"asin: {asin}\nprice: 9.0\n"
        '可点击的按钮: ["Buy Now", "< Prev", "Back to Search"]'
    )


def build_row(
    *,
    first_page_has_gold: bool = False,
    reward_type: str = "gold_purchase",
    purchased: str = GOLD,
    roles: list[str] | None = None,
) -> dict:
    page1 = search_page(
        [(1, GOLD) if first_page_has_gold else (1, OTHER), (2, OTHER), (3, OTHER)]
    )
    page2 = search_page([(21, OTHER), (22, GOLD), (23, OTHER)], page=2)
    steps = [
        {"step_index": 0, "tool_name": "search_products", "observation": page1},
        {"step_index": 1, "tool_name": "next_page", "observation": page2},
        {"step_index": 2, "tool_name": "open_product", "observation": detail_page()},
        {"step_index": 3, "tool_name": "buy_now", "observation": "购买已完成。"},
    ]
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for step in steps:
        messages.append({"role": "assistant", "content": "", "tool_calls": []})
        messages.append({"role": "tool", "content": step["observation"]})
    if roles is not None:
        messages = [{"role": role, "content": ""} for role in roles]
    return {
        "trajectory_id": "t-1",
        "task_id": 3,
        "messages": messages,
        "steps": steps,
        "terminal_result": {
            "reward_detail": {
                "reward_type": reward_type,
                "reward": 1.0,
                "purchase_success": True,
            },
            "goal": {"asin": GOLD},
            "purchase": {"asin": purchased},
        },
    }


class SearchPagesTest(unittest.TestCase):
    def test_records_rank_per_page(self) -> None:
        pages = module.search_pages(build_row()["steps"], GOLD)
        self.assertEqual([page["gold_rank"] for page in pages], [None, 22])
        self.assertEqual([page["rows"] for page in pages], [3, 3])

    def test_non_search_observations_are_skipped(self) -> None:
        pages = module.search_pages(build_row()["steps"], GOLD)
        self.assertEqual(len(pages), 2)


class CheckRolesTest(unittest.TestCase):
    def test_accepts_alternating(self) -> None:
        self.assertIsNone(module.check_roles(build_row()["messages"]))

    def test_rejects_missing_user(self) -> None:
        roles = ["system", "assistant", "tool"]
        self.assertIn("system→user", module.check_roles([{"role": r} for r in roles]))

    def test_rejects_two_tools_in_a_row(self) -> None:
        roles = ["system", "user", "assistant", "tool", "tool", "assistant"]
        self.assertIsNotNone(module.check_roles([{"role": r} for r in roles]))

    def test_rejects_tool_before_assistant(self) -> None:
        roles = ["system", "user", "tool", "assistant"]
        self.assertIsNotNone(module.check_roles([{"role": r} for r in roles]))


class InspectRowTest(unittest.TestCase):
    def test_clean_row_passes(self) -> None:
        report = module.inspect_row(build_row())
        self.assertEqual(report["problems"], [])
        self.assertEqual(report["found_on_page"], 2)
        self.assertEqual(report["gold_rank"], 22)

    def test_gold_on_first_page_fails(self) -> None:
        report = module.inspect_row(build_row(first_page_has_gold=True))
        self.assertTrue(any("第 1 页就出现 gold" in p for p in report["problems"]))

    def test_wrong_reward_type_fails(self) -> None:
        report = module.inspect_row(build_row(reward_type="wrong_purchase"))
        self.assertTrue(any("reward_type" in p for p in report["problems"]))

    def test_purchased_mismatch_fails(self) -> None:
        report = module.inspect_row(build_row(purchased=OTHER))
        self.assertTrue(any("≠ goal" in p for p in report["problems"]))

    def test_broken_messages_fail(self) -> None:
        report = module.inspect_row(build_row(roles=["system", "user", "tool"]))
        self.assertTrue(any("不成对" in p for p in report["problems"]))

    def test_no_search_page_fails(self) -> None:
        row = build_row()
        row["steps"] = [s for s in row["steps"] if s["tool_name"] not in {"search_products", "next_page"}]
        report = module.inspect_row(row)
        self.assertTrue(any("没有任何搜索页" in p for p in report["problems"]))


class RankBandTest(unittest.TestCase):
    def test_bands_are_one_based(self) -> None:
        self.assertEqual(module.rank_band(1), "1-10")
        self.assertEqual(module.rank_band(10), "1-10")
        self.assertEqual(module.rank_band(11), "11-20")
        self.assertEqual(module.rank_band(22), "21-30")
        self.assertEqual(module.rank_band(120), "111-120")


class SummarizeTest(unittest.TestCase):
    def test_counts_failures_and_histograms(self) -> None:
        reports = [
            module.inspect_row(build_row()),
            module.inspect_row(build_row(first_page_has_gold=True)),
        ]
        summary = module.summarize(reports)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["found_on_page"], {1: 1, 2: 1})
        self.assertEqual(summary["gold_rank_band"], {"1-10": 1, "21-30": 1})


class CliTest(unittest.TestCase):
    def test_returns_zero_when_all_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.jsonl"
            path.write_text(
                json.dumps(build_row(), ensure_ascii=False) + "\n", encoding="utf-8"
            )
            self.assertEqual(module.main([str(path), "--summary-only"]), 0)

    def test_returns_one_when_any_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.jsonl"
            path.write_text(
                json.dumps(build_row(first_page_has_gold=True), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(module.main([str(path), "--summary-only"]), 1)


if __name__ == "__main__":
    unittest.main()
