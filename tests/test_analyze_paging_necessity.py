import unittest

from scripts.analyze_paging_necessity import (
    classify,
    format_detail,
    format_report,
    pages_needed,
    rank_band,
    summarize,
)

PAGE = 20


def _row(tools, reward_type="gold_purchase"):
    return {
        "steps": [{"tool_name": name} for name in tools],
        "terminal_result": {"reward_detail": {"reward_type": reward_type}},
    }


class PagesNeededTest(unittest.TestCase):
    def test_first_page_needs_no_paging(self):
        self.assertEqual(pages_needed(1, PAGE), 0)
        self.assertEqual(pages_needed(20, PAGE), 0)

    def test_one_page_covers_21_to_40(self):
        self.assertEqual(pages_needed(21, PAGE), 1)
        self.assertEqual(pages_needed(40, PAGE), 1)

    def test_second_page_starts_at_41(self):
        self.assertEqual(pages_needed(41, PAGE), 2)
        self.assertEqual(pages_needed(60, PAGE), 2)

    def test_unknown_rank_is_none_not_zero(self):
        # 0 会被误读成「不需要翻页」，必须区分「不需要」与「不知道」。
        self.assertIsNone(pages_needed(None, PAGE))


class ClassifyTest(unittest.TestCase):
    def test_visible_gold_without_paging_is_normal(self):
        item = classify(_row(["search_products", "open_product", "buy_now"]), 5, PAGE)
        self.assertEqual(item["group"], "not_needed")
        self.assertTrue(item["success"])

    def test_paging_when_gold_was_already_visible_is_flagged(self):
        item = classify(_row(["search_products", "next_page", "buy_now"]), 5, PAGE)
        self.assertEqual(item["group"], "not_needed_paged")

    def test_paging_once_reaches_rank_21_to_40(self):
        item = classify(_row(["search_products", "next_page", "buy_now"]), 33, PAGE)
        self.assertEqual(item["group"], "needed_reached")

    def test_paging_once_is_short_for_rank_41(self):
        item = classify(_row(["search_products", "next_page", "buy_now"]), 41, PAGE)
        self.assertEqual(item["group"], "needed_paged_short")

    def test_never_paging_a_deep_rank_is_a_missed_opportunity(self):
        item = classify(_row(["search_products", "buy_now"]), 87, PAGE)
        self.assertEqual(item["group"], "needed_not_paged")

    def test_missing_gold_reference_is_unknown(self):
        item = classify(_row(["search_products", "buy_now"]), None, PAGE)
        self.assertEqual(item["group"], "rank_unknown")


class RankBandTest(unittest.TestCase):
    def test_bands_follow_page_boundaries(self):
        self.assertEqual(rank_band(10, PAGE), "1-20")
        self.assertEqual(rank_band(30, PAGE), "1 页")
        self.assertEqual(rank_band(50, PAGE), "2 页")
        self.assertEqual(rank_band(70, PAGE), "3-4 页")
        self.assertEqual(rank_band(130, PAGE), "5-7 页")
        self.assertEqual(rank_band(200, PAGE), "8 页以上")


class SummarizeTest(unittest.TestCase):
    def test_groups_are_counted_and_rate_is_per_group(self):
        items = [
            classify(_row(["search_products", "buy_now"]), 5, PAGE),
            classify(_row(["search_products", "next_page", "buy_now"]), 30, PAGE),
            classify(_row(["search_products", "next_page", "buy_now"]), 30, PAGE),
        ]
        items[2]["success"] = False
        stats = summarize(items, PAGE)
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["groups"]["not_needed"]["count"], 1)
        self.assertEqual(stats["groups"]["needed_reached"]["count"], 2)
        self.assertEqual(stats["groups"]["needed_reached"]["success"], 1)

    def test_report_states_how_many_needed_paging_and_how_many_reached(self):
        items = [
            classify(_row(["search_products", "next_page", "buy_now"]), 30, PAGE),
            classify(_row(["search_products", "buy_now"]), 87, PAGE),
            classify(_row(["search_products", "next_page", "buy_now"]), 5, PAGE),
        ]
        text = format_report("demo", summarize(items, PAGE))
        self.assertIn("该翻页的题 2 道：翻到 1、页数不够 0、完全没翻 1", text)
        self.assertIn("白翻", text)


class DetailTest(unittest.TestCase):
    def test_detail_counts_same_category_decoys_on_the_first_page(self):
        item = classify(_row(["search_products", "buy_now"]), 87, PAGE)
        item["gold"] = "G"
        item["purchased"] = "A1"
        item["first_page"] = ["A1", "A2", "B1"]
        text = format_detail([item], {"G": "耳机", "A1": "耳机", "A2": "耳机", "B1": "水杯"})
        self.assertIn("该翻却没看到 gold 的 1 道", text)
        self.assertIn("1/1 道的第一页里有同类商品", text)

    def test_detail_ignores_rows_that_never_needed_paging(self):
        item = classify(_row(["search_products", "buy_now"]), 5, PAGE)
        item["first_page"] = ["A1"]
        text = format_detail([item], {"A1": "耳机"})
        self.assertIn("0 道", text)


if __name__ == "__main__":
    unittest.main()
