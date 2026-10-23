import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from scripts.report_behavior_rate import (
    main,
    profile_row,
    summarize,
    summarize_actions,
)


def eval_row(task_id: int, calls: list[tuple[str, dict]]) -> dict:
    return {
        "task_id": task_id,
        "steps": [
            {"step_index": index, "tool_name": name, "parameters": parameters}
            for index, (name, parameters) in enumerate(calls)
        ],
    }


def pool_row(task_id: int, calls: list[tuple[str, dict]]) -> dict:
    return {
        "task_id": task_id,
        "messages": [
            {"role": "system", "content": "x"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call-{index}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(parameters)},
                    }
                    for index, (name, parameters) in enumerate(calls)
                ],
            },
        ],
    }


class BehaviorRateTest(unittest.TestCase):
    def test_next_page_counts_as_exploration(self):
        profile = profile_row(
            eval_row(1, [("search_products", {"query": "shirt"}), ("next_page", {})])
        )
        self.assertTrue(profile["paging"])
        self.assertEqual(profile["bucket"], "exploratory")

    def test_prev_page_is_navigation_not_exploration(self):
        """历史踩过的坑：把 prev_page 算成翻页会把 1,108 条模板误判成探索。"""
        profile = profile_row(
            eval_row(1, [("search_products", {"query": "shirt"}), ("prev_page", {})])
        )
        self.assertFalse(profile["paging"])
        self.assertEqual(profile["bucket"], "template")

    def test_two_distinct_queries_count_as_requery(self):
        profile = profile_row(
            eval_row(
                1,
                [
                    ("search_products", {"query": "blue shirt"}),
                    ("search_products", {"query": "Blue   Shirt"}),
                    ("search_products", {"query": "navy shirt"}),
                ],
            )
        )
        # 前两个归一化后相同，只有 2 个不同查询词
        self.assertEqual(profile["distinct_queries"], 2)
        self.assertTrue(profile["requery"])

    def test_same_query_twice_is_not_requery(self):
        profile = profile_row(
            eval_row(
                1,
                [
                    ("search_products", {"query": "shirt"}),
                    ("search_products", {"query": " shirt "}),
                    ("open_product", {"asin": "A1"}),
                ],
            )
        )
        self.assertFalse(profile["requery"])
        self.assertFalse(profile["multi_open"])
        self.assertEqual(profile["bucket"], "other")

    def test_two_distinct_asins_count_as_multi_open(self):
        profile = profile_row(
            eval_row(
                1,
                [
                    ("search_products", {"query": "shirt"}),
                    ("open_product", {"asin": "A1"}),
                    ("open_product", {"asin": "A2"}),
                ],
            )
        )
        self.assertTrue(profile["multi_open"])
        self.assertEqual(profile["bucket"], "exploratory")

    def test_pool_messages_format_gives_same_verdict(self):
        calls = [
            ("search_products", {"query": "shirt"}),
            ("open_product", {"asin": "A1"}),
            ("buy_now", {"asin": "A1"}),
        ]
        self.assertEqual(profile_row(eval_row(1, calls)), profile_row(pool_row(1, calls)))

    def test_think_does_not_count_as_a_tool_call(self):
        profile = profile_row(
            eval_row(1, [("think", {"note": "hmm"}), ("search_products", {"query": "shirt"})])
        )
        self.assertEqual(profile["tool_calls"], 1)

    def test_summarize_reports_rates(self):
        rows = [
            eval_row(1, [("search_products", {"query": "a"}), ("next_page", {})]),
            eval_row(2, [("search_products", {"query": "b"}), ("open_product", {"asin": "A1"})]),
            eval_row(
                3,
                [
                    ("search_products", {"query": "c"}),
                    ("open_product", {"asin": "A1"}),
                    ("open_product", {"asin": "A2"}),
                ],
            ),
        ]
        stats = summarize(rows)
        self.assertEqual(stats["trajectories"], 3)
        self.assertEqual(stats["paging"], 1)
        self.assertEqual(stats["multi_open"], 1)
        self.assertEqual(stats["exploratory"], 2)
        self.assertEqual(stats["template"], 1)

    def test_action_level_counts_the_decision_point(self):
        """行级配比和动作级配比是两个数：模型学的是后者。"""
        rows = [
            eval_row(1, [("search_products", {"query": "a"}), ("open_product", {"asin": "A1"})]),
            eval_row(
                2,
                [
                    ("search_products", {"query": "b"}),
                    ("next_page", {}),
                    ("open_product", {"asin": "A2"}),
                ],
            ),
        ]
        stats = summarize_actions(rows)
        self.assertEqual(stats["decision_points"], 2)
        self.assertEqual(stats["transitions"]["翻页"], 1)
        self.assertEqual(stats["transitions"]["打开候选（通常第 1 个）"], 1)
        self.assertEqual(stats["tool_counts"]["next_page"], 1)

    def test_expect_mismatch_is_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectories.jsonl"
            path.write_text(
                json.dumps(eval_row(1, [("search_products", {"query": "a"})])) + "\n",
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(path), "--expect", "200"]), 1)
                self.assertEqual(main([str(path), "--expect", "1"]), 0)


if __name__ == "__main__":
    unittest.main()
