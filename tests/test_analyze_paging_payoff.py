import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.analyze_paging_payoff import main, summarize


def row(task_id, calls, reward_type):
    return {
        "task_id": task_id,
        "steps": [
            {"step_index": i, "tool_name": name, "parameters": params}
            for i, (name, params) in enumerate(calls)
        ],
        "terminal_result": {"reward_detail": {"reward_type": reward_type}},
    }


class PagingPayoffTest(unittest.TestCase):
    def test_splits_by_whether_next_page_appeared(self):
        rows = [
            row(1, [("search_products", {"query": "a"}), ("next_page", {})], "gold_purchase"),
            row(2, [("search_products", {"query": "b"}), ("select_option", {})], "wrong_purchase"),
            row(3, [("search_products", {"query": "c"}), ("next_page", {})], "wrong_purchase"),
        ]
        stats = summarize(rows)
        self.assertEqual(stats["翻页"]["tasks"], 2)
        self.assertEqual(stats["翻页"]["gold"], 1)
        self.assertEqual(stats["没翻页"]["tasks"], 1)
        self.assertEqual(stats["没翻页"]["wrong"], 1)

    def test_prev_page_does_not_count_as_paging(self):
        rows = [row(1, [("search_products", {"query": "a"}), ("prev_page", {})], "gold_purchase")]
        stats = summarize(rows)
        self.assertNotIn("翻页", stats)
        self.assertEqual(stats["没翻页"]["tasks"], 1)

    def test_chain_coverage_is_counted_per_task(self):
        rows = [
            row(
                1,
                [
                    ("search_products", {"query": "a"}),
                    ("select_option", {}),
                    ("select_option", {}),
                ],
                "gold_purchase",
            ),
            row(2, [("search_products", {"query": "b"}), ("buy_now", {})], "gold_purchase"),
        ]
        stats = summarize(rows)
        self.assertEqual(stats["没翻页"]["chain"]["select_option"], 1)
        self.assertEqual(stats["没翻页"]["chain"]["buy_now"], 1)

    def test_missing_file_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(Path(tmp) / "nope.jsonl")]), 1)

    def test_reads_real_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traj.jsonl"
            path.write_text(
                json.dumps(row(1, [("search_products", {"query": "a"})], "gold_purchase")) + "\n",
                encoding="utf-8",
            )
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                self.assertEqual(main([str(path)]), 0)
            self.assertIn("没翻页", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
