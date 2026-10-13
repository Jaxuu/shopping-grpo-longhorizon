import json
import unittest

from scripts.merge_m2_increment import (
    _guard_reason,
    guard_pairs,
    merge_rows,
    normalize_row,
    repair_messages,
    uniform_tools,
)
from scripts.prepare_sft_curriculum import _validate_row

TOOLS = [
    {"type": "function", "function": {"name": name}}
    for name in ("search_products", "open_product", "select_option", "buy_now")
]
GUARD_OBSERVATION = (
    "上一工具调用被本地动作守卫拒绝（search_not_available_on_current_page），未执行。"
    "下一步只能从当前页面列出的目标中选择。"
)


def _call(name, arguments, call_id):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def _observation(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def trajectory(task_id, *, guard=True):
    messages = [
        {"role": "system", "content": "use tools"},
        {"role": "user", "content": "buy one"},
        _call("search_products", {"query": "item"}, "c1"),
        _observation("c1", "page_type: search_results"),
    ]
    if guard:
        messages.extend(
            [
                _call("search_products", {"query": "item again"}, "c2"),
                _observation("c2", GUARD_OBSERVATION),
            ]
        )
    messages.extend(
        [
            _call("open_product", {"asin": "a1"}, "c3"),
            _observation("c3", "page_type: product_detail"),
            _call("buy_now", {}, "c4"),
            _observation("c4", "购买已完成。"),
        ]
    )
    return {
        "trajectory_id": f"traj-{task_id}",
        "task_id": task_id,
        "messages": messages,
        "tools": json.loads(json.dumps(TOOLS)),
    }


class MergeM2IncrementTests(unittest.TestCase):
    def test_extracts_guard_reason_code(self):
        self.assertEqual(
            _guard_reason(GUARD_OBSERVATION), "search_not_available_on_current_page"
        )
        self.assertEqual(_guard_reason("no reason here"), "unknown")

    def test_repair_removes_illegal_turn_and_rejection(self):
        messages = trajectory(1)["messages"]
        pairs = guard_pairs(messages)
        self.assertEqual(len(pairs), 1)
        assistant_index, tool_index, reason, not_executed = pairs[0]
        self.assertIsNotNone(assistant_index)
        self.assertTrue(not_executed)

        repaired, removed = repair_messages(messages)
        self.assertEqual(len(removed), 1)
        self.assertEqual(len(repaired), len(messages) - 2)
        self.assertNotIn(GUARD_OBSERVATION, [m.get("content") for m in repaired])
        self.assertFalse(
            any(m.get("tool_calls") and m["tool_calls"][0]["id"] == "c2" for m in repaired)
        )

    def test_repaired_row_passes_curriculum_validation(self):
        with self.assertRaisesRegex(ValueError, "guard rejection"):
            _validate_row(trajectory(1))
        row, pairs = normalize_row(trajectory(1), TOOLS)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(set(row), {"trajectory_id", "task_id", "messages", "tools"})
        _validate_row(row)

    def test_clean_row_is_untouched(self):
        row, pairs = normalize_row(trajectory(1, guard=False), TOOLS)
        self.assertEqual(pairs, [])
        _validate_row(row)

    def test_merge_is_idempotent_and_deduplicates(self):
        pool = [trajectory(task_id, guard=False) for task_id in (1, 2, 3)]
        increment = [trajectory(4, guard=False), trajectory(2, guard=False)]

        merged, stats = merge_rows(pool, increment)
        self.assertEqual(stats["merged_rows"], 4)
        self.assertEqual(stats["replaced_existing"], 1)
        self.assertEqual([row["task_id"] for row in merged], [1, 2, 3, 4])

        merged_again, stats_again = merge_rows(merged, increment)
        self.assertEqual([row["task_id"] for row in merged_again], [1, 2, 3, 4])
        self.assertEqual(stats_again["merged_rows"], 4)

    def test_merge_rejects_duplicate_increment_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate task_ids"):
            merge_rows([], [trajectory(1, guard=False), trajectory(1, guard=False)])

    def test_uniform_tools_requires_a_single_schema(self):
        self.assertEqual(uniform_tools([trajectory(1, guard=False)]), TOOLS)
        with self.assertRaisesRegex(ValueError, "distinct tools schemas"):
            uniform_tools([trajectory(1, guard=False), {**trajectory(2, guard=False), "tools": []}])


if __name__ == "__main__":
    unittest.main()
