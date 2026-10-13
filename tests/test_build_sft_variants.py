import json
import unittest

from scripts.build_sft_variants import build_variants, strip_assistant_text

TOOLS = [{"type": "function", "function": {"name": "buy_now"}}]


def _row(task_id, *, assistant_text):
    return {
        "trajectory_id": f"traj-{task_id}",
        "task_id": task_id,
        "messages": [
            {"role": "system", "content": "system text"},
            {"role": "user", "content": "user text"},
            {
                "role": "assistant",
                "content": assistant_text,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "buy_now", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "购买已完成。"},
        ],
        "tools": json.loads(json.dumps(TOOLS)),
    }


class BuildSftVariantsTests(unittest.TestCase):
    def test_strip_blanks_only_assistant_text(self):
        messages = _row(1, assistant_text="let me buy").__getitem__("messages")
        stripped, turns, chars = strip_assistant_text(messages)

        self.assertEqual(turns, 1)
        self.assertEqual(chars, len("let me buy"))
        self.assertEqual(stripped[2]["content"], "")
        self.assertEqual(stripped[0]["content"], "system text")
        self.assertEqual(stripped[1]["content"], "user text")
        self.assertEqual(stripped[3]["content"], "购买已完成。")
        self.assertEqual(stripped[2]["tool_calls"][0]["function"]["name"], "buy_now")
        self.assertEqual(messages[2]["content"], "let me buy")

    def test_variants_keep_identical_task_ids_and_rows(self):
        rows = [_row(1, assistant_text="a"), _row(2, assistant_text="")]
        toolonly, stats = build_variants(rows)

        self.assertEqual(stats["rows"], stats["toolonly_rows"], 2)
        self.assertEqual(stats["rows_with_assistant_text"], 1)
        self.assertEqual(stats["assistant_turns_stripped"], 1)
        self.assertEqual(
            [row["task_id"] for row in toolonly], [row["task_id"] for row in rows]
        )
        self.assertTrue(all(m["content"] == "" for r in toolonly for m in r["messages"] if m["role"] == "assistant"))

    def test_rejects_row_without_tool_call(self):
        row = _row(1, assistant_text="text")
        row["messages"][2].pop("tool_calls")
        with self.assertRaisesRegex(ValueError, "no tool call"):
            build_variants([row])

    def test_rejects_row_with_unexpected_keys(self):
        row = _row(1, assistant_text="text")
        row["extra"] = 1
        with self.assertRaisesRegex(ValueError, "unexpected row keys"):
            build_variants([row])


if __name__ == "__main__":
    unittest.main()
