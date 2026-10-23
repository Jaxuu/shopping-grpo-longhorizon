import unittest

from scripts.augment_trajectory_reasoning import (
    action_text,
    build_prompt,
    clean_reply,
    fill_reasons,
    has_paging,
    render_history,
)


def _call(name, arguments="{}"):
    return {"function": {"name": name, "arguments": arguments}}


def _row(assistant_contents=("", "", ""), with_paging=True):
    """造一条最小轨迹：搜索 → (翻页) → 下单。"""

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "求一款净含量 60ml 的男士防晒霜，预算 350 到 450 元"},
        {
            "role": "assistant",
            "content": assistant_contents[0],
            "tool_calls": [_call("search_products", '{"query": "防晒霜 男"}')],
        },
        {"role": "tool", "content": "1|A1|352|绿茶名妆|类目|30ml,清爽|碧欧泉男士防晒霜\n2|A2|189|普小里|类目|清爽|某防晒"},
    ]
    if with_paging:
        messages.append(
            {"role": "assistant", "content": assistant_contents[1], "tool_calls": [_call("next_page")]}
        )
        messages.append({"role": "tool", "content": "21|B1|400|景涟|类目|60ml|欧莱雅男士小金管"})
    messages.append(
        {"role": "assistant", "content": assistant_contents[2], "tool_calls": [_call("buy_now")]}
    )
    return {"trajectory_id": "t-1", "task_id": 7, "messages": messages, "tools": []}


class ActionTextTest(unittest.TestCase):
    def test_renders_name_with_arguments(self):
        self.assertEqual(action_text([_call("next_page")]), "next_page()")
        self.assertIn("search_products", action_text([_call("search_products", '{"query": "x"}')]))

    def test_multiple_calls_are_joined(self):
        text = action_text([_call("next_page"), _call("prev_page")])
        self.assertEqual(text, "next_page() + prev_page()")


class HasPagingTest(unittest.TestCase):
    def test_detects_next_page(self):
        self.assertTrue(has_paging(_row()))
        self.assertFalse(has_paging(_row(with_paging=False)))


class CleanReplyTest(unittest.TestCase):
    def test_strips_label_and_quotes(self):
        self.assertEqual(clean_reply("理由：这一页没有 60ml 的。"), "这一页没有 60ml 的。")
        self.assertEqual(clean_reply("“这一页都超预算了。”"), "这一页都超预算了。")

    def test_collapses_whitespace(self):
        self.assertEqual(clean_reply("  这一页\n 没有合适的  "), "这一页 没有合适的")


class FillReasonsTest(unittest.TestCase):
    def test_fills_only_empty_assistant_turns(self):
        row = _row(assistant_contents=("", "", ""))
        prompts = []

        def ask(prompt):
            prompts.append(prompt)
            return "理由：这一页都是 30ml，不符合 60ml 要求。"

        result, filled = fill_reasons(row, ask)
        contents = [m["content"] for m in result["messages"] if m.get("role") == "assistant"]
        self.assertEqual(filled, 3)
        self.assertTrue(all(c == "这一页都是 30ml，不符合 60ml 要求。" for c in contents))
        self.assertEqual(len(prompts), 3)

    def test_keeps_existing_reasoning_untouched(self):
        row = _row(assistant_contents=("老师原话：先搜一下。", "", ""))
        asked = []

        def ask(prompt):
            asked.append(prompt)
            return "补的理由"

        result, filled = fill_reasons(row, ask)
        assistant = [m for m in result["messages"] if m.get("role") == "assistant"]
        self.assertEqual(assistant[0]["content"], "老师原话：先搜一下。")
        self.assertEqual(filled, 2)

    def test_does_not_touch_tool_calls(self):
        row = _row()
        before = [m.get("tool_calls") for m in row["messages"] if m.get("role") == "assistant"]
        result, _ = fill_reasons(row, lambda prompt: "理由")
        after = [m.get("tool_calls") for m in result["messages"] if m.get("role") == "assistant"]
        self.assertEqual(before, after)

    def test_empty_reply_leaves_turn_untouched(self):
        row = _row()
        _, filled = fill_reasons(row, lambda prompt: "   ")
        self.assertEqual(filled, 0)


class PromptTest(unittest.TestCase):
    def test_prompt_carries_instruction_and_action(self):
        text = build_prompt("买个 60ml 的防晒", [("search_products()", "1|A1|352")], "next_page()", 3)
        self.assertIn("买个 60ml 的防晒", text)
        self.assertIn("next_page()", text)
        self.assertIn("翻页", text)  # 要求说明这一页为什么不够

    def test_history_keeps_only_recent_steps(self):
        recent = [(f"open_product({i})", f"obs{i}") for i in range(6)]
        text = render_history(recent, 2)
        self.assertIn("obs5", text)
        self.assertNotIn("obs2", text)


if __name__ == "__main__":
    unittest.main()
