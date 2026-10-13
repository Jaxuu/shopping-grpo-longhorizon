"""验证 M5 的 loss-scope 开关（assistant vs tool-call-only）的标签语义。

这里的 tokenizer 刻意模拟 Qwen3.5 的模板形状：tool call 以 ``<tool_call>`` 块
追加在 assistant 自然语言 content 之后。这样可以在不加载真实模型的前提下，固定
"只监督 tool call、排除 assistant content" 的判定口径。
"""

import json
import tempfile
import unittest
from pathlib import Path

from shopping_grpo.training.sft.dataset import (
    IGNORE_INDEX,
    build_supervised_example,
    load_supervised_examples,
)


def _labeled_text(example):
    return QwenLikeTokenizer.decode(
        token for token in example["labels"] if token != IGNORE_INDEX
    )


class QwenLikeTokenizer:
    """按 Qwen3.5 的形状渲染：``<|im_start|>role\\n{content}{tool_calls}<|im_end|>\\n``。"""

    def apply_chat_template(
        self, messages, tools=None, tokenize=False, add_generation_prompt=False
    ):
        del tools, tokenize
        text = ""
        for message in messages:
            text += f"<|im_start|>{message['role']}\n"
            text += message.get("content") or ""
            for call in message.get("tool_calls") or []:
                function = call["function"]
                arguments = function["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                payload = {"name": function["name"], "arguments": arguments}
                text += "\n<tool_call>\n" + json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":")
                ) + "\n</tool_call>"
            text += "<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(char) for char in text]}

    @staticmethod
    def decode(token_ids):
        return "".join(chr(token) for token in token_ids)


class NonInsertionTokenizer(QwenLikeTokenizer):
    """去掉 tool_calls 会改变其它 token，模拟无法安全定位 tool call 的模板。"""

    def apply_chat_template(
        self, messages, tools=None, tokenize=False, add_generation_prompt=False
    ):
        text = ""
        for message in messages:
            text += f"<|im_start|>{message['role']}\n"
            text += message.get("content") or ""
            calls = message.get("tool_calls") or []
            for call in calls:
                text += f"[tool={call['function']['name']}]"
            text += "|calls|" if calls else "#"
            text += "<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text


def _messages():
    return [
        {"role": "system", "content": "system rule"},
        {"role": "user", "content": "买一个枕头"},
        {
            "role": "assistant",
            "content": "先搜索一下候选商品，再逐项比对规格。",
            "tool_calls": [
                {
                    "function": {
                        "name": "search_products",
                        "arguments": '{"query":"pillow"}',
                    }
                }
            ],
        },
        {"role": "tool", "content": "search result: item-1"},
        {
            "role": "assistant",
            "content": "确认价格与库存后下单。",
            "tool_calls": [{"function": {"name": "buy_now", "arguments": "{}"}}],
        },
    ]


class SftLossScopeTest(unittest.TestCase):
    def test_assistant_scope_supervises_content_and_tool_calls(self):
        example = build_supervised_example(
            messages=_messages(),
            tools=[{"type": "function"}],
            tokenizer=QwenLikeTokenizer(),
            max_length=10_000,
            loss_scope="assistant",
        )

        labeled = _labeled_text(example)
        self.assertIn("先搜索一下候选商品", labeled)
        self.assertIn("search_products", labeled)
        self.assertIn("buy_now", labeled)

    def test_tool_call_only_scope_excludes_assistant_content(self):
        example = build_supervised_example(
            messages=_messages(),
            tools=[{"type": "function"}],
            tokenizer=QwenLikeTokenizer(),
            max_length=10_000,
            loss_scope="tool-call-only",
        )

        labeled = _labeled_text(example)
        # 工具调用必须保留（含参数），assistant 的自然语言 content 必须被排除。
        self.assertIn("<tool_call>", labeled)
        self.assertIn("search_products", labeled)
        self.assertIn("buy_now", labeled)
        self.assertNotIn("先搜索一下候选商品", labeled)
        self.assertNotIn("确认价格与库存后下单", labeled)
        # 用户指令与环境回复本来就不参与 loss。
        self.assertNotIn("买一个枕头", labeled)
        self.assertNotIn("search result: item-1", labeled)

    def test_tool_call_only_is_strictly_narrower_than_assistant_scope(self):
        assistant = build_supervised_example(
            messages=_messages(),
            tools=[],
            tokenizer=QwenLikeTokenizer(),
            max_length=10_000,
            loss_scope="assistant",
        )
        tool_only = build_supervised_example(
            messages=_messages(),
            tools=[],
            tokenizer=QwenLikeTokenizer(),
            max_length=10_000,
            loss_scope="tool-call-only",
        )

        assistant_tokens = sum(1 for t in assistant["labels"] if t != IGNORE_INDEX)
        tool_only_tokens = sum(1 for t in tool_only["labels"] if t != IGNORE_INDEX)
        self.assertGreater(tool_only_tokens, 0)
        self.assertLess(tool_only_tokens, assistant_tokens)

    def test_tool_call_only_skips_assistant_turns_without_tool_calls(self):
        """只有 tool call 的回合才有监督信号；纯文本 assistant 回合在 B 臂被跳过。"""
        messages = [
            {"role": "user", "content": "buy"},
            {"role": "assistant", "content": "我先规划一下步骤，不做任何调用。"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "buy_now", "arguments": "{}"}}],
            },
        ]
        example = build_supervised_example(
            messages=messages,
            tools=[],
            tokenizer=QwenLikeTokenizer(),
            max_length=10_000,
            loss_scope="tool-call-only",
        )

        labeled = _labeled_text(example)
        self.assertIn("buy_now", labeled)
        self.assertNotIn("我先规划一下步骤", labeled)

    def test_tool_call_only_drops_rows_without_any_tool_call(self):
        example = build_supervised_example(
            messages=[
                {"role": "user", "content": "buy"},
                {"role": "assistant", "content": "这里没有任何工具调用。"},
            ],
            tools=[],
            tokenizer=QwenLikeTokenizer(),
            max_length=10_000,
            loss_scope="tool-call-only",
        )

        self.assertIsNone(example)

    def test_tool_call_only_excludes_template_scaffolding_on_blanked_data(self):
        """D_toolonly 上两个口径仍有差别：tool-call-only 不监督 ``<|im_end|>``。"""
        messages = [
            {"role": "user", "content": "buy"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "buy_now", "arguments": "{}"}}],
            },
        ]
        assistant = build_supervised_example(
            messages=messages,
            tools=[],
            tokenizer=QwenLikeTokenizer(),
            max_length=10_000,
            loss_scope="assistant",
        )
        tool_only = build_supervised_example(
            messages=messages,
            tools=[],
            tokenizer=QwenLikeTokenizer(),
            max_length=10_000,
            loss_scope="tool-call-only",
        )

        self.assertIn("<|im_end|>", _labeled_text(assistant))
        self.assertNotIn("<|im_end|>", _labeled_text(tool_only))
        self.assertIn("buy_now", _labeled_text(tool_only))

    def test_non_insertion_template_is_rejected_instead_of_mislabelled(self):
        """若模板不是"去掉 tool_calls = 纯删除"，必须丢弃样本而非产生错位标签。"""
        example = build_supervised_example(
            messages=[
                {"role": "user", "content": "buy"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "buy_now", "arguments": "{}"}}],
                },
            ],
            tools=[],
            tokenizer=NonInsertionTokenizer(),
            max_length=10_000,
            loss_scope="tool-call-only",
        )

        self.assertIsNone(example)

    def test_invalid_loss_scope_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "loss_scope"):
            build_supervised_example(
                messages=_messages(),
                tools=[],
                tokenizer=QwenLikeTokenizer(),
                max_length=10_000,
                loss_scope="content-only",
            )

    def test_loader_threads_loss_scope_to_every_row(self):
        row = {
            "task_id": 7,
            "trajectory_id": "7-0",
            "tools": [],
            "messages": _messages(),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "sft.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            examples, stats = load_supervised_examples(
                source,
                tokenizer=QwenLikeTokenizer(),
                max_length=10_000,
                loss_scope="tool-call-only",
            )

        self.assertEqual(stats, {"total": 1, "kept": 1, "dropped": 0})
        labeled = _labeled_text(examples[0])
        self.assertIn("buy_now", labeled)
        self.assertNotIn("先搜索一下候选商品", labeled)

    def test_default_scope_is_assistant_for_backward_compatibility(self):
        implicit = build_supervised_example(
            messages=_messages(),
            tools=[],
            tokenizer=QwenLikeTokenizer(),
            max_length=10_000,
        )
        explicit = build_supervised_example(
            messages=_messages(),
            tools=[],
            tokenizer=QwenLikeTokenizer(),
            max_length=10_000,
            loss_scope="assistant",
        )

        self.assertEqual(implicit["labels"], explicit["labels"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
