"""验证 M5 预检脚本对「超长丢弃」与「真缺陷」的区分。

超长丢弃是 ``build_supervised_example`` 的 by-design 行为（不做可能截断工具调用的
截断），训练同样会丢这些行；预检必须把它与 loss 口径缺陷区分开，否则会在
Pure V4 池的最长几行上误报失败、挡住整轮训练。

关键语义：``loss_scope`` 只影响**标签**，不影响渲染长度 —— 所以同一行在**同一文件**
里两个口径都会因超长被丢；「1223 在 toolonly 文件不被丢」是**跨文件**差异
（toolonly 文件里 assistant content 已被抹空），不是同一文件内的口径差异。
"""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_sft_loss_scope import (
    UNBOUNDED_MAX_LENGTH,
    _check_file,
)


class CharTokenizer:
    """字符级 tokenizer：input_ids 长度 == 文本长度，便于精确控制超长。"""

    def apply_chat_template(
        self, messages, tools=None, tokenize=False, add_generation_prompt=False
    ):
        del tools, tokenize
        text = ""
        for message in messages:
            text += f"<{message['role']}>"
            text += message.get("content") or ""
            for call in message.get("tool_calls") or []:
                text += f"[tool={call['function']['name']}]"
            text += f"</{message['role']}>"
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(char) for char in text]}

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(token) for token in token_ids)


def _row(task_id, user_text, assistant_text, with_tool_call=True):
    message = {
        "role": "assistant",
        "content": assistant_text,
        "tool_calls": (
            [{"function": {"name": "buy_now", "arguments": "{}"}}] if with_tool_call else []
        ),
    }
    return {
        "task_id": task_id,
        "tools": [],
        "messages": [
            {"role": "user", "content": user_text},
            message,
        ],
    }


def _write_rows(rows: list[dict]) -> Path:
    tmpdir = tempfile.mkdtemp()
    source = Path(tmpdir) / "pool.jsonl"
    source.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return source


def _check(rows: list[dict], max_length: int) -> dict:
    return _check_file(
        _write_rows(rows), limit=10, tokenizer=CharTokenizer(),
        template=CharTokenizer(), max_length=max_length,
    )


class VerifySftLossScopeTest(unittest.TestCase):
    def test_full_file_over_long_row_drops_both_scopes_and_is_not_a_failure(self):
        """full 文件里的 1223 型：content 在数据里，两个口径的渲染都超长 → 超长丢弃。"""
        result = _check([_row(1223, "买一个枕头", "思考" * 60, True)], max_length=60)

        self.assertEqual(result["failure_count"], 0)
        self.assertEqual(result["over_long_drops"], 2)
        self.assertEqual(result["rows_checked"], 1)

    def test_toolonly_file_blank_content_makes_the_same_row_trainable(self):
        """toolonly 文件里 content 已抹空 → 同一 task_id 不再超长，两个口径都可训。"""
        result = _check([_row(1223, "买一个枕头", "", True)], max_length=60)

        self.assertEqual(result["failure_count"], 0)
        self.assertEqual(result["over_long_drops"], 0)
        self.assertGreater(result["supervised_tokens"]["tool_call_only"], 0)
        self.assertLess(
            result["supervised_tokens"]["tool_call_only"],
            result["supervised_tokens"]["assistant"],
        )

    def test_row_over_long_in_both_scopes_counts_two_drops(self):
        """user content 极长 → 两个口径都超长（user content 在 toolonly 里也保留）。"""
        result = _check([_row(983, "需求" * 60, "简短", True)], max_length=60)

        self.assertEqual(result["failure_count"], 0)
        self.assertEqual(result["over_long_drops"], 2)

    def test_genuinely_untrainable_row_is_still_a_failure(self):
        """没有 assistant 回合的行即使放宽 max_length 也训不成 → 必须判失败。"""
        result = _check(
            [{"task_id": 1, "tools": [], "messages": [{"role": "user", "content": "hi"}]}],
            max_length=60,
        )

        self.assertEqual(result["failure_count"], 2)
        self.assertEqual(result["over_long_drops"], 0)
        self.assertEqual(result["failures"][0]["reason"], "assistant: dropped (非超长)")

    def test_unbounded_max_length_is_used_for_the_drop_recheck(self):
        """复检上限必须是放宽值，否则超长行会被误判为真缺陷。"""
        self.assertGreater(UNBOUNDED_MAX_LENGTH, 1_000_000)

    def test_normal_rows_keep_the_semantics_checks(self):
        """可训练行的语义检查不受影响：tool-call-only 仍须排除 content。"""
        result = _check(
            [_row(1, "买一个枕头", "先搜索候选，确认价格与库存后再下单。", True)],
            max_length=10_000,
        )

        self.assertEqual(result["failure_count"], 0)
        self.assertEqual(result["over_long_drops"], 0)
        self.assertLess(
            result["supervised_tokens"]["tool_call_only"],
            result["supervised_tokens"]["assistant"],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
