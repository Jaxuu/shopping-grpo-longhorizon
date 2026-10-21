"""`scripts/probe_teacher_api.py` 的请求体形状固定。

钉死三件事：

1. `base_url` 已经带 `/chat/completions` 时不再拼一次。
2. `temperature=None` 表示**不带这个字段**——这是区分「温度值被拒」和
   「整个字段不被支持」的唯一手段，不能悄悄补成 0.0。
3. `thinking` 字段只在 deepseek-v4 系列上带（其他模型不认）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import probe_teacher_api as module  # noqa: E402


class ChatUrlTest(unittest.TestCase):
    def test_appends_suffix(self):
        self.assertEqual(
            module.chat_url("https://opencode.ai/zen/go/v1"),
            "https://opencode.ai/zen/go/v1/chat/completions",
        )

    def test_does_not_double_append(self):
        self.assertEqual(
            module.chat_url("https://x/v1/chat/completions"),
            "https://x/v1/chat/completions",
        )

    def test_strips_trailing_slash(self):
        self.assertEqual(
            module.chat_url("https://x/v1/"), "https://x/v1/chat/completions"
        )


class BuildPayloadTest(unittest.TestCase):
    def test_temperature_none_omits_field(self):
        payload = module.build_payload("m", None, thinking=False)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("top_p", payload)

    def test_temperature_included(self):
        payload = module.build_payload("m", 0.9, thinking=False)
        self.assertEqual(payload["temperature"], 0.9)
        self.assertEqual(payload["top_p"], 1.0)

    def test_thinking_field_toggle(self):
        self.assertIn("thinking", module.build_payload("m", 0.9, thinking=True))
        self.assertNotIn("thinking", module.build_payload("m", 0.9, thinking=False))

    def test_always_has_one_tool(self):
        payload = module.build_payload("m", 0.0, thinking=False)
        self.assertEqual(len(payload["tools"]), 1)
        self.assertEqual(payload["tools"][0]["function"]["name"], "noop")


class ExcerptTest(unittest.TestCase):
    def test_truncates(self):
        self.assertEqual(module.excerpt("a" * 400, limit=10), "a" * 9 + "…")

    def test_flattens_newlines(self):
        self.assertEqual(module.excerpt("a\n\nb"), "a b")


class MainGuardTest(unittest.TestCase):
    def test_missing_base_url_returns_2(self):
        self.assertEqual(module.main(["--base-url", "", "--api-key", "k"]), 2)

    def test_missing_api_key_returns_2(self):
        self.assertEqual(module.main(["--base-url", "https://x/v1", "--api-key", ""]), 2)


if __name__ == "__main__":
    unittest.main()
