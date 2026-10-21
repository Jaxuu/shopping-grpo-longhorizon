"""`x-opencode-session` 请求头：缺它服务端直接 400 MissingSessionID。

为什么钉死它（2026-10-21）
--------------------------
探针冒烟 3/3 条 `HTTPError 400`，响应体是 ``MissingSessionID``——
跟温度、模型名、额度都无关，是 OpenCode Go 新增的硬性要求：
每个对话必须带一个稳定的 ``x-opencode-session``。

四条不变量：

1. 请求头里**必须有** ``x-opencode-session``，且非空。
2. **同一个 client 实例内稳定**（一条轨迹 = 一个对话，中途换 ID 会丢路由/缓存）。
3. **不同实例不同 ID**（按对话区分是文档的明确要求）。
4. 显式传入 > 环境变量 ``OPENAI_SESSION_ID`` > 随机。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.evaluation.rollout import (  # noqa: E402
    SESSION_ID_ENV,
    OpenAIChatClient,
    client_from_env,
)

MESSAGES = [{"role": "user", "content": "hi"}]
TOOLS: list = []


def response_payload():
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "ok",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "noop", "arguments": "{}"},
                        }
                    ],
                }
            }
        ]
    }


class CapturingTransport:
    def __init__(self):
        self.headers = []

    def __call__(self, url, payload, headers, timeout):
        self.headers.append(dict(headers))
        return response_payload()


def make_client(transport, **kwargs):
    return OpenAIChatClient(
        model="deepseek-v4-flash",
        base_url="https://example.invalid/v1",
        api_key="test-key",
        transport=transport,
        token_counter=lambda messages, tools: 1,
        **kwargs,
    )


class SessionHeaderTest(unittest.TestCase):
    def test_header_is_sent(self):
        transport = CapturingTransport()
        make_client(transport).complete(MESSAGES, TOOLS)
        self.assertIn("x-opencode-session", transport.headers[0])
        self.assertTrue(transport.headers[0]["x-opencode-session"])

    def test_header_is_stable_within_one_client(self):
        transport = CapturingTransport()
        client = make_client(transport)
        client.complete(MESSAGES, TOOLS)
        client.complete(MESSAGES, TOOLS)
        self.assertEqual(
            transport.headers[0]["x-opencode-session"],
            transport.headers[1]["x-opencode-session"],
        )

    def test_different_clients_get_different_ids(self):
        first, second = CapturingTransport(), CapturingTransport()
        make_client(first).complete(MESSAGES, TOOLS)
        make_client(second).complete(MESSAGES, TOOLS)
        self.assertNotEqual(
            first.headers[0]["x-opencode-session"],
            second.headers[0]["x-opencode-session"],
            "按对话区分是文档要求，不能所有实例共用一个 ID",
        )

    def test_explicit_session_id_wins(self):
        transport = CapturingTransport()
        make_client(transport, session_id="fixed-abc").complete(MESSAGES, TOOLS)
        self.assertEqual(transport.headers[0]["x-opencode-session"], "fixed-abc")

    def test_env_session_id_used_when_not_explicit(self):
        transport = CapturingTransport()
        with mock.patch.dict(os.environ, {SESSION_ID_ENV: "from-env"}):
            make_client(transport).complete(MESSAGES, TOOLS)
        self.assertEqual(transport.headers[0]["x-opencode-session"], "from-env")

    def test_explicit_beats_env(self):
        transport = CapturingTransport()
        with mock.patch.dict(os.environ, {SESSION_ID_ENV: "from-env"}):
            make_client(transport, session_id="explicit").complete(MESSAGES, TOOLS)
        self.assertEqual(transport.headers[0]["x-opencode-session"], "explicit")

    def test_other_headers_unchanged(self):
        transport = CapturingTransport()
        make_client(transport).complete(MESSAGES, TOOLS)
        headers = transport.headers[0]
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertIn("shopping-grpo-longhorizon", headers["User-Agent"])


class ClientFromEnvTest(unittest.TestCase):
    def test_passes_session_id_through(self):
        client = client_from_env(
            api_key="k", base_url="https://x/v1", session_id="s-1"
        )
        self.assertEqual(client.session_id, "s-1")

    def test_generates_when_absent(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(SESSION_ID_ENV, None)
            client = client_from_env(api_key="k", base_url="https://x/v1")
        self.assertTrue(client.session_id)
        self.assertEqual(len(client.session_id), 32)


if __name__ == "__main__":
    unittest.main()
