"""老师模型 4xx 必须把**响应体**带出来，而不是只留 "HTTP Error 400"。

为什么钉死它（2026-10-21）
--------------------------
M10-R 探针冒烟 3/3 条轨迹都是 ``HTTPError 400: Bad Request``，`urllib` 的字符串里
没有任何可用信息，于是「温度越界」「模型名错」「额度用光」三种完全不同的故障
看起来一模一样。480 条的采集如果这么跑，只能看到「全挂」而不知道挂在哪。

三条不变量：

1. 4xx 立刻抛 ``ModelRequestError``，**不重试**（请求本身错了，重试只是浪费）。
2. 异常消息里必须含响应体原文。
3. ``_is_infrastructure_failure`` 认它 ⇒ 采集器立刻停，不会烧完整个批次。
"""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.evaluation.rollout import (  # noqa: E402
    ModelRequestError,
    OpenAIChatClient,
    _is_infrastructure_failure,
)

MESSAGES = [{"role": "user", "content": "hi"}]
TOOLS: list = []


def make_client(transport):
    return OpenAIChatClient(
        model="deepseek-v4-flash",
        base_url="https://example.invalid/v1",
        api_key="test-key",
        transport=transport,
        token_counter=lambda messages, tools: 1,
    )


def http_error(code, body: bytes):
    return HTTPError(
        "https://example.invalid/v1/chat/completions",
        code,
        "Bad Request",
        {},
        io.BytesIO(body),
    )


class ModelRequestErrorTest(unittest.TestCase):
    def test_400_raises_with_body(self):
        body = b'{"error":{"message":"temperature must be <= 1"}}'

        def transport(url, payload, headers, timeout):
            raise http_error(400, body)

        client = make_client(transport)
        with self.assertRaises(ModelRequestError) as ctx:
            client.complete(MESSAGES, TOOLS)
        self.assertEqual(ctx.exception.status, 400)
        self.assertIn("temperature must be <= 1", str(ctx.exception))
        self.assertIn("temperature must be <= 1", ctx.exception.body)

    def test_400_is_not_retried(self):
        calls = []

        def transport(url, payload, headers, timeout):
            calls.append(1)
            raise http_error(400, b'{"error":"bad"}')

        with self.assertRaises(ModelRequestError):
            make_client(transport).complete(MESSAGES, TOOLS)
        self.assertEqual(len(calls), 1, "4xx 重试没有意义，必须只请求一次")

    def test_429_also_raises_immediately(self):
        calls = []

        def transport(url, payload, headers, timeout):
            calls.append(1)
            raise http_error(429, b'{"error":"rate limit"}')

        with self.assertRaises(ModelRequestError) as ctx:
            make_client(transport).complete(MESSAGES, TOOLS)
        self.assertEqual(ctx.exception.status, 429)
        self.assertIn("rate limit", str(ctx.exception))
        self.assertEqual(len(calls), 1)

    def test_empty_body_does_not_crash(self):
        def transport(url, payload, headers, timeout):
            raise HTTPError("https://x/y", 403, "Forbidden", {}, None)

        with self.assertRaises(ModelRequestError) as ctx:
            make_client(transport).complete(MESSAGES, TOOLS)
        self.assertEqual(ctx.exception.status, 403)
        self.assertEqual(ctx.exception.body, "")

    def test_message_is_truncated_but_status_kept(self):
        body = b"x" * 5000

        def transport(url, payload, headers, timeout):
            raise http_error(400, body)

        with self.assertRaises(ModelRequestError) as ctx:
            make_client(transport).complete(MESSAGES, TOOLS)
        self.assertLessEqual(len(str(ctx.exception)), 600)

    def test_network_error_still_retried(self):
        """URLError 不是 4xx，仍按原逻辑重试，行为不能被这次改动带偏。"""
        calls = []

        def transport(url, payload, headers, timeout):
            calls.append(1)
            raise URLError("connection refused")

        with self.assertRaises(URLError):
            make_client(transport).complete(MESSAGES, TOOLS)
        self.assertGreater(len(calls), 1)


class InfrastructureFailureTest(unittest.TestCase):
    def test_model_request_error_stops_collection(self):
        trajectory = {"error": {"type": "ModelRequestError", "message": "HTTP 400"}}
        self.assertTrue(_is_infrastructure_failure(trajectory))

    def test_plain_runtime_error_does_not_stop_collection(self):
        trajectory = {"error": {"type": "ValueError", "message": "boom"}}
        self.assertFalse(_is_infrastructure_failure(trajectory))


if __name__ == "__main__":
    unittest.main()
