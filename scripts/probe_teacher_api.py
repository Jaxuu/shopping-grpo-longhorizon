#!/usr/bin/env python3
"""老师 API 参数体检：到底哪个字段让服务端返回 400。

为什么需要它（2026-10-21）
--------------------------
M10-R 探针冒烟 3/3 条轨迹 `HTTPError 400: Bad Request`，但 `urllib` 抛出的
`HTTPError` 里**响应体被丢掉了**（`evaluation/rollout.py` 只把异常往上抛），
所以「400」这三个字什么都不能说明：可能是温度越界、可能是 `thinking` 字段、
可能是模型名、也可能是额度。

本脚本绕过整套采集流程，直接对同一个 endpoint 发**最小请求**，逐个变体试参数，
把 HTTP 状态码和**响应体原文**打出来。一次几十秒，不碰环境、不消耗采集额度。

用法::

    source ~/.m2_collect.env      # 先配好 OPENAI_BASE_URL / OPENAI_API_KEY
    .venv/bin/python scripts/probe_teacher_api.py

    # 只测指定温度
    .venv/bin/python scripts/probe_teacher_api.py --temperature 0.9

    # 换模型
    .venv/bin/python scripts/probe_teacher_api.py --model deepseek-v4-flash

判读
----
* 只有带 ``temperature>0`` 的变体 400 ⇒ 服务端不认这个温度，改用被接受的档位。
* 所有变体都 400 且响应体是 ``MissingSessionID`` ⇒ 缺 ``x-opencode-session`` 请求头。
  这是 2026-10-21 实测到的真因（和温度毫无关系）：OpenCode Go 要求每个对话带稳定
  会话 ID，缺失直接 400。客户端已在 ``OpenAIChatClient`` 里默认发送。
* 所有变体都 400 ⇒ 不是参数问题，看响应体（多半是模型名 / 额度 / 鉴权）。
* 全部 200 ⇒ API 侧没问题，异常在采集链路里（环境或工具调用转换）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid

#: 要试的温度档位。0.0 是历史采集用的档位（Step 2 在它下面跑通过）。
DEFAULT_TEMPERATURES = (0.0, 0.7, 0.9, 1.0)


def build_payload(model: str, temperature: float | None, *, thinking: bool) -> dict:
    """最小可用的 tool-call 请求体，字段与 ``OpenAIChatClient`` 一致。

    ``temperature=None`` 表示**不带这个字段**，用来区分「温度值被拒」和
    「整个字段不被支持」。
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "请调用 noop 工具，参数留空。"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "noop",
                    "description": "什么都不做的占位工具。",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "max_tokens": 64,
    }
    if temperature is not None:
        payload["temperature"] = temperature
        payload["top_p"] = 1.0
    if thinking:
        payload["thinking"] = {"type": "disabled"}
    return payload


def post(url: str, payload: dict, api_key: str, timeout: int, *, session_id: str = "") -> tuple[int, str]:
    """返回 (状态码, 响应体)。4xx/5xx 也读响应体——那才是原因所在。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "shopping-grpo-longhorizon/0.1",
    }
    # OpenCode Go 硬性要求：缺这个头直接 400 MissingSessionID。
    if session_id:
        headers["x-opencode-session"] = session_id
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as raw:
            return raw.status, raw.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace") if exc.fp else ""
        return exc.code, body
    except Exception as exc:  # 网络层问题单独归类，不要和 4xx 混
        return -1, f"{type(exc).__name__}: {exc}"


def chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def excerpt(text: str, *, limit: int = 300) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--temperature",
        type=float,
        action="append",
        dest="temperatures",
        help="可重复；默认测 0.0 / 0.7 / 0.9 / 1.0",
    )
    parser.add_argument(
        "--no-thinking-field",
        action="store_true",
        help="额外测一次「不带 thinking 字段」",
    )
    parser.add_argument(
        "--session-id",
        default=os.environ.get("OPENAI_SESSION_ID", ""),
        help="x-opencode-session 的值；默认取 OPENAI_SESSION_ID，再退随机 uuid",
    )
    parser.add_argument(
        "--without-session",
        action="store_true",
        help="额外测一次「不带 x-opencode-session」，用来确认它是不是必要条件",
    )
    args = parser.parse_args(argv)

    if not args.base_url:
        print("没有 OPENAI_BASE_URL，先 source 环境文件。", file=sys.stderr)
        return 2
    if not args.api_key:
        print("没有 OPENAI_API_KEY，先 source 环境文件。", file=sys.stderr)
        return 2

    url = chat_url(args.base_url)
    temperatures = tuple(args.temperatures or DEFAULT_TEMPERATURES)
    # deepseek-v4 系列默认开 thinking，必须显式关掉，否则不返回 tool_calls。
    thinking = args.model.casefold().startswith("deepseek-v4")
    session_id = args.session_id or uuid.uuid4().hex

    variants: list[tuple[str, float | None, bool, str]] = [
        (f"temperature={value}", value, thinking, session_id) for value in temperatures
    ]
    variants.append(("不带 temperature 字段", None, thinking, session_id))
    if args.no_thinking_field:
        variants.append(
            (f"temperature={temperatures[-1]} 且不带 thinking", temperatures[-1], False, session_id)
        )
    if args.without_session:
        variants.append(
            ("带温度但不带 x-opencode-session", temperatures[-1], thinking, "")
        )

    print(f"endpoint   : {url}")
    print(f"model      : {args.model}")
    print(f"session id : {session_id}（请求头 x-opencode-session）")
    print()
    print("| 变体 | 状态码 | 响应体（前 300 字） |")
    print("|---|---|---|")

    failures = 0
    for label, temperature, with_thinking, variant_session in variants:
        payload = build_payload(args.model, temperature, thinking=with_thinking)
        status, body = post(
            url, payload, args.api_key, args.timeout, session_id=variant_session
        )
        if status != 200:
            failures += 1
        print(f"| {label} | {status} | {excerpt(body).replace('|', chr(92) + '|')} |")

    print()
    if failures == 0:
        print("全部 200：API 侧没问题，异常在采集链路里（环境或工具调用转换）。")
        return 0
    if failures == len(variants):
        print("全部失败：不是温度问题。按响应体查模型名 / 额度 / 鉴权 / 工作区区域设置。")
        return 1
    print("部分失败：看哪一行红，就是那个参数被拒。把这张表贴回来。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
