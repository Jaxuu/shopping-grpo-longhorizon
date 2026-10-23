#!/usr/bin/env python3
"""给合成轨迹补上「为什么这么做」的思考文本。

M10-R v2 的池子里，非翻页轨迹 89.1% 带思考，**翻页轨迹只有 13.2%**——
那 144 条是脚本合成的，合成时只拼了动作序列，把老师的思考丢了。
结果是模型学会了「偶尔翻一下」，但学不到「什么情况该翻」。

本脚本只做一件事：对指定轨迹里 **content 为空的 assistant 回合**，
让老师模型补一句中文理由。动作序列、tool_calls、tools 一个字节都不动。

约束：
- 只补空的，已有思考原样保留（那些是老师真实输出的，比补的可靠）；
- 理由必须引用页面里的具体信息（价格/规格/属性/排名），否则模型学到的是空话；
- 按 trajectory_id 断点续跑，重复执行不会重复花钱。

用法：
    export OPENAI_BASE_URL=https://api.deepseek.com/v1 OPENAI_API_KEY=sk-xxx
    python scripts/augment_trajectory_reasoning.py \
        --source data/sft_m10r/capped/all.jsonl \
        --output data/sft_m10r/capped/all_reasoned.jsonl \
        --paging-only --workers 8

凭据只认老师模型的变量（``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``）。
**不要**用 ``.env`` 里的 ``LLM_BASE_URL``：那是评测用的本地 vLLM。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ACTION_LABELS = {
    "search_products": "搜索",
    "next_page": "翻到下一页",
    "prev_page": "返回",
    "open_product": "打开商品详情",
    "select_option": "选择规格",
    "view_features": "查看参数",
    "view_description": "查看描述",
    "view_reviews": "查看评价",
    "back_to_search": "回到搜索结果",
    "buy_now": "下单购买",
    "finish_without_purchase": "放弃购买",
}

PROMPT = """你在复现一条购物任务的轨迹，需要为其中一步补一句简短的中文说明。

【购物需求】
{instruction}

【最近看到的页面与操作】
{history}

【你这一步的决定】
{action}

请用**一句中文**（20–60 字）说明你为什么这么做。要求：
- 必须引用页面里的具体信息（价格、规格、属性、排名、店铺），不要泛泛而谈；
- 如果这一步是翻页，要说明这一页为什么不够（缺什么条件）；
- 只输出这一句话，不要解释、不要加引号、不要输出其他内容。"""


def action_text(tool_calls: list[dict]) -> str:
    """把一次工具调用渲染成「名称(参数)」，供 prompt 与日志使用。"""

    parts = []
    for call in tool_calls or []:
        fn = call.get("function") or {}
        name = fn.get("name") or "?"
        raw = fn.get("arguments")
        args = ""
        if isinstance(raw, str) and raw.strip():
            args = raw.strip()
        elif isinstance(raw, dict) and raw:
            args = json.dumps(raw, ensure_ascii=False)
        # next_page / buy_now 这类无参动作常被序列化成 "{}"，渲染成 () 更易读。
        if args in ("{}", "null", "None"):
            args = ""
        parts.append(f"{name}({args})" if args else f"{name}()")
    return " + ".join(parts)


def has_paging(row: dict) -> bool:
    for message in row.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if (call.get("function") or {}).get("name") == "next_page":
                return True
    return False


def instruction_of(row: dict) -> str:
    for message in row.get("messages") or []:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def render_history(recent: list[tuple[str, str]], limit: int) -> str:
    if not recent:
        return "（这是第一步）"
    lines = []
    for index, (action, observation) in enumerate(recent[-limit:], start=1):
        label = ACTION_LABELS.get(action.split("(")[0], action)
        snippet = " ".join((observation or "").split())[:700]
        lines.append(f"{index}. 执行「{label}」→ 页面内容：{snippet}")
    return "\n".join(lines)


def build_prompt(instruction: str, recent: list[tuple[str, str]], action: str, limit: int) -> str:
    return PROMPT.format(
        instruction=instruction.strip()[:1200],
        history=render_history(recent, limit),
        action=action,
    )


def clean_reply(text: str) -> str:
    """去掉模型爱加的引号/前缀，只留一句。"""

    line = " ".join(str(text or "").split())
    for prefix in ("理由：", "理由:", "说明：", "说明:"):
        if line.startswith(prefix):
            line = line[len(prefix) :].strip()
    return line.strip().strip('"“”「」').strip()


def fill_reasons(row: dict, ask, context_steps: int = 3) -> tuple[dict, int]:
    """给 content 为空的 assistant 回合补理由；返回 (row, 补了几条)。"""

    instruction = instruction_of(row)
    recent: list[tuple[str, str]] = []
    last_observation = ""
    filled = 0
    for message in row.get("messages") or []:
        role = message.get("role")
        if role == "tool":
            last_observation = str(message.get("content") or "")
            continue
        if role != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            continue
        action = action_text(tool_calls)
        existing = str(message.get("content") or "").strip()
        if not existing:
            prompt = build_prompt(instruction, recent, action, context_steps)
            reply = clean_reply(ask(prompt))
            if reply:
                message["content"] = reply
                filled += 1
        recent.append((action, last_observation))
    return row, filled


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                done.add(str(json.loads(line).get("trajectory_id")))
            except json.JSONDecodeError:
                continue
    return done


def _first_env(environ: dict, names: tuple[str, ...]) -> tuple[str | None, str | None]:
    for name in names:
        value = str(environ.get(name) or "").strip()
        if value:
            return value, name
    return None, None


def resolve_credentials(environ: dict | None = None) -> tuple[str | None, str | None, str]:
    """取老师模型的地址与密钥，返回 ``(base_url, api_key, 来源说明)``。

    优先 ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``——这是仓库里采集脚本
    （``collect_sft_data.py``、``label_sft_difficulty.py``）的老师模型口径。

    ``LLM_BASE_URL`` / ``LLM_API_KEY`` 只作兜底：``.env`` 里它们是**评测用**的本地
    vLLM（``http://127.0.0.1:8000/v1``）。误用它补思考 = 让 2B 学生模型给自己的
    轨迹写理由，数据会静默变废，所以调用方还要对本地地址再警告一次。
    """
    env = os.environ if environ is None else environ
    base_url, base_src = _first_env(env, ("OPENAI_BASE_URL", "LLM_BASE_URL"))
    api_key, key_src = _first_env(env, ("OPENAI_API_KEY", "LLM_API_KEY"))
    return base_url, api_key, f"{base_src or '--base-url'}/{key_src or '--api-key'}"


def looks_local(base_url: str | None) -> bool:
    """地址是否指向本机（多半是评测用的 vLLM，而不是老师模型）。"""
    if not base_url:
        return False
    host = base_url.split("//", 1)[-1].split("/", 1)[0].split(":", 1)[0].lower()
    return host in ("127.0.0.1", "localhost", "0.0.0.0", "::1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paging-only", action="store_true", help="只处理含 next_page 的轨迹")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--context-steps", type=int, default=3)
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL") or "deepseek-v4-flash",
    )
    parser.add_argument("--base-url", default=None, help="默认读 OPENAI_BASE_URL")
    parser.add_argument("--api-key", default=None, help="默认读 OPENAI_API_KEY")
    parser.add_argument("--dry-run", action="store_true", help="只统计要补多少条，不调用模型")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    targets = [r for r in rows if has_paging(r)] if args.paging_only else rows
    pending_messages = sum(
        1
        for row in targets
        for message in row.get("messages") or []
        if message.get("role") == "assistant"
        and (message.get("tool_calls") or [])
        and not str(message.get("content") or "").strip()
    )
    print(f"总轨迹 {len(rows)}，选中 {len(targets)}，待补 assistant 回合 {pending_messages}")
    if args.dry_run:
        return 0

    base_url, api_key, source = resolve_credentials()
    args.base_url = args.base_url or base_url
    args.api_key = args.api_key or api_key
    if not args.base_url or not args.api_key:
        raise SystemExit(
            "需要老师模型的地址与密钥。任选一种：\n"
            "  1) 传参：--base-url https://api.deepseek.com/v1 --api-key sk-xxx\n"
            "  2) 导出环境变量：export OPENAI_BASE_URL=... OPENAI_API_KEY=...\n"
            "     （若存在 ~/.m2_collect.env：set -a && source ~/.m2_collect.env && set +a）\n"
            "注意别用 .env 里的 LLM_BASE_URL：那是评测用的本地 vLLM，不是老师模型。"
        )
    print(f"老师模型：{args.model} @ {args.base_url}（来源 {source}）")
    if looks_local(args.base_url):
        print(
            f"[warn] {args.base_url} 是本地地址。补思考必须走老师模型；\n"
            "       若这里接的是评测用的 vLLM，等于让 2B 学生模型给自己的轨迹写理由，\n"
            "       数据会静默变废。确认无误再继续（Ctrl+C 退出）。",
            file=sys.stderr,
        )

    from shopping_grpo.evaluation.rollout import OpenAIChatClient

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.output)
    todo = [r for r in targets if str(r.get("trajectory_id")) not in done]
    print(f"已完成 {len(done)}，本次处理 {len(todo)}")
    if not todo:
        return 0

    lock = threading.Lock()
    total_filled = 0
    with args.output.open("a", encoding="utf-8") as sink:

        def work(row: dict) -> tuple[dict, int]:
            client = OpenAIChatClient(
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                temperature=0.3,
                max_tokens=160,
                session_id=f"augment-{row.get('trajectory_id')}",
            )

            def ask(prompt: str) -> str:
                reply = client.complete([{"role": "user", "content": prompt}], [])
                return str((reply or {}).get("content") or "")

            return fill_reasons(row, ask, args.context_steps)

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(work, row): row for row in todo}
            for index, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                try:
                    result, filled = future.result()
                except Exception as exc:  # 单条失败不该毁掉整批
                    print(f"[warn] {row.get('trajectory_id')}: {exc}")
                    continue
                with lock:
                    sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                    sink.flush()
                    total_filled += filled
                if index % 20 == 0 or index == len(todo):
                    print(f"  进度 {index}/{len(todo)}，已补 {total_filled} 条理由")

    print(f"完成：补写 {total_filled} 条理由 → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
