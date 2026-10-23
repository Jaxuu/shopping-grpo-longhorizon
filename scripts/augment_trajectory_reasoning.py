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
import re
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

PROMPT = """你在复现一条购物任务的轨迹，需要为其中一步补一句简短的说明。

【购物需求】
{instruction}

【最近看到的页面与操作】
{history}

【你这一步的决定】
{action}

请用**一句话**（30–60 字）说明你为什么这么做。要求：
- {language_rule}
- 必须引用页面里的具体信息（价格、规格、属性、排名、店铺），不要泛泛而谈；
- 如果这一步是翻页，要说明这一页为什么不够（缺什么条件）；
- 只输出这一句话，不要解释、不要加引号、不要换行、不要分点。"""

# 语言口径必须和池子里原有轨迹一致，否则模型会学到「一到翻页情形就换语言」
# 这个虚假相关。2026-10-24 实测：prompt 只写了「用中文」，老师仍然大段输出英文
# （"I'll search for a replacement AC compressor..."），所以这里既写死规则，
# 也在 ``usable_reason`` 里做硬校验 + 重试。
LANGUAGE_RULES = {
    "zh": "**必须用简体中文**：整句都是中文，商品名可保留原文，但不要出现英文句子；",
    "en": "**必须用英文**（与池子里其它轨迹保持一致），不要出现中文句子；",
}

RETRY_HINT = (
    "\n\n上一次的回答不合格（语言不对或太长）。请只输出**一句话**，严格遵守：\n"
    "- 语言必须符合上面的要求；\n"
    "- 不要换行、不要分点、不要引号、不要「理由：」这类前缀；\n"
    "- 长度 30–60 字。"
)

# 理由上限。prompt 要 30–60 字，实测有 1,428 字的离谱样本（占满 max_tokens 后
# 还带分点），会白白拉长训练序列，所以按句截断再硬切。
MAX_REASON_CHARS = 120

CJK = re.compile(r"[\u4e00-\u9fff]")
LATIN_WORD = re.compile(r"[A-Za-z]{3,}")
SENTENCE_END = re.compile(r"(?<=[。！？])")


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


def build_prompt(
    instruction: str,
    recent: list[tuple[str, str]],
    action: str,
    limit: int,
    language: str = "zh",
) -> str:
    if language not in LANGUAGE_RULES:
        raise ValueError(f"language must be one of {tuple(LANGUAGE_RULES)}, got {language!r}")
    return PROMPT.format(
        instruction=instruction.strip()[:1200],
        history=render_history(recent, limit),
        action=action,
        language_rule=LANGUAGE_RULES[language],
    )


def clean_reply(text: str) -> str:
    """去掉模型爱加的引号/前缀，只留一句。"""

    line = " ".join(str(text or "").split())
    for prefix in ("理由：", "理由:", "说明：", "说明:"):
        if line.startswith(prefix):
            line = line[len(prefix) :].strip()
    return line.strip().strip('"“”「」').strip()


def trim_reason(text: str) -> str:
    """截到第一句，再按 ``MAX_REASON_CHARS`` 硬切。

    实测有 1,428 字、带分点的样本（老师没理会「一句话」的要求），
    这种理由会把训练序列拉长却不多带信息，必须掐掉。
    """

    line = clean_reply(text)
    first = SENTENCE_END.split(line, maxsplit=1)[0].strip()
    if first:
        line = first
    if len(line) > MAX_REASON_CHARS:
        line = line[:MAX_REASON_CHARS].rstrip("，,、；;：: ")
    return line


def usable_reason(text: str, language: str = "zh") -> bool:
    """理由是否达标：长度够、语言对、不是纯填充句。

    语言这一条是硬要求。prompt 里写「用中文」时老师仍会大段输出英文
    （2026-10-24 实测 813 条里大部分是英文），不校验就会让「翻页情形 ⇒ 说英文」
    变成一条虚假相关混进训练数据。
    """

    line = str(text or "").strip()
    if len(line) < 8:
        return False
    has_cjk = bool(CJK.search(line))
    has_latin = bool(LATIN_WORD.search(line))
    if language == "zh":
        return has_cjk
    if language == "en":
        return has_latin and not has_cjk
    return True


def ask_reason(ask, prompt: str, language: str = "zh", attempts: int = 3) -> tuple[str, bool]:
    """问一次理由；不合格就带上更强硬的提示重问。返回 (理由, 是否达标)。

    三次都不达标时返回**最好的一次**（第一次非空结果），而不是空串——
    留空会让合并脚本报缺口、整批卡住；达标与否由调用方统计并打印，
    让人知道有几条是兜底的。
    """

    fallback = ""
    for attempt in range(max(1, attempts)):
        reply = trim_reason(ask(prompt if attempt == 0 else prompt + RETRY_HINT))
        if not reply:
            continue
        if not fallback:
            fallback = reply
        if usable_reason(reply, language):
            return reply, True
    return fallback, False


def fill_reasons(
    row: dict, ask, context_steps: int = 3, language: str = "zh"
) -> tuple[dict, int, int]:
    """给 content 为空的 assistant 回合补理由。

    返回 ``(row, 补了几条, 其中兜底几条)``。兜底 = 重问三次仍不合格，
    通常意味着该回合的页面信息太贫乏。
    """

    instruction = instruction_of(row)
    recent: list[tuple[str, str]] = []
    last_observation = ""
    filled = 0
    fallback = 0
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
            prompt = build_prompt(instruction, recent, action, context_steps, language)
            reply, ok = ask_reason(ask, prompt, language)
            if reply:
                message["content"] = reply
                filled += 1
                if not ok:
                    fallback += 1
        recent.append((action, last_observation))
    return row, filled, fallback


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
    parser.add_argument(
        "--language",
        choices=tuple(LANGUAGE_RULES),
        default="zh",
        help="理由语言，必须与池子里原有轨迹一致（默认 zh）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略已有输出，全部重跑（换了 prompt 或语言口径后用这个）",
    )
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
    if args.force and args.output.is_file():
        # 换 prompt / 换语言口径后必须重跑；直接截断，否则会按 trajectory_id 全跳过。
        args.output.unlink()
        print(f"[force] 已清空旧输出：{args.output}")
    done = load_done(args.output)
    todo = [r for r in targets if str(r.get("trajectory_id")) not in done]
    print(f"理由语言：{args.language}；已完成 {len(done)}，本次处理 {len(todo)}")
    if not todo:
        return 0

    lock = threading.Lock()
    total_filled = 0
    total_fallback = 0
    with args.output.open("a", encoding="utf-8") as sink:

        def work(row: dict) -> tuple[dict, int, int]:
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

            return fill_reasons(row, ask, args.context_steps, args.language)

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(work, row): row for row in todo}
            for index, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                try:
                    result, filled, fallback = future.result()
                except Exception as exc:  # 单条失败不该毁掉整批
                    print(f"[warn] {row.get('trajectory_id')}: {exc}")
                    continue
                with lock:
                    sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                    sink.flush()
                    total_filled += filled
                    total_fallback += fallback
                if index % 20 == 0 or index == len(todo):
                    print(f"  进度 {index}/{len(todo)}，已补 {total_filled} 条理由")

    print(f"完成：补写 {total_filled} 条理由 → {args.output}")
    if total_fallback:
        # 兜底条数就是「重问三次仍不合格」的回合数。它们会进池子，但语言/长度可能
        # 不合口径，所以必须显式报出来，而不是混在「813 条全部补上」里。
        print(
            f"[warn] 其中 {total_fallback} 条重问三次仍不合格（语言或长度），已按最好的一次保留；"
            "\n       跑完用质量抽检命令看一眼这些是否要单独处理。",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
