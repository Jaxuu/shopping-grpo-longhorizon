#!/usr/bin/env python3
"""在固定 ShopSimulator benchmark 上评测 OpenAI-compatible 本地或远端模型。"""

import argparse
import json
from pathlib import Path

from shopping_grpo.environment.context import LocalTextTokenCounter
from shopping_grpo.evaluation.summary import summarize_trajectories
from shopping_grpo.evaluation.rollout import (
    SYSTEM_PROMPT as FROZEN_SYSTEM_PROMPT,
    OpenAIChatClient,
    collect_tasks,
    load_system_prompt,
    load_tasks,
    system_prompt_digest,
)


def parse_args():
    parser = argparse.ArgumentParser(description="评测 Base、SFT 或 GRPO Shopping Agent")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="原始评测轨迹 JSONL")
    parser.add_argument("--summary", type=Path, required=True, help="汇总指标 JSON")
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--model", required=True)
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--api-key", required=True, help="本地 vLLM 可传 EMPTY")
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="单次模型生成上限；防止未调用工具时耗尽完整上下文。",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=24576,
        help="上下文窗口；传 0 禁用 vLLM /tokenize 依赖。",
    )
    parser.add_argument("--context-safety-margin", type=int, default=512)
    parser.add_argument(
        "--context-compaction",
        action="store_true",
        help="上下文接近上限时压缩较早的交互；默认关闭。",
    )
    parser.add_argument(
        "--observation-token-budget",
        type=int,
        default=1536,
        help="Observation token 预算；传 0 禁用 vLLM /tokenize 依赖。",
    )
    parser.add_argument("--observation-detail-token-budget", type=int, default=4096)
    parser.add_argument("--observation-generic-token-budget", type=int, default=768)
    parser.add_argument("--observation-search-top-k", type=int, default=20)
    parser.add_argument(
        "--tokenizer-path",
        default="",
        help=(
            "本地 HuggingFace tokenizer 目录，用于 observation 预算计数。"
            "评测远端 API 模型（如老师）时模型服务没有 /tokenize，必须用它才能保持"
            "与本地模型同口径的截断；留空则维持原行为（走 vLLM /tokenize）。"
        ),
    )
    parser.add_argument(
        "--min-opened-before-buy",
        type=int,
        default=0,
        help=(
            "诊断开关：买之前必须打开过多少个不同候选，默认 0=关闭。"
            # argparse 会对 help 做 % 格式化，裸百分号必须写成 %%。
            "开启后评测分数不可与历史 62.0%% 直接比较，它只用于测量探索收益上界。"
        ),
    )
    parser.add_argument(
        "--system-prompt-file",
        default="",
        help=(
            "系统提示词变体文件（UTF-8 全文）。留空 = 用冻结的 SYSTEM_PROMPT。"
            "用于「同一权重、只改提示词」的 A/B；换提示词等于换 agent 配置，"
            "分数不能与冻结口径的历史分数直接比，protocol 里会记指纹。"
        ),
    )
    return parser.parse_args()


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    args = parse_args()
    if args.max_steps < 1:
        raise SystemExit("--max-steps 必须为正数")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens 必须为正数")
    if args.context_window < 0:
        raise SystemExit("--context-window 不能为负数")
    if args.context_window and args.context_window <= args.max_tokens + args.context_safety_margin:
        raise SystemExit("--context-window 必须大于 --max-tokens 与安全余量之和")
    if args.observation_token_budget < 0:
        raise SystemExit("--observation-token-budget 不能为负数")
    if args.tokenizer_path and args.context_window:
        # 本地 tokenizer 只负责 observation 的纯文本计数；chat 级上下文预算依赖
        # 服务端的 chat template 渲染，本地复现容易与 vLLM 不一致，所以不混用。
        raise SystemExit(
            "--tokenizer-path 只替代 observation 文本计数，不能替代 chat 级上下文计数；"
            "评测远端 API 模型时请同时传 --context-window 0"
        )
    observation_token_counter = (
        LocalTextTokenCounter(args.tokenizer_path) if args.tokenizer_path else None
    )
    system_prompt = load_system_prompt(args.system_prompt_file or None)
    tasks = load_tasks(args.benchmark)
    client = OpenAIChatClient(
        model=args.model,
        base_url=args.llm_base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        context_window=args.context_window,
        context_safety_margin=args.context_safety_margin,
        context_compaction_enable=args.context_compaction,
        observation_token_budget=args.observation_token_budget,
        observation_detail_token_budget=args.observation_detail_token_budget,
        observation_generic_token_budget=args.observation_generic_token_budget,
        observation_search_top_k=args.observation_search_top_k,
        observation_token_counter=observation_token_counter,
    )
    collect_tasks(
        tasks,
        client=client,
        output_path=args.output,
        base_url=args.base_url,
        max_steps=args.max_steps,
        min_opened_before_buy=args.min_opened_before_buy,
        system_prompt=system_prompt,
    )
    summary = summarize_trajectories(
        [task["task_id"] for task in tasks], _read_jsonl(args.output)
    )
    summary["protocol"] = {
        "benchmark": str(args.benchmark),
        "model": args.model,
        "reward_contract": "shopsimulator-reward-v3",
        "max_steps": args.max_steps,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "context_window": args.context_window,
        "context_safety_margin": args.context_safety_margin,
        "context_compaction": args.context_compaction,
        "observation_token_budget": args.observation_token_budget,
        "observation_detail_token_budget": args.observation_detail_token_budget,
        "observation_generic_token_budget": args.observation_generic_token_budget,
        "observation_search_top_k": args.observation_search_top_k,
        "observation_tokenizer": args.tokenizer_path or "vllm:/tokenize",
        "min_opened_before_buy": args.min_opened_before_buy,
        "system_prompt_file": args.system_prompt_file or "frozen:SYSTEM_PROMPT",
        "system_prompt_sha256": system_prompt_digest(
            system_prompt if system_prompt is not None else FROZEN_SYSTEM_PROMPT
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
