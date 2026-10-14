#!/usr/bin/env python3
"""M5 离线预检：用真实 chat template 验证 loss-scope 的标签边界。

本脚本**不训练、不加载模型权重**，只加载 tokenizer / processor，并复用
``shopping_grpo.training.sft.dataset.build_supervised_example``，对 Pure V4 池的
若干行断言三件事：

1. 两个口径都能为每一行产出可训练样本。**例外**：超过 ``--max-length`` 的行会被
   ``build_supervised_example`` 有意丢弃（不做可能截断工具调用的截断），这与训练行为一致。
   脚本会用放宽后的 ``max_length`` 复检一次来区分「超长丢弃」（计入 ``over_long_drops``，
   **不算失败**）与「真缺陷」（计入 ``failures``，判失败）；
2. ``tool-call-only`` 仍完整保留 tool call（函数名逐个出现在被监督的 token 里）；
3. 对有自然语言 content 的行，``tool-call-only`` **不**监督这些 content，且监督
   token 数严格少于 ``assistant`` 口径。

任一条不成立即以非零码退出 —— 符合 P8：loss 口径必须先有离线证据，才能进训练。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FULL = ROOT / "data/sft_pure_v4/all.jsonl"
DEFAULT_TOOLONLY = ROOT / "data/sft_pure_v4/all_toolonly.jsonl"
DEFAULT_REPORT = ROOT / "plans/m5-loss-scope-preflight.json"
SNIPPET_CHARS = 24
#: 复检「超长丢弃」用的放宽上限；只用于判定丢弃原因，不参与真实训练。
UNBOUNDED_MAX_LENGTH = 10_000_000


def _load_template(model: str, revision: str | None = None):
    """按目标模型选择 chat template 的持有者（与 train_lora_sft 保持一致）。"""
    try:
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - 依赖缺失时给出可执行指引
        raise SystemExit(
            "缺少 transformers；请先在仓库根执行：bash scripts/setup.sh"
        ) from exc

    load_kwargs = {"trust_remote_code": True}
    if revision:
        load_kwargs["revision"] = revision
    config = AutoConfig.from_pretrained(model, **load_kwargs)
    if str(getattr(config, "model_type", "")).startswith("qwen3_5"):
        processor = AutoProcessor.from_pretrained(model, **load_kwargs)
        return processor.tokenizer, processor
    tokenizer = AutoTokenizer.from_pretrained(model, **load_kwargs)
    return tokenizer, tokenizer


def _read_rows(path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if len(rows) >= limit:
            break
    return rows


def _content_snippets(messages: list[dict]) -> list[str]:
    snippets = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = (message.get("content") or "").strip()
        if len(content) >= SNIPPET_CHARS:
            snippets.append(content[:SNIPPET_CHARS])
    return snippets


def _function_names(messages: list[dict]) -> list[str]:
    names = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name")
            if name:
                names.append(name)
    return names


def _supervised_tokens(example: dict) -> int:
    return sum(1 for token in example["labels"] if token != -100)


def _labeled_text(tokenizer, example: dict) -> str:
    return tokenizer.decode(
        [token for token in example["labels"] if token != -100],
        skip_special_tokens=False,
    )


def _check_file(path: Path, limit: int, tokenizer, template, max_length: int) -> dict:
    from shopping_grpo.training.sft.dataset import build_supervised_example

    rows = _read_rows(path, limit)
    failures: list[dict] = []
    over_long_drops = 0
    total_assistant_tokens = 0
    total_toolonly_tokens = 0
    rows_with_content = 0

    for row in rows:
        messages = row["messages"]
        tools = row.get("tools") or []
        task_id = row.get("task_id")

        examples = {}
        for scope in ("assistant", "tool-call-only"):
            example = build_supervised_example(
                messages=messages,
                tools=tools,
                tokenizer=tokenizer,
                max_length=max_length,
                chat_template=template,
                loss_scope=scope,
            )
            if example is None:
                # 区分「超长丢弃」（by design：不做可能截断工具调用的截断，训练同样会丢）
                # 与「真缺陷」。放宽 max_length 复检：能训成 → 超长；仍失败 → 真缺陷。
                retry = build_supervised_example(
                    messages=messages,
                    tools=tools,
                    tokenizer=tokenizer,
                    max_length=UNBOUNDED_MAX_LENGTH,
                    chat_template=template,
                    loss_scope=scope,
                )
                if retry is not None:
                    over_long_drops += 1
                else:
                    failures.append(
                        {"task_id": task_id, "reason": f"{scope}: dropped (非超长)"}
                    )
            examples[scope] = example

        assistant_example = examples["assistant"]
        toolonly_example = examples["tool-call-only"]
        if assistant_example is None or toolonly_example is None:
            continue

        assistant_tokens = _supervised_tokens(assistant_example)
        toolonly_tokens = _supervised_tokens(toolonly_example)
        total_assistant_tokens += assistant_tokens
        total_toolonly_tokens += toolonly_tokens

        toolonly_text = _labeled_text(tokenizer, toolonly_example)
        for name in _function_names(messages):
            if name not in toolonly_text:
                failures.append(
                    {"task_id": task_id, "reason": f"tool-call-only 丢失 tool call: {name}"}
                )

        snippets = _content_snippets(messages)
        if snippets:
            rows_with_content += 1
            for snippet in snippets:
                if snippet in toolonly_text:
                    failures.append(
                        {
                            "task_id": task_id,
                            "reason": f"tool-call-only 仍在监督 assistant content: {snippet!r}",
                        }
                    )
            if toolonly_tokens >= assistant_tokens:
                failures.append(
                    {
                        "task_id": task_id,
                        "reason": (
                            "tool-call-only 监督 token 数未少于 assistant 口径 "
                            f"({toolonly_tokens} vs {assistant_tokens})"
                        ),
                    }
                )
        if toolonly_tokens <= 0:
            failures.append({"task_id": task_id, "reason": "tool-call-only 无监督 token"})

    return {
        "path": str(Path(path).relative_to(ROOT).as_posix())
        if str(path).startswith(str(ROOT))
        else str(path),
        "rows_checked": len(rows),
        "rows_with_assistant_content": rows_with_content,
        "over_long_drops": over_long_drops,
        "supervised_tokens": {
            "assistant": total_assistant_tokens,
            "tool_call_only": total_toolonly_tokens,
            "ratio": (
                round(total_toolonly_tokens / total_assistant_tokens, 4)
                if total_assistant_tokens
                else None
            ),
        },
        "failures": failures[:20],
        "failure_count": len(failures),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B", help="tokenizer / processor 来源")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--toolonly", type=Path, default=DEFAULT_TOOLONLY)
    parser.add_argument("--limit", type=int, default=200, help="每个文件抽查的行数")
    parser.add_argument("--max-length", type=int, default=24576)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer, template = _load_template(args.model, args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    report = {
        "schema_version": "shopping-m5-loss-scope-preflight-v1",
        "model": args.model,
        "max_length": args.max_length,
        "snippet_chars": SNIPPET_CHARS,
        "files": {},
    }
    failed = False
    for key, path in (("full", args.full), ("toolonly", args.toolonly)):
        result = _check_file(path, args.limit, tokenizer, template, args.max_length)
        report["files"][key] = result
        failed = failed or result["failure_count"] > 0

    report["status"] = "failed" if failed else "passed"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({k: v for k, v in report.items() if k != "files"}, ensure_ascii=False, indent=2))
    for key, result in report["files"].items():
        print(
            f"[{key}] rows={result['rows_checked']} "
            f"tokens(assistant/tool-call-only)="
            f"{result['supervised_tokens']['assistant']}/"
            f"{result['supervised_tokens']['tool_call_only']} "
            f"ratio={result['supervised_tokens']['ratio']} "
            f"over_long_drops={result['over_long_drops']} "
            f"failures={result['failure_count']}"
        )
        for failure in result["failures"]:
            print("  -", failure)
    print("report:", args.report)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
