"""诊断 M10-R v3 的行为塌缩：翻页从 v2 的 33% 掉到 0%，到底是「数据没进梯度」还是「理由教错了方向」。

两个独立问题，一次跑完：

A. 机械问题：补了理由的翻页轨迹变长，会不会整条被 8192 截掉（训练日志里 dropped=28）？
   如果被丢的 28 条里翻页轨迹占多数，那 v3 等于「翻页示范被删掉了」，属于工程事故，
   修法只是调 max_length 或压短理由，值得重跑。

B. 内容问题：补进 next_page 回合的理由，是在教「该翻页」还是在教「该换词重搜」？
   老师模型自己的偏好就是换词重搜（采集探针 48% vs 8%），如果它给翻页写的理由
   多数在讲「重新搜索」，学生学到的就是换词 —— 这属于方向错，重跑同样的数据没用。

用法（服务器上，不需要显卡）：

    .venv/bin/python scripts/diagnose_v3_collapse.py \
        --source data/sft_m10r/capped/v3/all.jsonl \
        --manifest data/sft_m10r/capped/v3/manifest.json \
        --base-model outputs/models/sft-b-curriculum/stage-c/merged \
        --max-length 8192

加 ``--skip-tokenize`` 只跑 B 部分（几秒出结果）。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PAGE_WORDS = (
    "翻页",
    "下一页",
    "下页",
    "第二页",
    "第三页",
    "第2页",
    "第3页",
    "往后",
    "继续看",
    "后面的",
    "更多结果",
    "next page",
)
REQUERY_WORDS = (
    "重新搜索",
    "重新搜",
    "改搜",
    "再搜",
    "换关键词",
    "换词",
    "换个词",
    "新关键词",
    "调整关键词",
    "缩小范围",
    "search again",
    "re-search",
    "new query",
    "different query",
)


def action_turns(row: dict) -> list[dict]:
    return [
        message
        for message in row.get("messages") or []
        if message.get("role") == "assistant" and (message.get("tool_calls") or [])
    ]


def tool_name(message: dict) -> str:
    calls = message.get("tool_calls") or []
    if not calls:
        return "?"
    return str(((calls[0] or {}).get("function") or {}).get("name") or "?")


def has_paging(row: dict) -> bool:
    return any(tool_name(message) == "next_page" for message in action_turns(row))


def hits(text: str, words) -> bool:
    lowered = str(text or "").lower()
    return any(word.lower() in lowered for word in words)


def load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def part_b(rows: list[dict]) -> dict:
    """理由内容方向：翻页回合在讲翻页，还是在讲换词重搜。"""
    print("\n== B. 理由在教什么（文本分析，不需要模型）")
    next_page_reasons: list[str] = []
    other_reasons: list[str] = []
    for row in rows:
        for message in action_turns(row):
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            if tool_name(message) == "next_page":
                next_page_reasons.append(content)
            else:
                other_reasons.append(content)

    def report(name: str, reasons: list[str]) -> dict:
        total = len(reasons)
        if not total:
            print(f"   {name}: 0 条")
            return {"total": 0, "page": 0.0, "requery": 0.0}
        page = sum(1 for text in reasons if hits(text, PAGE_WORDS))
        requery = sum(1 for text in reasons if hits(text, REQUERY_WORDS))
        print(
            f"   {name}: {total} 条 | 提到翻页 {page}（{page / total:.1%}）"
            f" | 提到换词重搜 {requery}（{requery / total:.1%}）"
        )
        return {"total": total, "page": page / total, "requery": requery / total}

    stats = {
        "next_page": report("next_page 回合的理由", next_page_reasons),
        "other": report("其它动作回合的理由", other_reasons),
    }

    if next_page_reasons:
        page = stats["next_page"]["page"]
        requery = stats["next_page"]["requery"]
        print("\n   判定：", end="")
        if page >= 0.6 and requery <= 0.4:
            print("翻页理由以「该翻页」为主 —— 方向对，问题不在内容。")
        elif requery > page:
            print(
                "翻页理由以「换词重搜」为主 —— 学生在按理由行动，学到的是换词，"
                "这解释了 v3 换词涨到 2.5%、翻页归零。重跑同款数据无用。"
            )
        else:
            print("两种说法混杂 —— 学生拿到的是含糊示范，需要改写成单一指向。")
    return stats


def part_a(rows: list[dict], source: Path, manifest: dict, base_model: str, max_length: int) -> None:
    """机械问题：哪些训练行被 8192 截掉了，里面有没有翻页轨迹。"""
    print("\n== A. 超长丢弃（复现训练的 tokenize 口径）")
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    load_kwargs = {"trust_remote_code": True}
    config = AutoConfig.from_pretrained(base_model, **load_kwargs)
    if str(getattr(config, "model_type", "")).startswith("qwen3_5"):
        processor = AutoProcessor.from_pretrained(base_model, **load_kwargs)
        tokenizer, template = processor.tokenizer, processor
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, **load_kwargs)
        template = tokenizer

    train_ids: set[int] = set()
    for bucket in (manifest.get("buckets") or {}).values():
        train_ids.update(int(task_id) for task_id in bucket.get("train_task_ids") or [])
    if not train_ids:
        print("   [warn] manifest 里没有 train_task_ids，跳过 A")
        return

    from shopping_grpo.training.sft.dataset import load_supervised_examples

    examples, stats = load_supervised_examples(
        source,
        tokenizer,
        max_length=max_length,
        chat_template=template,
        task_ids=train_ids,
        loss_scope="tool-call-only",
    )
    print(f"   {stats}")

    kept = {str(example.get("trajectory_id")) for example in examples}
    paging_rows = {str(row.get("trajectory_id")): row for row in rows if has_paging(row)}
    dropped = [
        row
        for row in rows
        if int(row.get("task_id", -1)) in train_ids
        and str(row.get("trajectory_id")) not in kept
    ]
    dropped_paging = [row for row in dropped if has_paging(row)]
    print(
        f"   训练行 {len(train_ids)}，丢弃 {len(dropped)}，其中翻页轨迹 {len(dropped_paging)}"
        f"（翻页轨迹共 {len(paging_rows)} 条）"
    )
    if dropped:
        print("   被丢的行（task_id / 工具步 / 是否翻页）：")
        for row in sorted(dropped, key=lambda r: -len(action_turns(r)))[:15]:
            print(
                f"     {row.get('task_id'):>6} | {len(action_turns(row)):>2} 步"
                f" | {'翻页' if has_paging(row) else '普通'}"
            )

    lengths_paging = [
        len(example["input_ids"])
        for example in examples
        if str(example.get("trajectory_id")) in paging_rows
    ]
    lengths_plain = [
        len(example["input_ids"])
        for example in examples
        if str(example.get("trajectory_id")) not in paging_rows
    ]
    for name, values in (("翻页轨迹", lengths_paging), ("普通轨迹", lengths_plain)):
        if not values:
            continue
        values = sorted(values)
        print(
            f"   {name} {len(values)} 条：中位 {statistics.median(values):.0f}"
            f"，P90 {values[int(len(values) * 0.9)]}，最长 {values[-1]}"
        )


def part_c(paths: list[Path]) -> None:
    """评测轨迹里模型有没有「先说话再动手」——判断完整监督到底有没有生效。"""
    print("\n== C. 评测轨迹：模型开始说话了吗")
    print("   文件 | 轨迹 | 动作回合 | 有内容 | 中位长度 | 翻页")
    for path in paths:
        if not path.exists():
            print(f"   {path.name}: 不存在，跳过")
            continue
        rows = load_rows(path)
        turns = [message for row in rows for message in action_turns(row)]
        contents = [str(m.get("content") or "").strip() for m in turns]
        filled = [text for text in contents if text]
        median = statistics.median([len(text) for text in filled]) if filled else 0
        paging = sum(1 for row in rows if has_paging(row))
        print(
            f"   {path.parent.name:>12} | {len(rows):>3} | {len(turns):>4} | {len(filled):>4}"
            f"（{len(filled) / max(len(turns), 1):.1%}） | {median:.0f} | {paging}"
        )
        if filled:
            print(f"     开头几句：{filled[0][:60]} / {filled[len(filled) // 2][:60]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--base-model", default="outputs/models/sft-b-curriculum/stage-c/merged")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--skip-tokenize", action="store_true", help="只跑 B 部分")
    parser.add_argument("--eval", type=Path, action="append", default=[], help="评测轨迹，可重复")
    args = parser.parse_args()

    rows = load_rows(args.source)
    paging = sum(1 for row in rows if has_paging(row))
    print(f"== 池子 {len(rows)} 条，其中含 next_page 的 {paging} 条")

    part_b(rows)
    if args.eval:
        part_c(args.eval)

    if args.skip_tokenize:
        return 0
    if args.manifest is None or not args.manifest.exists():
        print("\n[warn] 没有 --manifest，跳过 A（超长丢弃检查）")
        return 0
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    part_a(rows, args.source, manifest, args.base_model, args.max_length)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
