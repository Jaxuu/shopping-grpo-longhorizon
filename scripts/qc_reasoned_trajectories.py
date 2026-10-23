#!/usr/bin/env python3
"""补写结果的质量抽检：语言、长度、信息密度、模板化、翻页理由是否说清。

为什么要单独一个脚本：2026-10-24 第一次补写后，813 条里大部分是**英文**
（prompt 明确写了「用中文」），而且最长一条 1,428 字、还有个填充句重复了 4 次。
这些都不能靠「抽 3 条肉眼看」发现——必须按池子口径统计。

关键是**分三组比**，否则会得出错误结论：

1. 原池非翻页轨迹（1,066 条）——这是语言口径的基准，补写要和它一致；
2. 原池翻页轨迹里**原本就有**的思考——同样算基准；
3. 新补的回合——要检查的对象。

只看第 3 组「有 82.9% 含数字」是没意义的，因为不知道基准长什么样。

用法：
    python scripts/qc_reasoned_trajectories.py \
        --baseline data/sft_m10r/capped/all.jsonl \
        --reasoned data/sft_m10r/capped/all_reasoned_paging.jsonl \
        --show 3
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

CJK = re.compile(r"[\u4e00-\u9fff]")
LATIN_WORD = re.compile(r"[A-Za-z]{3,}")
DIGIT = re.compile(r"\d")
# 判断「这一步翻页的理由是否说清了这页为什么不够」。中英文都要覆盖：
# 只写中文关键词会把英文理由全部误判成不合格（第一次抽检 59.2% 就是这么来的）。
INSUFFICIENT_MARKERS = (
    "缺",
    "没有",
    "不够",
    "不全",
    "超",
    "少",
    "满足",
    "no ",
    "not ",
    "none",
    "lack",
    "miss",
    "exceed",
    "over budget",
    "either",
    "only",
    "fewer",
)
MIN_LEN = 8


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def action_turns(row: dict) -> list[dict]:
    return [
        message
        for message in row.get("messages") or []
        if message.get("role") == "assistant" and (message.get("tool_calls") or [])
    ]


def tool_name(message: dict) -> str:
    calls = message.get("tool_calls") or []
    return str(((calls[0] or {}).get("function") or {}).get("name") or "?") if calls else "?"


def has_paging(row: dict) -> bool:
    return any(tool_name(m) == "next_page" for m in action_turns(row))


def language_of(text: str) -> str:
    """粗分三类：zh（含汉字）/ en（无汉字但有英文词）/ other。"""
    text = str(text or "").strip()
    if not text:
        return "empty"
    if CJK.search(text):
        return "zh"
    if LATIN_WORD.search(text):
        return "en"
    return "other"


def explains_insufficiency(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in INSUFFICIENT_MARKERS)


def summarize(label: str, turns: list[tuple[str, str]], show_top: int = 0) -> dict:
    """打印一组回合的统计，返回汇总数字供上层比较。"""

    contents = [text for _, text in turns if str(text or "").strip()]
    total = len(turns)
    filled = len(contents)
    empty_stats = {
        "filled": 0,
        "zh_ratio": 0.0,
        "en_ratio": 0.0,
        "digit_ratio": 0.0,
        "median_len": 0,
        "max_len": 0,
        "top_repeat": 0,
    }
    print(f"\n== {label}")
    print(f"   动作回合 {total}，有内容 {filled}（{filled / max(total, 1):.1%}）")
    if not filled:
        return empty_stats

    langs = collections.Counter(language_of(text) for text in contents)
    zh_ratio = langs["zh"] / filled
    en_ratio = langs["en"] / filled
    print(
        f"   语言：中文 {langs['zh']}（{zh_ratio:.1%}）"
        f"，英文 {langs['en']}（{en_ratio:.1%}）"
        f"，其它 {langs['other']}，空 {langs['empty']}"
    )
    lengths = sorted(len(text) for text in contents)
    print(
        f"   长度：中位 {lengths[len(lengths) // 2]}，最短 {lengths[0]}，"
        f"最长 {lengths[-1]}，超过 120 字 {sum(1 for n in lengths if n > 120)} 条"
    )
    digit_ratio = sum(1 for text in contents if DIGIT.search(text)) / filled
    print(f"   含数字（价格/规格/名次）：{digit_ratio:.1%}")

    paging = [text for name, text in turns if name == "next_page" and str(text or "").strip()]
    if paging:
        explained = sum(1 for text in paging if explains_insufficiency(text))
        print(f"   next_page 理由说明「这页不够」：{explained}/{len(paging)} = {explained / len(paging):.1%}")

    if show_top:
        repeated = collections.Counter(contents).most_common(show_top)
        print(f"   最高频 {show_top} 句：")
        for sentence, count in repeated:
            print(f"     {count:>3}x {sentence[:70]}")

    return {
        "filled": filled,
        "zh_ratio": zh_ratio,
        "en_ratio": en_ratio,
        "digit_ratio": digit_ratio,
        "median_len": lengths[len(lengths) // 2],
        "max_len": lengths[-1],
        "top_repeat": collections.Counter(contents).most_common(1)[0][1] if contents else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True, help="完整池（补写前）")
    parser.add_argument("--reasoned", type=Path, required=True, help="补写结果（只含被选中的轨迹）")
    parser.add_argument("--show", type=int, default=0, help="打印最高频几句 + 几条样本")
    args = parser.parse_args()

    baseline = read_jsonl(args.baseline)
    reasoned = read_jsonl(args.reasoned)
    reasoned_ids = {str(row.get("trajectory_id")) for row in reasoned}

    plain = [row for row in baseline if not has_paging(row)]
    paging = [row for row in baseline if has_paging(row)]

    base_plain = [(tool_name(m), m.get("content") or "") for row in plain for m in action_turns(row)]
    base_paging = [
        (tool_name(m), m.get("content") or "")
        for row in paging
        for m in action_turns(row)
        if str(m.get("content") or "").strip()
    ]

    # 新补的回合 = 补写结果里「原本为空」的那些；原本有内容的老师原话单独算，
    # 因为它们不是这次生成的，混在一起会高估补写质量。
    new_turns: list[tuple[str, str]] = []
    kept_turns: list[tuple[str, str]] = []
    original_paging = {str(row.get("trajectory_id")): row for row in paging}
    for row in reasoned:
        key = str(row.get("trajectory_id"))
        origin = original_paging.get(key)
        if origin is None:
            new_turns.extend((tool_name(m), m.get("content") or "") for m in action_turns(row))
            continue
        before_turns = action_turns(origin)
        after_turns = action_turns(row)
        if len(before_turns) != len(after_turns):
            print(f"[warn] {key} 的动作回合数变了：{len(before_turns)} → {len(after_turns)}", flush=True)
        for old, new in zip(before_turns, after_turns):
            if str(old.get("content") or "").strip():
                kept_turns.append((tool_name(new), new.get("content") or ""))
            else:
                new_turns.append((tool_name(new), new.get("content") or ""))

    print(f"池子 {len(baseline)} 条（非翻页 {len(plain)} / 翻页 {len(paging)}）；补写结果 {len(reasoned)} 条")
    base_stats = summarize("原池 · 非翻页轨迹（语言口径基准）", base_plain, args.show)
    paging_stats = summarize("原池 · 翻页轨迹里原本就有的思考", base_paging)
    kept_stats = summarize("翻页轨迹里被保留的老师原话", kept_turns)
    new_stats = summarize("本次新补的回合（检查对象）", new_turns, args.show)

    print("\n== 结论")
    reference = base_stats["zh_ratio"]
    if not new_stats["filled"]:
        print("   – 没有新补的回合（补写结果与池子一致？检查 --reasoned 是否指错文件）")
        return 0
    if reference >= 0.8 and new_stats["zh_ratio"] < 0.8:
        print(
            f"   ✗ 语言不一致：原池非翻页轨迹 {reference:.0%} 是中文，"
            f"新补的只有 {new_stats['zh_ratio']:.0%} 是中文。"
            "\n     重跑：加 --force --language zh（prompt 已加硬校验，会自动重问）"
        )
    elif base_stats["en_ratio"] >= 0.8 and new_stats["en_ratio"] < 0.8:
        print(
            f"   ✗ 语言不一致：原池以英文为主（{base_stats['en_ratio']:.0%}），"
            f"新补的英文只有 {new_stats['en_ratio']:.0%}。"
            "\n     重跑：加 --force --language en"
        )
    else:
        print(f"   ✓ 语言与原池一致（原池中文 {reference:.0%}，新补中文 {new_stats['zh_ratio']:.0%}）")

    if new_stats["digit_ratio"] < 0.6:
        print(f"   ✗ 含数字比例偏低（{new_stats['digit_ratio']:.0%} < 60%）：理由没引用页面上的具体信息")
    else:
        print(f"   ✓ 含数字 {new_stats['digit_ratio']:.0%}（≥60%）")

    if new_stats["max_len"] > 200:
        print(f"   ✗ 最长一条 {new_stats['max_len']} 字：截断没生效或 prompt 被绕过")
    else:
        print(f"   ✓ 最长 {new_stats['max_len']} 字，中位 {new_stats['median_len']}")

    if new_stats["top_repeat"] > 3:
        print(f"   ✗ 同一句重复 {new_stats['top_repeat']} 次：老师在套模板")
    else:
        print(f"   ✓ 最高频句只重复 {new_stats['top_repeat']} 次")

    if args.show:
        print(f"\n== 抽 {args.show} 条新补的（含前后两个回合的上下文）")
        shown = 0
        for row in reasoned:
            key = str(row.get("trajectory_id"))
            origin = original_paging.get(key)
            if origin is None:
                continue
            before_turns = action_turns(origin)
            after_turns = action_turns(row)
            if not any(not str(m.get("content") or "").strip() for m in before_turns):
                continue
            request = next(
                (m.get("content") for m in row.get("messages") or [] if m.get("role") == "user"), ""
            )
            print(f"\n== {key} | {' '.join(str(request).split())[:70]}")
            for old, new in zip(before_turns, after_turns):
                tag = "补" if not str(old.get("content") or "").strip() else "原"
                print(f"   [{tag}] {tool_name(new)} | {str(new.get('content') or '')[:110]}")
            shown += 1
            if shown >= args.show:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
