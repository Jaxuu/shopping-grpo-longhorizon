#!/usr/bin/env python3
"""训练过的题为什么还做错：是「照做都做不到」，还是「做完了挑错」？

为什么不用「再训几个 epoch」来判
--------------------------------
m10r 只训了 1 个 epoch，末段 loss 在 0.06–0.21 之间震荡，看不出趋势；sft-b 训满
3 个 epoch 到 0.043，考卷反而只有 62.0%（m10r 1 个 epoch 是 63.5%）。所以「加 epoch
能不能救」用 loss 判不了，只能花钱试——但有一个**免费**的直接测法。

测什么
------
训练池里每道题都有老师（deepseek-v4-flash）的完整动作序列，而且只有 gold_purchase
才被收进池子。所以可以逐回合对齐：**模型在同一个观测下，做出的动作和老师一样吗？**

第一次分歧出现的位置决定了病因：

* 第 1–2 步就分叉（连搜索词/该打开哪个都不同）⇒ **照做都做不到** ⇒ 模仿/容量问题，
  加 epoch 或加数据**可能**有用；
* 跟到中后段，在 `open_product` / `select_option` 处分叉 ⇒ **流程会走，最后挑错**
  ⇒ 判别问题，加 epoch 只会让它背下这道题的答案（自测涨、dev 不涨），**无用**。

只在**第一次**分歧之前比较：分叉之后两边的观测已经不同，再对齐没有意义。

用法::

    .venv/bin/python scripts/diagnose_imitation.py \\
        --run outputs/evaluation/selftest-300 \\
        --pool data/sft_m10r/capped/all.jsonl
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = ROOT / "data" / "sft_m10r" / "capped" / "all.jsonl"

QUERY_KEYS = {"query"}
PRODUCT_KEYS = {"asin"}
OPTION_KEYS = {"value", "option", "option_value", "key"}
EARLY_STEPS = 2  # 前两步内分叉 = 「照做都做不到」


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canon(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split()).casefold()


def canon_args(args: Any) -> dict[str, str]:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return {"__raw__": canon(args)}
    if not isinstance(args, Mapping):
        return {}
    return {str(key): canon(value) for key, value in args.items()}


def teacher_actions(row: Mapping[str, Any]) -> list[tuple[str, dict[str, str]]]:
    """从训练池的一行里抽老师的动作序列。"""
    actions: list[tuple[str, dict[str, str]]] = []
    for message in row.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            actions.append(
                (str(function.get("name")), canon_args(function.get("arguments")))
            )
    return actions


def model_actions(record: Mapping[str, Any]) -> list[tuple[str, dict[str, str]]]:
    actions: list[tuple[str, dict[str, str]]] = []
    for step in record.get("steps") or []:
        actions.append((str(step.get("tool_name")), canon_args(step.get("parameters"))))
    return actions


def is_success(record: Mapping[str, Any]) -> bool:
    detail = ((record.get("terminal_result") or {}).get("reward_detail")) or {}
    return detail.get("purchase_success") is True or detail.get("reward_type") == "gold_purchase"


def first_divergence(
    teacher: Sequence[tuple[str, dict[str, str]]],
    model: Sequence[tuple[str, dict[str, str]]],
) -> int | None:
    """第一次不一致的回合下标；完全一致返回 None。"""
    for index in range(min(len(teacher), len(model))):
        if teacher[index] != model[index]:
            return index
    if len(teacher) == len(model):
        return None
    return min(len(teacher), len(model))


def classify(
    teacher: Sequence[tuple[str, dict[str, str]]],
    model: Sequence[tuple[str, dict[str, str]]],
    index: int,
) -> str:
    if index >= len(model):
        return "提前结束"
    if index >= len(teacher):
        return "多走回合"
    t_name, t_args = teacher[index]
    m_name, m_args = model[index]
    if t_name != m_name:
        return f"换工具（{t_name}→{m_name}）"
    changed = {key for key in set(t_args) | set(m_args) if t_args.get(key) != m_args.get(key)}
    if not changed:
        return "参数相同"
    if changed <= QUERY_KEYS:
        return "换查询"
    if changed <= PRODUCT_KEYS:
        return "换商品"
    if changed <= OPTION_KEYS:
        return "换规格"
    return "换" + "/".join(sorted(changed))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="自测评测目录")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--show", type=int, default=3, help="打印几个失败样例")
    args = parser.parse_args(argv)

    pool: dict[int, list[tuple[str, dict[str, str]]]] = {}
    for row in read_jsonl(args.pool):
        pool[int(row["task_id"])] = teacher_actions(row)
    if not pool:
        raise SystemExit(f"训练池为空或读不出动作：{args.pool}")

    records = read_jsonl(Path(args.run) / "trajectories.jsonl")
    if not records:
        raise SystemExit(f"轨迹为空：{args.run}")

    matched_tasks = 0
    missing = 0
    successes = 0
    failures = 0
    fail_positions: Counter = Counter()
    fail_kinds: Counter = Counter()
    ok_positions: Counter = Counter()
    step_ratios: list[float] = []
    examples: list[str] = []

    for record in records:
        task_id = int(record["task_id"])
        teacher = pool.get(task_id)
        if not teacher:
            missing += 1
            continue
        matched_tasks += 1
        success = is_success(record)
        if success:
            successes += 1
        else:
            failures += 1
        model = model_actions(record)
        if model:
            step_ratios.append(len(model) / max(len(teacher), 1))
        index = first_divergence(teacher, model)
        if index is None:
            ok_positions["完全一致"] += 1
            continue
        position = f"第{index + 1}步"
        if success:
            ok_positions[position] += 1
            continue
        fail_positions[position] += 1
        kind = classify(teacher, model, index)
        fail_kinds[kind] += 1
        if len(examples) < args.show:
            model_name = model[index][0] if index < len(model) else "（无）"
            model_args = model[index][1] if index < len(model) else ""
            examples.append(
                f"task {task_id}：第 {index + 1} 步 {kind}；"
                f"老师 {teacher[index][0]}{teacher[index][1]} → 模型 {model_name}{model_args}"
            )

    if not matched_tasks:
        raise SystemExit("没有一道题能在训练池里找到对应轨迹：--pool 指错了？")

    print("# 模仿对齐：模型在训练过的题上，动作和老师差在哪一步")
    print()
    print(f"- 能对齐的题 {matched_tasks} 道（池里找不到的 {missing} 道）")
    print(f"- 做对 {successes} 道，做错 {failures} 道")
    if step_ratios:
        step_ratios.sort()
        print(
            f"- 模型步数 / 老师步数：中位 {step_ratios[len(step_ratios) // 2]:.2f}"
            f"（<1 表示提前收尾，>1 表示绕路）"
        )
    print()

    print("## 1. 第一次分歧发生在第几步")
    print()
    print("| 位置 | 做错的题 | 做对的题 |")
    print("|---|---:|---:|")
    for position in sorted(
        set(fail_positions) | set(ok_positions), key=lambda k: (k != "完全一致", k)
    ):
        print(f"| {position} | {fail_positions.get(position, 0)} | {ok_positions.get(position, 0)} |")
    print()

    print("## 2. 做错的题，分歧是什么类型")
    print()
    print("| 类型 | 道数 | 占失败 |")
    print("|---|---:|---:|")
    for kind, count in fail_kinds.most_common():
        print(f"| {kind} | {count} | {count / failures:.0%} |")
    if not fail_kinds:
        print("| — | 0 | — |")
    print()

    if examples:
        print("## 3. 样例")
        print()
        for line in examples:
            print(f"- {line}")
        print()

    early = sum(
        count for position, count in fail_positions.items()
        if position.startswith("第") and int(position[1:-1]) <= EARLY_STEPS
    )
    print("## 4. 判定")
    print()
    if not failures:
        print("没有失败题，无法判定（这次跑的题全都做对了）。")
    elif early / failures >= 0.5:
        print(
            f"**照做都做不到**：{early}/{failures}（{early / failures:.0%}）在头 "
            f"{EARLY_STEPS} 步就和老师分叉。⇒ 是模仿/容量问题，"
            "加 epoch 或加数据**可能**有用（但也可能只是背下答案）。"
        )
    else:
        print(
            f"**流程会走，最后挑错**：只有 {early}/{failures}"
            f"（{early / failures:.0%}）在头 {EARLY_STEPS} 步分叉。⇒ 是判别问题，"
            "加 epoch 只会让它背下这些题的答案（自测涨、dev 不涨），**加量无用**。"
        )
    print()
    print("> 只在第一次分歧前比较：分叉之后两边观测已不同，再对齐没有意义。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
