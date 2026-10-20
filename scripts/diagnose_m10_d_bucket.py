#!/usr/bin/env python3
"""M10 D 桶诊断：规格选错是「数据不够」还是「模型分辨不了」。

为什么需要这一步
----------------
D 桶 18 题 = 「商品买对了，规格没选对」，是 76 条失败里的第三大块。
补数据之前必须先分清两种情况，否则白花钱：

* **数据不够** ⇒ 池子里缺「选项很多、长得很像」的硬例 ⇒ 从 6,501 条供给里补，有效。
* **分辨不了** ⇒ 池子里硬例已经够多，是 2B 模型抄不准长字符串 ⇒ 补数据无效，得换思路。

判定方法（两条独立证据）
------------------------
1. **评测侧**：把 18 条失败逐条分类（没选 / 漏轴 / 选第 1 个 / 选相邻 / 其它）。
2. **训练侧**：池子里 1,694 次 `select_option` 的**选项个数分布**与**被选中的位置分布**。

   * 如果池子里「轴内 ≥4 个选项」的硬例本来就很多，且位置分布接近均匀
     ⇒ 模型见过足够多的硬例，也见过各个位置，**不是数据问题**。
   * 如果池子里正确选项几乎总在第 0 位 ⇒ 模型会学成「选第一个」，
     那就和「开第一个」是同一个病，靠重配比能治。

用法::

    .venv/Scripts/python.exe scripts/diagnose_m10_d_bucket.py \\
        --trajectories outputs/evaluation/sft-b-curriculum/trajectories.jsonl \\
        --pool data/sft_pure_v4/all.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

AVAILABLE_OPTIONS = re.compile(r"^available_options: (?P<payload>.+)$", re.MULTILINE)
NORMALIZE_TABLE = str.maketrans({"｜": "/", "|": "/", "\\": "/", "\xa0": " ", "／": "/"})

#: 分类标签，顺序即优先级。
CATEGORIES = (
    "no_selection",
    "required_not_offered",
    "missing_axis",
    "overwrote",
    "picked_first",
    "adjacent",
    "other",
)
CATEGORY_LABELS = {
    "no_selection": "一次规格都没选",
    "required_not_offered": "要求的规格值不在选项里（环境/口径问题）",
    "missing_axis": "只选了部分轴，漏掉了必须选的轴",
    "overwrote": "选对过，又被另一次选择覆盖掉",
    "picked_first": "选了第 1 个，正确答案在后面",
    "adjacent": "选了相邻的那一个（差 1 位）",
    "other": "选了别的（不是第 1 个也不是相邻）",
}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def norm(value: str) -> str:
    """把选项字符串归一：**删掉所有空白**、统一斜杠、忽略大小写。

    环境投影会做三件事，不归一会大量误报「要求的规格值不在选项里」：
    `A559` → `a559`（压平大小写）、`1/2` → `1 | 2`（斜杠改写成竖线）、
    `\\xa0 ` → 两个空格（不间断空格）。注意必须**删掉**空白而不是压成单空格，
    否则 `1 | 2` 会留下 `1 / 2`，和 `1/2` 仍然对不上。
    """
    return "".join(str(value).translate(NORMALIZE_TABLE).split()).casefold()


def observation_text(step: dict) -> str:
    raw = step.get("raw_observation")
    if isinstance(raw, str) and raw:
        return raw
    return str(step.get("observation") or "")


def available_options(observation: str) -> dict[str, list[str]]:
    """从商品详情页观测里取 `available_options`；取不到返回空字典。"""
    match = AVAILABLE_OPTIONS.search(observation)
    if not match:
        return {}
    try:
        payload = json.loads(match.group("payload"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(axis): [str(v) for v in values]
        for axis, values in payload.items()
        if isinstance(values, list)
    }


def last_available_options(steps: list[dict]) -> dict[str, list[str]]:
    """轨迹里最后一次出现的选项表（后出现的更接近购买时的状态）。"""
    found: dict[str, list[str]] = {}
    for step in steps:
        parsed = available_options(observation_text(step))
        if parsed:
            found = parsed
    return found


def select_values(steps: list[dict]) -> list[str]:
    return [
        str((step.get("parameters") or {}).get("value"))
        for step in steps
        if step.get("tool_name") == "select_option"
        and (step.get("parameters") or {}).get("value")
    ]


def _axis_of(available_norm: dict[str, list[str]], value: str) -> str | None:
    for axis, values in available_norm.items():
        if value in values:
            return axis
    return None


def classify(required: list[str], selected: list[str], available: dict) -> dict:
    """把一条规格失败归类；返回分类、位置信息与明细。"""
    available_norm = {
        axis: [norm(v) for v in values] for axis, values in available.items()
    }
    required_norm = [(v, norm(v)) for v in required]
    detail: dict = {"required": required, "selected": selected, "positions": []}

    if not selected:
        return {"category": "no_selection", **detail}

    required_axis = {}
    for original, normalized in required_norm:
        axis = _axis_of(available_norm, normalized)
        if axis is None:
            return {"category": "required_not_offered", **detail}
        required_axis[original] = axis

    final: dict[str, str] = {}
    seen: set[str] = set()
    for value in selected:
        normalized = norm(value)
        axis = _axis_of(available_norm, normalized)
        if axis is None:
            continue
        final[axis] = normalized
        seen.add(normalized)

    if all(
        final.get(required_axis[original]) == normalized
        for original, normalized in required_norm
    ):
        return {"category": "selected_correct", **detail}

    if all(normalized in seen for _, normalized in required_norm):
        return {"category": "overwrote", **detail}

    missing = [
        original
        for original, normalized in required_norm
        if final.get(required_axis[original]) is None
    ]
    if missing:
        return {"category": "missing_axis", **detail}

    for original, normalized in required_norm:
        values = available_norm[required_axis[original]]
        required_index = values.index(normalized)
        chosen = final[required_axis[original]]
        chosen_index = values.index(chosen) if chosen in values else None
        detail["positions"].append(
            {
                "axis": required_axis[original],
                "required_index": required_index,
                "chosen_index": chosen_index,
                "options": len(values),
            }
        )

    if any(p["chosen_index"] == 0 and p["required_index"] > 0 for p in detail["positions"]):
        return {"category": "picked_first", **detail}
    if any(
        p["chosen_index"] is not None and abs(p["chosen_index"] - p["required_index"]) == 1
        for p in detail["positions"]
    ):
        return {"category": "adjacent", **detail}
    return {"category": "other", **detail}


def diagnose_trajectory(trajectory: dict) -> dict:
    goal = (trajectory.get("terminal_result") or {}).get("goal") or {}
    required = [str(v) for v in goal.get("goal_options") or []]
    steps = trajectory.get("steps") or []
    result = classify(required, select_values(steps), last_available_options(steps))
    return {"task_id": trajectory.get("task_id"), **result}


def is_grade_failure(trajectory: dict) -> bool:
    """只挑「买对商品、只差规格」的失败：`target_asin_match` 为真但没拿到 gold。"""
    detail = (trajectory.get("terminal_result") or {}).get("reward_detail") or {}
    return bool(detail.get("target_asin_match")) and detail.get("reward_type") != "gold_purchase"


def pool_selection_stats(rows: list[dict]) -> dict:
    """训练池里每一次 `select_option` 的难度与位置。"""
    sizes: collections.Counter = collections.Counter()
    positions: collections.Counter = collections.Counter()
    hard_positions: collections.Counter = collections.Counter()
    unmatched = 0
    for row in rows:
        current: dict[str, list[str]] = {}
        for message in row.get("messages") or []:
            if message.get("role") == "tool":
                parsed = available_options(str(message.get("content") or ""))
                if parsed:
                    current = parsed
                continue
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                if function.get("name") != "select_option":
                    continue
                try:
                    value = json.loads(function.get("arguments") or "{}").get("value")
                except (TypeError, json.JSONDecodeError):
                    continue
                if value is None:
                    continue
                normalized = norm(value)
                for values in current.values():
                    normalized_values = [norm(v) for v in values]
                    if normalized in normalized_values:
                        index = normalized_values.index(normalized)
                        sizes[len(values)] += 1
                        positions[index] += 1
                        if len(values) >= 4:
                            hard_positions[index] += 1
                        break
                else:
                    unmatched += 1
    return {
        "selections": sum(sizes.values()) + unmatched,
        "matched": sum(sizes.values()),
        "unmatched": unmatched,
        "axis_size": dict(sorted(sizes.items())),
        "trivial_single_option": sizes.get(1, 0),
        "hard_ge4": sum(count for size, count in sizes.items() if size >= 4),
        "position": dict(sorted(positions.items())),
        "hard_position": dict(sorted(hard_positions.items())),
        "share_at_index0": (positions.get(0, 0) / sum(positions.values())) if positions else 0.0,
        "hard_share_at_index0": (
            hard_positions.get(0, 0) / sum(hard_positions.values()) if hard_positions else 0.0
        ),
    }


def verdict(categories: collections.Counter, pool: dict, *, hard_threshold: int = 300) -> str:
    """给出「补数据有用 / 没用」的判定。"""
    hard = pool["hard_ge4"]
    share0 = pool["hard_share_at_index0"]
    if hard < hard_threshold:
        return "数据不够：池子里硬例不足，补数据有效"
    if share0 > 0.5:
        return "位置偏置：正确选项大多在第 0 位，重配比有效（和「开第一个」同一个病）"
    return "分辨力问题：硬例充足且位置接近均匀，补数据大概率无效"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=ROOT / "outputs/evaluation/sft-b-curriculum/trajectories.jsonl",
    )
    parser.add_argument("--pool", type=Path, default=ROOT / "data/sft_pure_v4/all.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trajectories = read_jsonl(args.trajectories)
    failures = [t for t in trajectories if is_grade_failure(t)]
    diagnosed = [diagnose_trajectory(t) for t in failures]
    categories = collections.Counter(item["category"] for item in diagnosed)

    pool = pool_selection_stats(read_jsonl(args.pool))

    print(f"评测轨迹 {len(trajectories)} 条；「买对商品、只差规格」的失败 {len(failures)} 条")
    print("  失败分类：")
    for name in CATEGORIES:
        count = categories.get(name, 0)
        if count:
            print(f"    {name:<22} {count:>3}  {CATEGORY_LABELS[name]}")
    print()
    for item in diagnosed:
        if item["category"] == "other":
            print(f"    [other] task {item['task_id']} 需 {item['required']} 选 {item['selected']}")
    print()
    print(f"训练池 {pool['selections']} 次 select_option：")
    print(f"  轴内只有 1 个选项（白给）：{pool['trivial_single_option']}")
    print(f"  轴内 ≥4 个选项（硬例）：{pool['hard_ge4']}")
    print(f"  被选中位置在第 0 位：全部 {pool['share_at_index0']:.1%}；仅硬例 {pool['hard_share_at_index0']:.1%}")
    print(f"  硬例位置分布（前 6）：{dict(list(pool['hard_position'].items())[:6])}")
    print()
    print("判定：" + verdict(categories, pool))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
