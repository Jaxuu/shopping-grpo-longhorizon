#!/usr/bin/env python3
"""解剖「买对了商品却没拿满分」的失败：卡在哪个维度、要求值当时在不在页面上。

为什么单独做这件事：2026-10-19 之前我们统计过「失分维度是 key_options」，
但那个统计是**循环论证**——被统计的行买的是**别的商品**，正确商品的规格轴根本
不在页面上，模型没有机会选对。那个结论已撤回。

本脚本把口径收紧到**只统计买对了正确商品的行**（`purchase.asin == goal.asin`），
此时正确商品的规格轴必然在页面上，才谈得上「模型有机会却没选对」。

同时必须回答第二个问题：**要求值当时到底在不在模型看到的页面上**。
不做这一步就又会犯同一个错——把「没机会」算成「没能力」。
三种情形分开计：
  verbatim  —— 原文逐字可见
  normalized—— 仅大小写 / 标点 / 反斜杠转义不同（页面上实际有，只是写法不同）
  absent    —— 真的不在页面上（模型没有机会选对，不算模型的问题）
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

SUCCESS_TYPES = frozenset({"gold_purchase", "valid_alternative_purchase"})
# 只关心「因为偏好维度没满分」的失败；这些不是偏好维度的失败，单独归类。
NON_PREFERENCE_TYPES = frozenset(
    {"repeat_loop", "max_steps", "early_abstain", "graceful_abstain"}
)


def _normalize(text):
    """把大小写、斜杠、竖线、空格、全角括号抹平后再比较。"""
    value = str(text).lower()
    value = re.sub(r"[\\/|\s]+", "", value)
    for src, dst in (("（", "("), ("）", ")"), ("：", ":"), ("，", ",")):
        value = value.replace(src, dst)
    return value


def _visibility(required, observation):
    if not observation:
        return "absent"
    if required in observation:
        return "verbatim"
    if _normalize(required) in _normalize(observation):
        return "normalized"
    return "absent"


def _gold_page_observation(trajectory, gold_asin):
    """取正确商品最后一次 product_detail 观测（模型实际看到的那份文本）。"""
    observation = ""
    for step in trajectory.get("steps") or []:
        text = step.get("observation") or ""
        if "page_type: product_detail" in text and gold_asin in text:
            observation = text
    return observation


def diagnose(trajectories):
    """返回逐题诊断结果列表。"""
    items = []
    for trajectory in trajectories:
        terminal = trajectory.get("terminal_result") or {}
        detail = terminal.get("reward_detail") or {}
        reward_type = str(detail.get("reward_type") or "")
        if reward_type in SUCCESS_TYPES:
            continue
        goal = terminal.get("goal") or {}
        gold_asin = str(goal.get("asin") or "")
        purchase = terminal.get("purchase") or {}
        bought_asin = str(purchase.get("asin") or "")
        if not gold_asin or bought_asin != gold_asin:
            continue
        preference = (detail.get("evidence") or {}).get("preference_scoring") or {}
        dimensions = preference.get("dimensions") or {}
        failing = sorted(
            name
            for name, payload in dimensions.items()
            if payload.get("active") and float(payload.get("score") or 0.0) < 1.0
        )
        observation = _gold_page_observation(trajectory, gold_asin)
        option_checks = []
        for name in failing:
            if name != "key_options":
                continue
            for item in (dimensions[name].get("results") or []):
                if item.get("passed"):
                    continue
                required = item.get("required") or {}
                wrong = (item.get("evidence") or {}).get("wrong_values") or []
                for axis, meta in required.items():
                    value = str(meta.get("value") or "")
                    option_checks.append(
                        {
                            "axis": axis,
                            "required": value,
                            "selected": "、".join(
                                str(w.get("selected") or "")
                                for w in wrong
                                if str(w.get("axis") or "") == str(axis)
                            ),
                            "visibility": _visibility(value, observation),
                        }
                    )
        items.append(
            {
                "task_id": trajectory.get("task_id"),
                "reward_type": reward_type,
                "failing_dimensions": failing,
                "option_checks": option_checks,
            }
        )
    return items


def summarize(items):
    dimension_counts = Counter()
    visibility = Counter()
    for item in items:
        for name in item["failing_dimensions"]:
            dimension_counts[name] += 1
        for check in item["option_checks"]:
            visibility[check["visibility"]] += 1
    return {
        "tasks": len(items),
        "failing_dimension_counts": dict(dimension_counts),
        "option_visibility_counts": dict(visibility),
        "only_key_options": sum(
            1 for item in items if item["failing_dimensions"] == ["key_options"]
        ),
    }


def render(items, summary):
    lines = [
        "# 「买对了商品却没满分」的失败解剖",
        "",
        f"- 这类失败共 **{summary['tasks']}** 条"
        "（口径：`purchase.asin == goal.asin` 且未达到严格成功）",
        f"- 其中**只**卡在 `key_options` 的：**{summary['only_key_options']}** 条",
        "",
        "## 1. 卡在哪个维度",
        "",
        "| 维度 | 条数 |",
        "|---|---:|",
    ]
    for name, count in sorted(
        summary["failing_dimension_counts"].items(), key=lambda kv: -kv[1]
    ):
        lines.append(f"| `{name}` | {count} |")
    lines += [
        "",
        "## 2. 要求值当时在不在模型看到的页面上",
        "",
        "| 情形 | 条数 | 含义 |",
        "|---|---:|---|",
    ]
    meanings = {
        "verbatim": "原文逐字可见，模型看得到却没选",
        "normalized": "仅大小写/标点/转义不同，页面上实际有",
        "absent": "真的不在页面上，模型没有机会选对",
    }
    for name in ("verbatim", "normalized", "absent"):
        lines.append(
            f"| {name} | {summary['option_visibility_counts'].get(name, 0)} "
            f"| {meanings[name]} |"
        )
    reachable = summary["option_visibility_counts"].get(
        "verbatim", 0
    ) + summary["option_visibility_counts"].get("normalized", 0)
    total = sum(summary["option_visibility_counts"].values())
    if total:
        lines += [
            "",
            f"⇒ **{reachable}/{total}** 条的要求值就在页面上（含写法差异），"
            "模型是「挑了一个相近的变体」而不是「找不到」。",
        ]
    lines += ["", "## 3. 逐条明细", ""]
    for item in items:
        dims = ", ".join(item["failing_dimensions"]) or "（无活跃维度不满）"
        lines.append(f"### task {item['task_id']} — {item['reward_type']} — {dims}")
        if not item["option_checks"]:
            lines.append("")
            lines.append("- 无 `key_options` 明细")
            lines.append("")
            continue
        for check in item["option_checks"]:
            lines.append(
                f"- 轴 `{check['axis']}`：要求 `{check['required']}`"
                f" / 实选 `{check['selected']}` / 可见性 **{check['visibility']}**"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="写 markdown 报告")
    parser.add_argument("--json", action="store_true", help="输出 JSON 摘要")
    args = parser.parse_args(argv)

    with args.trajectories.open(encoding="utf-8") as handle:
        trajectories = [json.loads(line) for line in handle if line.strip()]
    items = diagnose(trajectories)
    summary = summarize(items)
    if args.json:
        print(json.dumps({"summary": summary, "items": items}, ensure_ascii=False, indent=2))
        return 0
    report = render(items, summary)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"已写入 {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
