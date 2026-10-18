#!/usr/bin/env python3
"""Final-200 失败解剖：把「62% 天花板」拆成**有区分度的**过程缺陷与零梯度质量。

为什么要有这个
--------------
终局奖励是离散查表：`gold 1.0 / valid_alt 0.55 / partial ≤0.25 / repeat_loop −0.65 /
max_steps −0.50 / wrong −0.85`。同一条失败轨迹内部，无论走了 3 步还是 20 步，
奖励都是同一个常数 ⇒ 梯度恒为 0。要加过程奖励，必须先回答三个问题：

1. **过程形状有区分度吗？** 步数 / 搜索次数 / 打开商品数 / 动作循环——
   如果成功与失败的分布**一样**，那么"奖励少走弯路"就是伪靶点。
2. **失败的质量缺陷在哪？** 按 `evidence.preference_scoring` 的**激活维度**
   （brand / model / core_functions / key_options）看，失败卡在"没查到"还是"选错"。
3. **零梯度质量有多大？** 同一 `reward_type` 内奖励是否为常数——
   是则整块没有排序信息，shaping 才有空间。

⚠️ **缺陷必须"有区分度"才算可及空间**：若某缺陷在成功题里同样普遍
（本仓库实测 `opens < 2` 覆盖 99% 的题，连 gold 也一样），它不是缺陷，是行为常态，
拿它算"可及空间"会把 100% 的失败都算成可救 —— 这是自欺。

用法::

    .venv/bin/python scripts/analyze_eval_failures.py outputs/evaluation/m6-step50

    # 落盘 + 机器可读
    .venv/bin/python scripts/analyze_eval_failures.py outputs/evaluation/m6-step50 \
        --out outputs/evaluation/m6-step50/failure_anatomy.md --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

# 过程缺陷阈值（写死并写进报告，避免口径漂移）。
REPEAT_RUN_THRESHOLD = 3  # 连续 ≥3 次相同动作 ⇒ 视为循环
COMPARISON_OPEN_THRESHOLD = 2  # 打开过 ≥2 个不同商品才算"做过比较"
LOW_EVIDENCE_THRESHOLD = 0.5  # evidence_coverage 低于此值视为证据不足
SUCCESS_REWARD_THRESHOLD = 0.5  # 终局奖励 ≥0.5 视为成功（gold 1.0 / valid_alt 0.55）

# 区分度判据：失败率 - 成功率 ≥ 0.15 且 提升比 ≥ 1.5 ⇒ 该缺陷才有区分度。
DISCRIMINATION_GAP = 0.15
DISCRIMINATION_LIFT = 1.5

PROCESS_DEFECTS = ("action_loop", "no_comparison", "low_evidence")


def _detail(row: dict[str, Any]) -> dict[str, Any]:
    terminal = row.get("terminal_result") or {}
    detail = terminal.get("reward_detail")
    if isinstance(detail, dict) and detail:
        return detail
    return terminal if isinstance(terminal, dict) else {}


def _reward_type(row: dict[str, Any]) -> str:
    detail = _detail(row)
    return str(
        detail.get("reward_type")
        or detail.get("termination_reason")
        or (row.get("terminal_result") or {}).get("termination_reason")
        or row.get("status")
        or "unknown"
    )


def _action_signature(step: dict[str, Any]) -> str:
    name = str(step.get("tool_name") or "")
    params = step.get("parameters")
    if isinstance(params, dict):
        try:
            payload = json.dumps(params, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            payload = str(params)
    else:
        payload = str(params)
    return f"{name}|{payload}"


def max_run(sequence: Sequence[str]) -> int:
    """最长连续相同元素长度；空序列返回 0。"""
    best = 0
    current = 0
    previous: str | None = None
    for item in sequence:
        if item == previous:
            current += 1
        else:
            current = 1
            previous = item
        best = max(best, current)
    return best


def active_dimensions(detail: dict[str, Any]) -> dict[str, float]:
    """只取 `evidence.preference_scoring` 里 **active=True** 的维度分数。

    未激活的维度分数恒为 0（如任务不要求 brand 时 `brand: 0.0`），
    直接平均会把"没要求"误读成"没做到"。
    """
    evidence = detail.get("evidence")
    if not isinstance(evidence, dict):
        return {}
    scoring = evidence.get("preference_scoring")
    if not isinstance(scoring, dict):
        return {}
    dimensions = scoring.get("dimensions")
    if not isinstance(dimensions, dict):
        return {}
    result: dict[str, float] = {}
    for name, payload in dimensions.items():
        if not isinstance(payload, dict) or not payload.get("active"):
            continue
        score = payload.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            result[str(name)] = float(score)
    return result


def fingerprint(row: dict[str, Any]) -> dict[str, Any]:
    """把一条评测轨迹压成过程指纹。"""
    steps = row.get("steps") or []
    names = [str(step.get("tool_name") or "") for step in steps]
    signatures = [_action_signature(step) for step in steps]
    opened = {
        str((step.get("parameters") or {}).get("asin"))
        for step in steps
        if step.get("tool_name") == "open_product" and (step.get("parameters") or {}).get("asin")
    }
    detail = _detail(row)
    coverage = detail.get("evidence_coverage")
    coverage_value = (
        float(coverage) if isinstance(coverage, (int, float)) and not isinstance(coverage, bool) else None
    )
    reward = row.get("final_reward")
    reward_value = (
        float(reward) if isinstance(reward, (int, float)) and not isinstance(reward, bool) else None
    )
    return {
        "task_id": row.get("task_id"),
        "reward_type": _reward_type(row),
        "final_reward": reward_value,
        "steps": len(steps),
        "max_tool_run": max_run(names),
        "max_action_run": max_run(signatures),
        "distinct_opens": len(opened),
        "searches": sum(1 for name in names if name == "search_products"),
        "purchases": sum(1 for name in names if name == "purchase"),
        "evidence_coverage": coverage_value,
        "active_dimensions": active_dimensions(detail),
    }


def classify(fp: dict[str, Any]) -> list[str]:
    """给一条轨迹打上过程缺陷标签（可多个）。"""
    defects = []
    if fp["max_action_run"] >= REPEAT_RUN_THRESHOLD:
        defects.append("action_loop")
    if fp["distinct_opens"] < COMPARISON_OPEN_THRESHOLD:
        defects.append("no_comparison")
    if fp["evidence_coverage"] is not None and fp["evidence_coverage"] < LOW_EVIDENCE_THRESHOLD:
        defects.append("low_evidence")
    return defects


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _median(values: Sequence[float]) -> float:
    """中位数（偶数取中间两数均值）；空序列返回 nan。"""
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def discrimination(failures: Sequence[dict[str, Any]], successes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """缺陷在失败组 vs 成功组的出现率 + 提升比。

    只有「失败组显著更高」的缺陷才配叫缺陷；两边都 100% 的（本仓库的
    `no_comparison`）是行为常态，算可及空间会把全部失败都误算成可救。
    """
    rows = []
    for defect in PROCESS_DEFECTS:
        fail_rate = _rate(sum(1 for fp in failures if defect in fp["defects"]), len(failures))
        succ_rate = _rate(sum(1 for fp in successes if defect in fp["defects"]), len(successes))
        lift = (fail_rate / succ_rate) if succ_rate > 0 else (math.inf if fail_rate > 0 else 1.0)
        discriminating = (fail_rate - succ_rate) >= DISCRIMINATION_GAP and lift >= DISCRIMINATION_LIFT
        rows.append(
            {
                "defect": defect,
                "failure_rate": fail_rate,
                "success_rate": succ_rate,
                "lift": lift,
                "discriminating": discriminating,
            }
        )
    return {"rows": rows}


def flat_reward_types(prints: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """每个 `reward_type` 内部的奖励是否恒定（恒定 ⇒ 该块没有排序信息）。"""
    buckets: dict[str, list[float]] = defaultdict(list)
    for fp in prints:
        if fp["final_reward"] is not None:
            buckets[fp["reward_type"]].append(fp["final_reward"])
    rows = []
    for name, rewards in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        low, high = min(rewards), max(rewards)
        rows.append(
            {
                "reward_type": name,
                "tasks": len(rewards),
                "min": low,
                "max": high,
                "flat": (high - low) <= 1e-9,
                "mean": _mean(rewards),
            }
        )
    return rows


def analyze(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """核心分析：结局分布 × 过程区分度 × 激活维度 × 零梯度质量。"""
    if not rows:
        raise ValueError("评测轨迹为空")
    prints = [fingerprint(row) for row in rows]
    for fp in prints:
        fp["defects"] = classify(fp)
        fp["is_success"] = (fp["final_reward"] or 0.0) >= SUCCESS_REWARD_THRESHOLD

    successes = [fp for fp in prints if fp["is_success"]]
    failures = [fp for fp in prints if not fp["is_success"]]

    by_type: dict[str, dict[str, Any]] = {}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fp in prints:
        buckets[fp["reward_type"]].append(fp)
    for name, items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        rewards = [fp["final_reward"] for fp in items if fp["final_reward"] is not None]
        by_type[name] = {
            "tasks": len(items),
            "share": len(items) / len(prints),
            "mean_reward": _mean(rewards),
            "mean_steps": _mean([fp["steps"] for fp in items]),
            "mean_searches": _mean([fp["searches"] for fp in items]),
            "mean_opens": _mean([fp["distinct_opens"] for fp in items]),
            "action_loop_rate": _rate(sum(1 for fp in items if "action_loop" in fp["defects"]), len(items)),
        }

    shape = {
        "success": {
            "tasks": len(successes),
            "mean_steps": _mean([fp["steps"] for fp in successes]),
            "median_steps": _median([fp["steps"] for fp in successes]),
            "mean_searches": _mean([fp["searches"] for fp in successes]),
            "mean_opens": _mean([fp["distinct_opens"] for fp in successes]),
            "mean_max_run": _mean([fp["max_action_run"] for fp in successes]),
            "long_share": _rate(sum(1 for fp in successes if fp["steps"] > 12), len(successes)),
            "step_histogram": dict(sorted(Counter(fp["steps"] for fp in successes).items())),
        },
        "failure": {
            "tasks": len(failures),
            "mean_steps": _mean([fp["steps"] for fp in failures]),
            "median_steps": _median([fp["steps"] for fp in failures]),
            "mean_searches": _mean([fp["searches"] for fp in failures]),
            "mean_opens": _mean([fp["distinct_opens"] for fp in failures]),
            "mean_max_run": _mean([fp["max_action_run"] for fp in failures]),
            "long_share": _rate(sum(1 for fp in failures if fp["steps"] > 12), len(failures)),
            "step_histogram": dict(sorted(Counter(fp["steps"] for fp in failures).items())),
        },
    }

    disc = discrimination(failures, successes)

    dimension_rows = []
    dimension_names = sorted({name for fp in prints for name in fp["active_dimensions"]})
    for name in dimension_names:
        values = [fp["active_dimensions"][name] for fp in prints if name in fp["active_dimensions"]]
        fail_values = [fp["active_dimensions"][name] for fp in failures if name in fp["active_dimensions"]]
        succ_values = [fp["active_dimensions"][name] for fp in successes if name in fp["active_dimensions"]]
        dimension_rows.append(
            {
                "dimension": name,
                "active_tasks": len(values),
                "mean_score": _mean(values),
                "failure_mean_score": _mean(fail_values),
                "success_mean_score": _mean(succ_values),
                "failure_zero_rate": _rate(sum(1 for value in fail_values if value <= 0.0), len(fail_values)),
            }
        )

    flat_rows = flat_reward_types(prints)
    zero_gradient_failures = [
        fp for fp in failures if any(row["reward_type"] == fp["reward_type"] and row["flat"] for row in flat_rows)
    ]

    discriminating = [row["defect"] for row in disc["rows"] if row["discriminating"]]
    addressable = [
        fp for fp in failures if any(defect in fp["defects"] for defect in discriminating)
    ]

    return {
        "tasks": len(prints),
        "successes": len(successes),
        "success_rate": len(successes) / len(prints),
        "mean_steps": _mean([fp["steps"] for fp in prints]),
        "by_reward_type": by_type,
        "shape": shape,
        "defects": disc["rows"],
        "dimensions": dimension_rows,
        "flat_reward_types": flat_rows,
        "addressable": {
            "failures": len(failures),
            "discriminating_defects": discriminating,
            "with_discriminating_defect": len(addressable),
            "share_of_failures": _rate(len(addressable), len(failures)),
            "zero_gradient_failures": len(zero_gradient_failures),
            "zero_gradient_share_of_failures": _rate(len(zero_gradient_failures), len(failures)),
        },
        "thresholds": {
            "repeat_run": REPEAT_RUN_THRESHOLD,
            "comparison_opens": COMPARISON_OPEN_THRESHOLD,
            "low_evidence": LOW_EVIDENCE_THRESHOLD,
            "success_reward": SUCCESS_REWARD_THRESHOLD,
            "discrimination_gap": DISCRIMINATION_GAP,
            "discrimination_lift": DISCRIMINATION_LIFT,
        },
    }


def _shape_verdict(summary: dict[str, Any]) -> str:
    """过程形状是否有区分度（一句话）。

    用**中位数 + 长尾占比**判定，不用均值：均值会被个别 20/35 步的离群轨迹带偏
    （本仓库实测失败组均值 8.74 vs 成功 8.21，差 0.53 全部来自 3 条离群轨迹）。
    """
    success = summary["shape"]["success"]
    failure = summary["shape"]["failure"]
    if not success["tasks"] or not failure["tasks"]:
        return "样本不足，无法判断"
    median_gap = abs(failure["median_steps"] - success["median_steps"])
    long_gap = abs(failure["long_share"] - success["long_share"])
    if median_gap < 1.0 and long_gap < 0.05:
        return (
            "**过程形状没有区分度**：成功与失败的**步数中位数相同**"
            f"（{success['median_steps']:.0f} vs {failure['median_steps']:.0f}）、"
            f"长尾（>12 步）占比接近（{success['long_share']:.1%} vs {failure['long_share']:.1%}）⇒ "
            "「奖励少走弯路」不是本任务的靶点，失败出在**选品质量**而不是**探索不足**。"
            "⚠️ 均值差看似存在，但由个别离群轨迹造成，别据此下结论。"
        )
    return (
        f"过程形状有区分度：步数中位数 {success['median_steps']:.0f} vs "
        f"{failure['median_steps']:.0f}、长尾占比 {success['long_share']:.1%} vs "
        f"{failure['long_share']:.1%} ⇒ turn-level 过程奖励有明确靶点。"
    )


def render(summary: dict[str, Any]) -> str:
    """渲染 markdown 报告。"""
    out: list[str] = []
    out.append("# Final-200 失败解剖（过程奖励可及空间）")
    out.append("")
    out.append(
        f"> 样本 {summary['tasks']} 题，成功率 **{summary['success_rate']:.1%}**"
        f"（{summary['successes']}/{summary['tasks']}），平均 {summary['mean_steps']:.2f} 步。"
    )
    out.append("")

    out.append("## 1. 结局分布")
    out.append("")
    out.append("| reward_type | 题数 | 占比 | 平均奖励 | 平均步数 | 平均搜索 | 平均打开 | 动作循环率 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for name, row in summary["by_reward_type"].items():
        out.append(
            f"| `{name}` | {row['tasks']} | {row['share']:.1%} | {row['mean_reward']:.3f} | "
            f"{row['mean_steps']:.2f} | {row['mean_searches']:.2f} | {row['mean_opens']:.2f} | "
            f"{row['action_loop_rate']:.0%} |"
        )
    out.append("")

    out.append("## 2. 过程形状：成功 vs 失败")
    out.append("")
    out.append("| 组 | 题数 | 平均步数 | 步数中位 | 平均搜索 | 平均打开 | 最长同动作 | 长尾(>12步) | 步数分布 |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for key, label in (("success", "成功"), ("failure", "失败")):
        row = summary["shape"][key]
        histogram = ", ".join(f"{steps}步×{count}" for steps, count in row["step_histogram"].items())
        out.append(
            f"| {label} | {row['tasks']} | {row['mean_steps']:.2f} | {row['median_steps']:.0f} | "
            f"{row['mean_searches']:.2f} | {row['mean_opens']:.2f} | {row['mean_max_run']:.2f} | "
            f"{row['long_share']:.1%} | {histogram} |"
        )
    out.append("")
    out.append(_shape_verdict(summary))
    out.append("")

    out.append("## 3. 缺陷的区分度（失败率 vs 成功率）")
    out.append("")
    out.append("| 缺陷 | 失败组出现率 | 成功组出现率 | 提升比 | 有区分度 |")
    out.append("|---|---|---|---|---|")
    for row in summary["defects"]:
        lift = "∞" if math.isinf(row["lift"]) else f"{row['lift']:.2f}"
        out.append(
            f"| `{row['defect']}` | {row['failure_rate']:.1%} | {row['success_rate']:.1%} | "
            f"{lift} | {'✅' if row['discriminating'] else '❌'} |"
        )
    out.append("")
    addressable = summary["addressable"]
    if addressable["discriminating_defects"]:
        out.append(
            f"**可及空间（只算有区分度的缺陷）**：失败 {addressable['failures']} 题中 "
            f"**{addressable['with_discriminating_defect']} 题"
            f"（{addressable['share_of_failures']:.1%}）**带 "
            f"{'、'.join('`' + name + '`' for name in addressable['discriminating_defects'])}。"
        )
    else:
        out.append(
            "**可及空间 ≈ 0**：三个过程缺陷在成功组与失败组同样普遍 ⇒ "
            "**turn-level 动作形状不是本任务的靶点**，别把奖励加在这里。"
        )
    out.append("")

    out.append("## 4. 激活维度失分（奖励该奖励「核对什么」）")
    out.append("")
    out.append("> 只统计 `evidence.preference_scoring.dimensions` 里 `active=True` 的维度；"
        "未激活的维度恒为 0，不能当失分。")
    out.append("")
    out.append("| 维度 | 激活题数 | 全体均分 | 成功均分 | 失败均分 | 失败中为 0 |")
    out.append("|---|---|---|---|---|---|")
    for row in summary["dimensions"]:
        out.append(
            f"| `{row['dimension']}` | {row['active_tasks']} | {row['mean_score']:.3f} | "
            f"{row['success_mean_score']:.3f} | {row['failure_mean_score']:.3f} | "
            f"{row['failure_zero_rate']:.0%} |"
        )
    out.append("")

    out.append("## 5. 零梯度质量（同 reward_type 内奖励是否恒定）")
    out.append("")
    out.append("| reward_type | 题数 | 奖励区间 | 恒定？ | 均值 |")
    out.append("|---|---|---|---|---|")
    for row in summary["flat_reward_types"]:
        out.append(
            f"| `{row['reward_type']}` | {row['tasks']} | [{row['min']:.3f}, {row['max']:.3f}] | "
            f"{'**是**' if row['flat'] else '否'} | {row['mean']:.3f} |"
        )
    out.append("")
    out.append(
        f"**零梯度失败题 = {addressable['zero_gradient_failures']} 题"
        f"（占失败的 {addressable['zero_gradient_share_of_failures']:.1%}）**："
        "这些题的奖励在类型内是常数，无论策略做得多接近都拿同一个数 ⇒ GRPO 拿不到任何排序信息。"
        "这是 shaping 唯一真正有空间的地方。"
    )
    out.append("")

    out.append("## 6. 判读")
    out.append("")
    thresholds = summary["thresholds"]
    out.append(
        f"- 口径写死：连续 ≥{thresholds['repeat_run']} 次相同动作 = 循环；"
        f"打开 <{thresholds['comparison_opens']} 个商品 = 无比较；"
        f"`evidence_coverage` <{thresholds['low_evidence']} = 证据不足；"
        f"有区分度 = 失败率−成功率 ≥{thresholds['discrimination_gap']:.2f} 且 提升比 ≥{thresholds['discrimination_lift']:.1f}。"
    )
    out.append(
        "- 只有**有区分度**的缺陷才算可及空间；在成功题里同样普遍的「缺陷」是行为常态，"
        "拿它算空间会把全部失败都误算成可救。"
    )
    out.append(
        "- 若过程形状无区分度、而零梯度失败题占比高 ⇒ 该做的是**给失败档位补排序信息**"
        "（例如按买到商品的属性覆盖度给 soft 分），不是加 turn-level 惩罚。"
    )
    out.append("")
    return "\n".join(out)


def load_rows(eval_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """读评测目录下的 `trajectories.jsonl` 与 `summary.json`。"""
    traj_path = eval_dir / "trajectories.jsonl"
    if not traj_path.is_file():
        raise ValueError(f"找不到 {traj_path}")
    rows: list[dict[str, Any]] = []
    with traj_path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{traj_path}:{lineno} 不是合法 JSON：{exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    summary_path = eval_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                summary = loaded
        except (OSError, json.JSONDecodeError):
            summary = {}
    return rows, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("eval_dir", type=Path, help="评测目录（含 trajectories.jsonl）")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而非 markdown")
    parser.add_argument("--out", type=Path, default=None, help="把 markdown 报告写到该路径")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows, _ = load_rows(args.eval_dir)
    except (OSError, ValueError) as exc:
        print(f"读取失败：{exc}", file=sys.stderr)
        return 2
    if not rows:
        print(f"{args.eval_dir} 里没有轨迹", file=sys.stderr)
        return 2

    summary = analyze(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else render(summary))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render(summary), encoding="utf-8")
        print(f"\n报告已写入 {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
