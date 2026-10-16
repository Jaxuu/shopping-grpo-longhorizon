#!/usr/bin/env python3
"""分析 GRPO 训练诊断文件：量化"50 步里真正更新了几步"，以及动态采样为什么跳过。

用途
----
M6 第一轮 50 步里 `skipped_updates_total` 累积到 32+，意味着**真实参数更新只有约 18 步**。
本脚本从 `training_diagnostics.jsonl` 里把这件事量化清楚：

  1. 步数总览：完成更新的步 / 被跳过的步 / 跳过率
  2. 动态采样实测的组级有效率（与全池探针的预测值对比）
  3. 每步用掉几个 generation batch（用来判断 `max_num_gen_batches` 是不是卡在阈值上）
  4. 组被丢弃的原因分布（constant_reward / sampling_invalid / …）

只读文件，不加载 torch / veRL。

用法
----
    python scripts/analyze_grpo_diagnostics.py outputs/models/grpo-m6/training_diagnostics.jsonl
    python scripts/analyze_grpo_diagnostics.py <path> --json
    # 与探针结果对比（可选）
    python scripts/analyze_grpo_diagnostics.py <path> --probe-labels outputs/probe/m6r-full/difficulty_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("diagnostics", type=Path, help="training_diagnostics.jsonl 路径")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出（供报告引用）")
    parser.add_argument(
        "--probe-labels",
        type=Path,
        default=None,
        help="可选：全池探针的 difficulty_labels.jsonl，用于交叉核对有效率",
    )
    parser.add_argument(
        "--max-gen-batches",
        type=int,
        default=None,
        help="可选：本次训练配置的 shopping_dynamic_sampling.max_num_gen_batches",
    )
    return parser.parse_args()


def read_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"诊断文件不存在：{path}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc
    if not records:
        raise SystemExit(f"诊断文件为空：{path}")
    return records


def analyse(records: list[dict]) -> dict:
    by_event = Counter(record.get("event") for record in records)

    generation_batches: dict[int, list[dict]] = defaultdict(list)
    skipped_steps: list[int] = []
    updated_steps: set[int] = set()

    for record in records:
        event = record.get("event")
        step = record.get("global_step")
        if event == "generation_batch":
            generation_batches[int(step)].append(record)
        elif event == "skipped_update":
            skipped_steps.append(int(step))
        elif event == "optimizer_step":
            updated_steps.add(int(step))

    # 组级统计（动态采样的实测有效率）
    seen_groups = 0
    kept_groups = 0
    accepted_groups = 0
    drop_reasons: Counter[str] = Counter()
    per_batch_kept: list[float] = []
    per_batch_groups: list[int] = []

    for step, batches in generation_batches.items():
        for batch in batches:
            groups = batch.get("groups") or []
            if not groups:
                continue
            batch_kept = 0
            for group in groups:
                seen_groups += 1
                if group.get("kept"):
                    kept_groups += 1
                    batch_kept += 1
                if group.get("accepted"):
                    accepted_groups += 1
                reason = group.get("drop_reason")
                drop_reasons[str(reason) if reason else "kept"] += 1
            per_batch_groups.append(len(groups))
            per_batch_kept.append(batch_kept / len(groups))

    batches_per_step = {step: len(batches) for step, batches in generation_batches.items()}
    steps_with_generation = sorted(batches_per_step)

    summary: dict = {
        "records": len(records),
        "events": dict(sorted(by_event.items())),
        "steps": {
            "generation_steps": len(steps_with_generation),
            "optimizer_updated_steps": len(updated_steps),
            "skipped_steps": len(skipped_steps),
            "skip_rate": (
                len(skipped_steps) / (len(updated_steps) + len(skipped_steps))
                if (updated_steps or skipped_steps)
                else 0.0
            ),
            "first_skipped_step": min(skipped_steps) if skipped_steps else None,
            "last_skipped_step": max(skipped_steps) if skipped_steps else None,
        },
        "groups": {
            "seen": seen_groups,
            "kept": kept_groups,
            "accepted": accepted_groups,
            "kept_rate": kept_groups / seen_groups if seen_groups else 0.0,
            "accepted_rate": accepted_groups / seen_groups if seen_groups else 0.0,
            "drop_reason_counts": dict(sorted(drop_reasons.items())),
        },
        "generation_batches": {
            "total": sum(batches_per_step.values()),
            "per_step_counts": dict(sorted(Counter(batches_per_step.values()).items())),
            "mean_per_step": (
                statistics.mean(batches_per_step.values()) if batches_per_step else 0.0
            ),
            "max_per_step": max(batches_per_step.values()) if batches_per_step else 0,
            "mean_kept_rate_per_batch": (
                statistics.mean(per_batch_kept) if per_batch_kept else 0.0
            ),
            "mean_groups_per_batch": (
                statistics.mean(per_batch_groups) if per_batch_groups else 0.0
            ),
        },
    }
    return summary


def compare_with_probe(summary: dict, labels_path: Path) -> dict:
    rows = []
    with Path(labels_path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    total = len(rows)
    eligible = sum(1 for row in rows if row.get("eligible"))
    varying = sum(1 for row in rows if row.get("reward_varying"))
    return {
        "probe_tasks": total,
        "probe_reward_varying": varying,
        "probe_reward_varying_rate": varying / total if total else 0.0,
        "probe_eligible": eligible,
        "probe_eligible_rate": eligible / total if total else 0.0,
        "training_kept_rate": summary["groups"]["kept_rate"],
    }


def render(summary: dict, probe: dict | None, max_gen_batches: int | None) -> str:
    steps = summary["steps"]
    groups = summary["groups"]
    batches = summary["generation_batches"]

    total_steps = steps["optimizer_updated_steps"] + steps["skipped_steps"]
    lines = [
        "# GRPO 训练诊断分析",
        "",
        f"- 记录数：{summary['records']}；事件：{summary['events']}",
        "",
        "## 1. 步数总览（**最关键的一张表**）",
        "",
        "| 项 | 值 |",
        "|---|---:|",
        f"| 总步数 | {total_steps} |",
        f"| **完成 optimizer 更新的步** | **{steps['optimizer_updated_steps']}** |",
        f"| **被跳过的步** | **{steps['skipped_steps']}** |",
        f"| 跳过率 | {steps['skip_rate']:.1%} |",
        f"| 首个跳过步 / 末个跳过步 | {steps['first_skipped_step']} / {steps['last_skipped_step']} |",
        "",
        "> 被跳过的步**不产生任何梯度** —— 训练日志看起来一切正常，只有这个比值在涨。",
        "",
        "## 2. 动态采样实测的组级有效率",
        "",
        "| 项 | 值 |",
        "|---|---:|",
        f"| 生成过的组（prompt）数 | {groups['seen']} |",
        f"| kept（有梯度）组数 | {groups['kept']} |",
        f"| **实测 kept 率** | **{groups['kept_rate']:.1%}** |",
        f"| accepted（真正进 batch）组数 | {groups['accepted']} |",
        "",
        "丢弃原因分布：",
        "",
        "| 原因 | 组数 |",
        "|---|---:|",
    ]
    for reason, count in sorted(
        groups["drop_reason_counts"].items(), key=lambda item: -item[1]
    ):
        lines.append(f"| `{reason}` | {count} |")

    lines += [
        "",
        "## 3. 每步用掉几个 generation batch",
        "",
        f"- 合计 generation batch：{batches['total']}，平均 **{batches['mean_per_step']:.2f}** 个/步，最多 {batches['max_per_step']} 个",
        f"- 每个 batch 平均 {batches['mean_groups_per_batch']:.2f} 个组，组内 kept 率 {batches['mean_kept_rate_per_batch']:.1%}",
        "",
        "| 每步 batch 数 | 出现步数 |",
        "|---:|---:|",
    ]
    for count, steps_count in batches["per_step_counts"].items():
        lines.append(f"| {count} | {steps_count} |")

    if max_gen_batches:
        at_cap = batches["per_step_counts"].get(max_gen_batches, 0)
        lines += [
            "",
            f"⚠️ 配置 `max_num_gen_batches = {max_gen_batches}`：有 **{at_cap}** 步用满了上限",
            "（用满仍凑不满 → 该步被跳过）。用满的步数越多，说明上限越紧。",
        ]

    if probe:
        lines += [
            "",
            "## 4. 与全池探针的交叉核对",
            "",
            "| 项 | 探针（离线，n=4, T=0.7） | 训练（动态采样实测） |",
            "|---|---:|---:|",
            f"| reward_varying 率 | {probe['probe_reward_varying_rate']:.1%} | — |",
            f"| eligible（有梯度）率 | **{probe['probe_eligible_rate']:.1%}** | **{probe['training_kept_rate']:.1%}** |",
            f"| 题数 | {probe['probe_tasks']} | — |",
            "",
            "> 两者应当同量级。若训练侧明显更低，说明训练期的 rollout 路径与探针有工程差异"
            "（上下文处理、步数上限等），需要按训练侧的数字做决策。",
        ]

    lines += ["", "## 5. 判读", ""]
    skip_rate = steps["skip_rate"]
    at_cap_steps = (
        batches["per_step_counts"].get(max_gen_batches, 0) if max_gen_batches else 0
    )
    if skip_rate >= 0.3:
        lines += [
            f"- **跳过率 {skip_rate:.0%}** ⇒ 有相当比例的算力白烧（被跳过的步不产生任何梯度）。",
        ]
        if max_gen_batches and at_cap_steps:
            share = at_cap_steps / max(total_steps, 1)
            lines += [
                f"- 有 **{at_cap_steps}/{total_steps}（{share:.0%}）**步用满了 "
                f"`max_num_gen_batches = {max_gen_batches}` 仍凑不满。",
                "  ⇒ **上限太紧**：把它从 3 提到 8 即可让这些步完成。"
                "rollout 只占单步耗时的约 4%（第一轮实测 gen 37–45 s vs update_actor 313–745 s），"
                "所以这个改动的代价很小，**且不动训练数据**。",
            ]
        lines += [
            f"- 另一条路是**过滤训练集**（只留 kept 的题）：代价是训练池分布变化。"
            "但注意 —— 动态采样本来就只接受 kept 的组，**两条路给到梯度的是同一批题**，"
            "区别只在「要不要为退化题付 rollout 成本」。",
            "- **决策依据**：看 §2 的实测 kept 率与 §4 的探针对比。",
            "  · 训练侧 kept 率与探针同量级（~36%）⇒ 池子没问题，**改上限**即可；",
            "  · 训练侧明显更低 ⇒ 训练期 rollout 与探针有工程差异，**以训练侧数字为准**，"
            "此时过滤训练集更稳妥。",
        ]
    elif skip_rate >= 0.1:
        lines.append(
            f"- 跳过率 {skip_rate:.0%}：有浪费但不致命。优先看 §2/§4 判断池子构成是否是主因。"
        )
    else:
        lines.append(
            f"- 跳过率 {skip_rate:.0%} 很低，动态采样工作正常；瓶颈在别处（看池子构成与 lr）。"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    records = read_records(args.diagnostics)
    summary = analyse(records)
    probe = (
        compare_with_probe(summary, args.probe_labels) if args.probe_labels else None
    )
    if probe:
        summary["probe"] = probe
    if args.max_gen_batches:
        summary["configured_max_gen_batches"] = args.max_gen_batches

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render(summary, probe, args.max_gen_batches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
